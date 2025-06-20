#!/usr/bin/env python
"""Inference script for Swin Transformer V2 classifier.

Usage examples
--------------
Single image:
    python inference.py --ckpt models/best_model.pth --image data/sample/img.jpg

Batch (folder) inference:
    python inference.py --ckpt models/best_model.pth --image_dir data/sample --topk 3 --device cuda

The script automatically infers the number of classes from the checkpoint and
restores any saved ``class_to_idx`` mapping for human-readable labels.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from timm import create_model
import shutil
from torchvision import transforms

# -----------------------------------------------------------------------------
# Image preprocessing
# -----------------------------------------------------------------------------
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
_preprocess = transforms.Compose(
    [
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
)


def _load_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[Dict[str, torch.Tensor], List[str]]:
    """Load checkpoint returning state_dict and idx_to_class list (best effort)."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Detect state_dict
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Assume whole file is state dict
        state_dict = ckpt

    # Try to recover label mapping
    idx_to_class: list[str] | None = None
    if "class_to_idx" in ckpt:
        class_to_idx: Dict[str, int] = ckpt["class_to_idx"]
        idx_to_class = [None] * len(class_to_idx)
        for cls, idx in class_to_idx.items():
            idx_to_class[idx] = cls
    elif "idx_to_class" in ckpt:
        idx_to_class = ckpt["idx_to_class"]
    elif "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
        idx_to_class = list(ckpt["classes"])

    return state_dict, idx_to_class or []


@torch.no_grad()
def _predict_one_image(img_path: str | Path, model: torch.nn.Module, device: torch.device, topk: int,
                       idx_to_class: list[str]) -> List[tuple[str, float]]:
    """Return top-k predictions as (label, prob) tuples."""
    img = Image.open(img_path).convert("RGB")
    tensor = _preprocess(img).unsqueeze(0).to(device)
    output = model(tensor)
    # HuggingFace models return an object with `.logits`
    logits = output.logits if hasattr(output, "logits") else output
    probs = F.softmax(logits, dim=1)[0]
    topk_prob, topk_idx = probs.topk(topk)
    results = []
    for p, idx in zip(topk_prob.tolist(), topk_idx.tolist()):
        label = idx_to_class[idx] if idx < len(idx_to_class) and idx_to_class[idx] is not None else str(idx)
        results.append((label, p))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Swin Transformer V2 inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image")
    group.add_argument("--image_dir", type=str, help="Directory containing images")
    parser.add_argument("--topk", type=int, default=1, help="Top-k predictions to show")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_classes", type=int, default=None, help="Manually specify number of classes if it cannot be auto-inferred")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to copy images by predicted label")
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

    state_dict, idx_to_class = _load_checkpoint(ckpt_path, device)

        # ------------------------------------------------------------------
    # Infer number of classes
    # ------------------------------------------------------------------
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        try:
            # Prefer classifier/head weight in checkpoint for class count detection
            weight_key = None
            for cand in ("classifier.weight", "head.weight"):
                matches = [k for k in state_dict.keys() if k.endswith(cand)]
                if matches:
                    weight_key = matches[0]
                    break
            if weight_key is None:
                raise StopIteration()
            num_classes = state_dict[weight_key].size(0)
        except StopIteration:
            # Try other heuristics
            if idx_to_class:
                num_classes = len(idx_to_class)
            else:
                # pick first 2-D weight tensor (linear layer) as guess
                linear_weights = [v for v in state_dict.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
                if not linear_weights:
                    raise ValueError("Cannot infer number of classes from checkpoint. Pass --num_classes explicitly.")
                num_classes = linear_weights[0].size(0)


        # Build model. Try timm first, else fallback to Hugging Face implementation used during training.
    try:
        model = create_model("swinv2_large", pretrained=False, num_classes=num_classes)
    except RuntimeError:
        print("timm에서 'swinv2_large' 모델을 지원하지 않습니다. Hugging Face 모델로 전환합니다.")
        from transformers import AutoModelForImageClassification  # local import to reduce startup cost when not needed
        model = AutoModelForImageClassification.from_pretrained(
            "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[Warning] Unexpected keys in state_dict (showing first 5): {unexpected[:5]}")
    if missing:
        print(f"[Warning] Missing keys in state_dict (showing first 5): {missing[:5]}")
    model.eval().to(device)

    # Collect image paths
    if args.image:
        images = [Path(args.image)]
    else:
        images = [p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        images.sort()

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        preds = _predict_one_image(img_path, model, device, args.topk, idx_to_class)
        print(f"\n{img_path.name}:")
        for label, prob in preds:
                        print(f"  {label:<20s} {prob * 100:5.2f}%")

        # Copy to label folder using top-1 prediction
        top_label = preds[0][0]
        dest_dir = output_root / top_label
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest_dir / img_path.name)


if __name__ == "__main__":
    main()
