# Swin Transformer V2 Weather Classification

Simple inference script for a Swin-Transformer V2 image-classification model.

---

## 1. Environment setup

```bash
# Create and activate a Python virtual environment (Python ≥3.9)
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Project structure

```text
swin-transformer-cls/
├── inference.py        # main inference script
├── requirements.txt    # Python dependencies
├── models/             # put your *.pth / *.safetensors checkpoints here
└── output/             # images will be copied into label-named sub-folders after inference
```

Create the folders if they do not exist:

```bash
mkdir -p models output
```

> ❗ Place your trained checkpoint (e.g. `best_model.pth`) in the `models/` directory.

## 3. Usage examples

### Single image

```bash
python inference.py \
  --ckpt models/best_model.pth \
  --image data/sample/img.jpg \
  --device cuda   # or cpu
```

### Batch inference (folder)

```bash
python inference.py \
  --ckpt models/best_model.pth \
  --image_dir data/sample \
  --topk 3 \
  --device cuda
```

The script will print top-k predictions for each image and, if `--output_dir` is left as default (`output/`), copy each image into `output/<predicted_label>/` for easy inspection.

---

## 4. Troubleshooting

* **Classifier shape mismatch** – ensure `--num_classes` matches the number of classes your model was trained on if automatic detection fails.
* **CUDA not available** – pass `--device cpu` to run on CPU.

---

## 5. License

This project is released under the MIT License.
