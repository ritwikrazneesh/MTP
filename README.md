# O‑TPT on RemoteCLIP/OpenCLIP for Remote Sensing (Single Entry CLI)

Single-entry CLI for zero‑shot and label‑free O‑TPT (Orthogonality‑constrained Test‑time Prompt Tuning) in the remote sensing domain.

- Backends:
  - remoteclip: load RemoteCLIP checkpoints via open-clip-torch (per RemoteCLIP README)
  - openclip: OpenCLIP pretrained IDs (fallback)
- Dataset: EuroSAT (RGB, single-label)
- Metrics: Top‑1, Balanced Accuracy, NLL, ECE

## Install

```bash
pip install -r requirements.txt
```

## RemoteCLIP (per README)

1) Install open-clip:
```bash
pip install open-clip-torch
```

2) Download a RemoteCLIP checkpoint (e.g., `remoteclip_vitb32.pt`) and pass its path via `--pretrained-ckpt`.

## Quickstart (EuroSAT)

Zero-shot with RemoteCLIP checkpoint:
```bash
python -m otpt.cli \
  --dataset eurosat \
  --backend remoteclip \
  --model-name ViT-B-32 \
  --pretrained-ckpt /path/to/remoteclip_vitb32.pt \
  --mode zeroshot \
  --data-root ./data \
  --batch-size 128
```

O‑TPT (label‑free):
```bash
python -m otpt.cli \
  --dataset eurosat \
  --backend remoteclip \
  --model-name ViT-B-32 \
  --pretrained-ckpt /path/to/remoteclip_vitb32.pt \
  --mode otpt \
  --data-root ./data \
  --batch-size 64 \
  --n-ctx 8 \
  --tta-steps 2 \
  --lambda-orth 0.1 \
  --selection-p 0.1
```

OpenCLIP fallback:
```bash
python -m otpt.cli \
  --dataset eurosat \
  --backend openclip \
  --model-name ViT-B-16 \
  --pretrained-id laion2b_s34b_b88k \
  --mode otpt
```

## Structure

- otpt/
  - cli.py (single entry point; choose dataset/backend/mode via args)
  - data/
    - registry.py (dataset registry)
    - eurosat.py (torchvision EuroSAT loader)
  - models/
    - remoteclip_adapter.py (RemoteCLIP via open-clip checkpoint)
    - openclip_adapter.py (OpenCLIP fallback)
    - otpt_core.py (entropy, orthogonality, TTA loop, AMP-safe)
  - eval/
    - metrics.py (ECE)
- scripts/
  - run_eurosat.sh (example runs)

## Notes

- Template defaults to "a satellite photo of a {}". You can customize with `--template`.
- We reset the prompt context per batch for stability at test time.
- CPU/GPU compatible: AMP is enabled only on CUDA automatically.
- Kaggle: clone the repo, install requirements, upload checkpoint to an accessible path, and run the CLI.
