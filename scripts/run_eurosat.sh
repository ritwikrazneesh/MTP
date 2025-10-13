#!/usr/bin/env bash
set -e

# Zero-shot with RemoteCLIP checkpoint
python -m otpt.cli \
  --dataset eurosat \
  --backend remoteclip \
  --model-name ViT-B-32 \
  --pretrained-ckpt /path/to/remoteclip_vitb32.pt \
  --mode zeroshot \
  --batch-size 128

# O-TPT with RemoteCLIP checkpoint
python -m otpt.cli \
  --dataset eurosat \
  --backend remoteclip \
  --model-name ViT-B-32 \
  --pretrained-ckpt /path/to/remoteclip_vitb32.pt \
  --mode otpt \
  --batch-size 64 \
  --n-ctx 8 \
  --tta-steps 2 \
  --lambda-orth 0.1 \
  --selection-p 0.1
