#!/bin/bash
# Example script to run O-TPT evaluation on EuroSAT

echo "Running O-TPT + RemoteCLIP on EuroSAT..."
echo ""

# Run with RemoteCLIP backend
python -m otpt.cli \
    --dataset eurosat \
    --backend remoteclip \
    --mode eval \
    --otpt-lr 0.001 \
    --otpt-steps 1 \
    --entropy-threshold 0.6 \
    --orth-reg 0.1 \
    --batch-size 64 \
    --seed 42

echo ""
echo "Evaluation complete!"
