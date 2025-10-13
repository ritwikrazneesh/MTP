# O-TPT + RemoteCLIP for Remote Sensing

This repository provides a Kaggle-ready implementation of **O-TPT** (Online Test-Time Prompt Tuning) with **RemoteCLIP** for remote sensing image classification.

## Features

- **O-TPT**: Entropy minimization on confident samples with orthogonality regularization
- **RemoteCLIP**: Remote sensing-specific CLIP model via open-clip checkpoints
- **EuroSAT Dataset**: RGB, single-label land use classification with stratified 80/20 split
- **Metrics**: Top-1 accuracy, Balanced accuracy, NLL, and ECE

## Quickstart

### Installation

```bash
pip install -r requirements.txt
```

### Run Evaluation on EuroSAT

```bash
# Using RemoteCLIP backend
python -m otpt.cli --dataset eurosat --backend remoteclip --mode eval

# Using OpenCLIP fallback
python -m otpt.cli --dataset eurosat --backend openclip --mode eval

# With custom O-TPT hyperparameters
python -m otpt.cli --dataset eurosat --backend remoteclip --mode eval \
  --otpt-lr 0.001 --otpt-steps 1 --entropy-threshold 0.6 --orth-reg 0.1
```

### Using the Shell Script

```bash
bash scripts/run_eurosat.sh
```

### Validation

To verify the installation and setup:

```bash
python scripts/validate.py
```

## CLI Usage

```
python -m otpt.cli [OPTIONS]

Options:
  --dataset STR          Dataset to use (default: eurosat)
  --backend STR          Model backend: remoteclip or openclip (default: remoteclip)
  --mode STR             Mode: eval (default: eval)
  --otpt-lr FLOAT        O-TPT learning rate (default: 0.001)
  --otpt-steps INT       O-TPT optimization steps per batch (default: 1)
  --entropy-threshold FLOAT  Confidence threshold for entropy filtering (default: 0.6)
  --orth-reg FLOAT       Orthogonality regularization weight (default: 0.1)
  --batch-size INT       Batch size (default: 64)
  --seed INT             Random seed (default: 42)
```

## Project Structure

```
otpt/
├── cli.py                      # Main CLI entry point
├── data/
│   ├── registry.py            # Dataset registry
│   └── eurosat.py             # EuroSAT dataset loader
├── models/
│   ├── remoteclip_adapter.py  # RemoteCLIP adapter
│   ├── openclip_adapter.py    # OpenCLIP fallback
│   └── otpt_core.py           # O-TPT implementation
└── eval/
    └── metrics.py             # Evaluation metrics

scripts/
└── run_eurosat.sh             # Example evaluation script
```

## References

- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)
- [O-TPT Paper](https://arxiv.org/abs/2209.07511)
- [EuroSAT Dataset](https://github.com/phelber/EuroSAT)