"""Main CLI entry point for O-TPT evaluation."""

import argparse
import random
import numpy as np
import torch

from otpt.data.registry import get_dataset
from otpt.models.remoteclip_adapter import RemoteCLIPAdapter
from otpt.models.openclip_adapter import OpenCLIPAdapter
from otpt.models.otpt_core import OTPT
from otpt.eval.metrics import compute_metrics


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="O-TPT + RemoteCLIP CLI")
    parser.add_argument("--dataset", type=str, default="eurosat", 
                        help="Dataset to use")
    parser.add_argument("--backend", type=str, default="remoteclip",
                        choices=["remoteclip", "openclip"],
                        help="Model backend")
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["eval"],
                        help="Mode: eval")
    parser.add_argument("--otpt-lr", type=float, default=0.001,
                        help="O-TPT learning rate")
    parser.add_argument("--otpt-steps", type=int, default=1,
                        help="O-TPT optimization steps per batch")
    parser.add_argument("--entropy-threshold", type=float, default=0.6,
                        help="Confidence threshold for entropy filtering")
    parser.add_argument("--orth-reg", type=float, default=0.1,
                        help="Orthogonality regularization weight")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    train_loader, val_loader, class_names = get_dataset(
        args.dataset, 
        batch_size=args.batch_size
    )
    print(f"Classes: {len(class_names)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Load model
    print(f"\nLoading model backend: {args.backend}")
    if args.backend == "remoteclip":
        model = RemoteCLIPAdapter(class_names, device)
    else:
        model = OpenCLIPAdapter(class_names, device)
    
    # Setup O-TPT
    print(f"\nInitializing O-TPT:")
    print(f"  Learning rate: {args.otpt_lr}")
    print(f"  Steps per batch: {args.otpt_steps}")
    print(f"  Entropy threshold: {args.entropy_threshold}")
    print(f"  Orthogonality reg: {args.orth_reg}")
    
    otpt = OTPT(
        model=model,
        lr=args.otpt_lr,
        steps=args.otpt_steps,
        entropy_threshold=args.entropy_threshold,
        orth_reg=args.orth_reg,
        device=device
    )
    
    # Run evaluation
    print(f"\n{'='*60}")
    print("Running evaluation on validation set")
    print('='*60)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Run O-TPT
        probs = otpt.forward(images)
        preds = probs.argmax(dim=1)
        
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    # Concatenate results
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Results")
    print('='*60)
    
    metrics = compute_metrics(all_probs, all_labels, all_preds)
    
    print(f"Top-1 Accuracy:     {metrics['top1_accuracy']:.4f}")
    print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"NLL:                {metrics['nll']:.4f}")
    print(f"ECE:                {metrics['ece']:.4f}")
    print('='*60)


if __name__ == "__main__":
    main()
