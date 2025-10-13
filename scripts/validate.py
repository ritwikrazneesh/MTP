#!/usr/bin/env python3
"""
Validation script to test O-TPT + RemoteCLIP scaffold.
This script verifies that all components work correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from otpt.data.registry import get_dataset
        from otpt.models.remoteclip_adapter import RemoteCLIPAdapter
        from otpt.models.openclip_adapter import OpenCLIPAdapter
        from otpt.models.otpt_core import OTPT
        from otpt.eval.metrics import compute_metrics
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    try:
        from otpt.data.registry import get_dataset
        train_loader, val_loader, class_names = get_dataset('eurosat', batch_size=32)
        print(f"✓ Dataset loaded: {len(class_names)} classes, {len(val_loader.dataset)} val samples")
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_model():
    """Test model initialization."""
    print("\nTesting model initialization...")
    try:
        from otpt.models.openclip_adapter import OpenCLIPAdapter
        device = torch.device('cpu')
        class_names = ['class1', 'class2', 'class3']
        model = OpenCLIPAdapter(class_names, device)
        print("✓ Model initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False


def test_metrics():
    """Test metrics computation."""
    print("\nTesting metrics computation...")
    try:
        from otpt.eval.metrics import compute_metrics
        # Create dummy predictions and labels
        probs = torch.randn(100, 10).softmax(dim=1)
        labels = torch.randint(0, 10, (100,))
        preds = probs.argmax(dim=1)
        
        metrics = compute_metrics(probs, labels, preds)
        print(f"✓ Metrics computed: Top-1={metrics['top1_accuracy']:.4f}, NLL={metrics['nll']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Metrics computation failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("O-TPT + RemoteCLIP Validation")
    print("="*60)
    
    tests = [
        test_imports,
        test_dataset,
        test_model,
        test_metrics,
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
