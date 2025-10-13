"""Evaluation metrics for classification."""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def compute_top1_accuracy(preds, labels):
    """
    Compute top-1 accuracy.
    
    Args:
        preds: Predicted class indices [N]
        labels: Ground truth labels [N]
        
    Returns:
        accuracy: Top-1 accuracy
    """
    correct = (preds == labels).sum().item()
    total = len(labels)
    return correct / total


def compute_balanced_accuracy(preds, labels):
    """
    Compute balanced accuracy (handles class imbalance).
    
    Args:
        preds: Predicted class indices [N]
        labels: Ground truth labels [N]
        
    Returns:
        balanced_acc: Balanced accuracy
    """
    preds_np = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    
    return balanced_accuracy_score(labels_np, preds_np)


def compute_nll(probs, labels):
    """
    Compute negative log-likelihood.
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: Ground truth labels [N]
        
    Returns:
        nll: Negative log-likelihood
    """
    # Get probabilities for true class
    true_probs = probs[range(len(labels)), labels]
    
    # Compute NLL
    nll = -torch.log(true_probs + 1e-10).mean().item()
    
    return nll


def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: Ground truth labels [N]
        n_bins: Number of bins for calibration
        
    Returns:
        ece: Expected Calibration Error
    """
    # Get confidence (max probability) and predictions
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()
    
    # Convert to numpy for binning
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Compute average confidence and accuracy in this bin
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def compute_metrics(probs, labels, preds=None):
    """
    Compute all evaluation metrics.
    
    Args:
        probs: Predicted probabilities [N, num_classes]
        labels: Ground truth labels [N]
        preds: Predicted class indices [N] (optional, computed from probs if not provided)
        
    Returns:
        metrics: Dictionary of metrics
    """
    if preds is None:
        preds = probs.argmax(dim=1)
    
    metrics = {
        'top1_accuracy': compute_top1_accuracy(preds, labels),
        'balanced_accuracy': compute_balanced_accuracy(preds, labels),
        'nll': compute_nll(probs, labels),
        'ece': compute_ece(probs, labels),
    }
    
    return metrics
