"""EuroSAT dataset loader with stratified 80/20 split."""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


# EuroSAT class names
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


def get_eurosat_loaders(batch_size=64, data_root="./data", num_workers=4):
    """
    Get EuroSAT data loaders with stratified 80/20 train/val split.
    
    Args:
        batch_size: Batch size for data loaders
        data_root: Root directory for downloading/loading data
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
    """
    # Setup transforms for CLIP models (224x224, ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # Download/load EuroSAT dataset
    data_path = Path(data_root) / "eurosat"
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Try to load dataset
    try:
        # EuroSAT is available via torchvision in ImageFolder format
        # We'll use a custom download or assume it's in the correct format
        full_dataset = datasets.ImageFolder(
            root=str(data_path / "2750"),
            transform=transform
        )
    except Exception as e:
        print(f"Note: EuroSAT dataset not found at {data_path}")
        print("Please download EuroSAT from: https://github.com/phelber/EuroSAT")
        print("Extract the '2750' folder to: ./data/eurosat/")
        print("\nFor testing purposes, creating a mock dataset...")
        
        # Create mock dataset for testing
        full_dataset = create_mock_eurosat(transform)
    
    # Get labels for stratified split
    targets = [label for _, label in full_dataset]
    indices = list(range(len(full_dataset)))
    
    # Stratified 80/20 split
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=targets,
        random_state=42
    )
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, EUROSAT_CLASSES


def create_mock_eurosat(transform):
    """Create a mock EuroSAT dataset for testing purposes."""
    from torch.utils.data import TensorDataset
    import numpy as np
    
    print("Creating mock dataset with 1000 samples (100 per class)...")
    
    # Create random images
    num_samples = 1000
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create random RGB image
        img = torch.randint(0, 256, (3, 224, 224), dtype=torch.uint8).float() / 255.0
        # Apply normalization from transform
        img = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )(img)
        images.append(img)
        labels.append(i % 10)  # 10 classes
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    class MockDataset:
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
            
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]
    
    return MockDataset(images, labels)
