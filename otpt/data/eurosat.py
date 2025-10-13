from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit

EUROSAT_CLASSES = [
    "annual_crop",
    "forest",
    "herbaceous_vegetation",
    "highway",
    "industrial",
    "pasture",
    "permanent_crop",
    "residential",
    "river",
    "sea_lake",
]

def _get_targets(ds):
    # Be robust across torchvision versions
    if hasattr(ds, "targets"):
        return ds.targets
    if hasattr(ds, "labels"):
        return ds.labels
    if hasattr(ds, "_labels"):
        return ds._labels
    raise AttributeError("EuroSAT dataset has no targets/labels attributes.")

def make_loaders(root: str, preprocess, batch_size: int = 128, num_workers: int = 2, val_ratio: float = 0.2, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    ds = datasets.EuroSAT(root=root, download=True, transform=preprocess)
    targets = _get_targets(ds)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx_train, idx_val = next(splitter.split(range(len(ds)), targets))
    train = torch.utils.data.Subset(ds, idx_train)
    val = torch.utils.data.Subset(ds, idx_val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
