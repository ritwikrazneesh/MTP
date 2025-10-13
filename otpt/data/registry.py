from typing import Tuple, List
from otpt.data.eurosat import make_loaders as eurosat_make_loaders, EUROSAT_CLASSES

_REGISTRY = {
    "eurosat": (eurosat_make_loaders, EUROSAT_CLASSES),
    # Add more datasets here as needed
}

def get_dataset(name: str, data_root: str, preprocess, batch_size: int, num_workers: int):
    name = name.lower()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(_REGISTRY.keys())}")
    make_loaders, classes = _REGISTRY[name]
    train_loader, val_loader = make_loaders(
        root=data_root,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return (train_loader, val_loader), classes
