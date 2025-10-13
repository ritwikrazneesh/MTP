"""Dataset registry for loading datasets."""

from otpt.data.eurosat import get_eurosat_loaders


DATASET_REGISTRY = {
    "eurosat": get_eurosat_loaders,
}


def get_dataset(dataset_name, batch_size=64):
    """
    Get dataset loaders and class names.
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size for data loaders
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")
    
    loader_fn = DATASET_REGISTRY[dataset_name]
    return loader_fn(batch_size=batch_size)
