from __future__ import annotations
from typing import Tuple, List, Any, Optional
from torch.utils.data import DataLoader

from .rs_imagefolder import SUPPORTED_RS_DATASETS, build_imagefolder_eval

def get_dataset(
    name: str,
    data_root: str,
    preprocess: Any,
    batch_size: int,
    num_workers: int,
) -> Tuple[Tuple[Optional[DataLoader], DataLoader], List[str]]:
    """
    Unified dataset factory for single-label RS datasets.

    Supported (via ImageFolder with class subfolders, with or without 'test/'):
      - eurosat
      - patternnet
      - nwpu-resisc45
      - ucm
      - whu-rs19
      - aid

    Behavior:
      - If --data-root points directly to a folder with class subfolders, that folder is used.
      - Else it looks under --data-root for a subfolder named like the dataset (case-insensitive).
      - If a 'test/' subfolder with class subfolders exists, it is preferred; otherwise the dataset folder is used.

    Returns:
      - (None, val_loader), classnames
    """
    ds_key = name.lower()
    if ds_key in SUPPORTED_RS_DATASETS:
        return build_imagefolder_eval(
            dataset_name=ds_key,
            data_root=data_root,
            preprocess=preprocess,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    supported = ", ".join(sorted(SUPPORTED_RS_DATASETS))
    raise ValueError(f"Unsupported dataset '{name}'. Supported: {supported}")
