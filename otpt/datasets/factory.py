from __future__ import annotations
from typing import Tuple, List, Any
from .rs_imagefolder import KNOWN_IMAGEFOLDER_DATASETS, build_imagefolder_dataset

def build_dataset_and_classnames(dataset_name: str, preprocess: Any) -> Tuple[Any, List[str]]:
    """
    Unified dataset factory for otpt.cli.
    - Handles: eurosat, patternnet, nwpu-resisc45, ucm, whu-rs19, aid
    - All are treated as ImageFolder-style datasets (class subfolders).
    - If you later add others, extend KNOWN_IMAGEFOLDER_DATASETS or branch here.
    """
    ds_key = dataset_name.lower()
    if ds_key in KNOWN_IMAGEFOLDER_DATASETS:
        return build_imagefolder_dataset(ds_key, preprocess)

    supported = ", ".join(sorted(KNOWN_IMAGEFOLDER_DATASETS.keys()))
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
