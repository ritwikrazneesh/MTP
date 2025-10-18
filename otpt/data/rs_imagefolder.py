from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
from torchvision import datasets
from torch.utils.data import DataLoader

# All treated as single-label ImageFolder datasets (class subdirectories)
SUPPORTED_RS_DATASETS = {
    "eurosat",
    "patternnet",
    "nwpu-resisc45",
    "ucm",
    "whu-rs19",
    "aid",
}

def _has_class_subdirs(p: Path) -> bool:
    try:
        return p.exists() and any(c.is_dir() for c in p.iterdir())
    except Exception:
        return False

def _canonicalize_classname(name: str) -> str:
    # "mobile_home_park" -> "mobile home park", "baseball-diamond" -> "baseball diamond"
    return name.replace("_", " ").replace("-", " ").strip()

def _casefold_match(base: Path, key: str) -> Optional[Path]:
    # Case-insensitive match for a child dir named like the dataset
    if not base.exists():
        return None
    key_cf = key.casefold()
    for child in base.iterdir():
        if child.is_dir() and child.name.casefold() == key_cf:
            return child
    return None

def resolve_eval_root(data_root: str, dataset_name: str) -> Path:
    """
    Resolution priority:
    1) If data_root itself has class subfolders, use it directly (no split).
    2) Else look for a child folder named like dataset_name (case-insensitive).
       - If that folder has 'test/' with class subfolders, use 'test/'
       - Else use the folder itself.
    """
    base = Path(data_root).expanduser()
    if _has_class_subdirs(base):
        return base

    cand = _casefold_match(base, dataset_name)
    if cand is None:
        # Try simple normalizations as fallback
        for v in (dataset_name, dataset_name.lower(), dataset_name.upper(),
                  dataset_name.replace("-", "_"), dataset_name.replace("_", "-")):
            p = base / v
            if p.exists() and p.is_dir():
                cand = p
                break

    if cand is None:
        raise FileNotFoundError(
            f"Could not locate dataset directory for '{dataset_name}' under '{data_root}'. "
            f"Either point --data-root directly to the dataset folder (with class subfolders), "
            f"or ensure a subfolder named like '{dataset_name}' exists there."
        )

    test_dir = cand / "test"
    if test_dir.exists() and _has_class_subdirs(test_dir):
        return test_dir
    return cand

def build_imagefolder_eval(
    dataset_name: str,
    data_root: str,
    preprocess,
    batch_size: int,
    num_workers: int,
) -> Tuple[Tuple[Optional[DataLoader], DataLoader], List[str]]:
    """
    Returns (None, val_loader) and a normalized classnames list.
    This matches your CLI usage: _, val_loader = loaders
    """
    eval_root = resolve_eval_root(data_root, dataset_name)
    ds = datasets.ImageFolder(str(eval_root), transform=preprocess)
    classnames = [_canonicalize_classname(c) for c in ds.classes]
    val_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return (None, val_loader), classnames
