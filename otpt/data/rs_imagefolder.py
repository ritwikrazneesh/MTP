from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from torchvision import datasets
from torch.utils.data import DataLoader
import difflib

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
    # normalize folder-style names to human phrases
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

# ---------- Canonical mapping utilities ----------
from .canonical_rs_classes import CANONICAL_BY_DATASET

# Common synonym normalization before fuzzy matching
_REPLACEMENTS: Dict[str, str] = {
    "residential buildings": "residential",
    "residential area": "residential",
    "sparse residence": "sparse residential",
    "medium residence": "medium residential",
    "dense residence": "dense residential",
    "parkinglot": "parking lot",
    "parking area": "parking lot",
    "tennis-court": "tennis court",
    "baseball field": "baseball diamond",  # often named differently
    "basketball-court": "basketball court",
    "railwaystation": "railway station",
    "railway station": "railway station",
    "circular farmland": "circular farmland",
    "rectangular farmland": "rectangular farmland",
    "harbour": "harbor",
    "sea/lake": "sea lake",
    "sea_lake": "sea lake",
    "highway or road": "highway",
    "industrial buildings": "industrial",
    "industrial area": "industrial area",
    "football field": "football field",
    "soccer field": "football field",
    "snowy mountain": "snowberg",
}

def _normalize(s: str) -> str:
    s2 = s.lower().strip().replace("_", " ").replace("-", " ")
    s2 = " ".join(s2.split())
    return _REPLACEMENTS.get(s2, s2)

def _fuzzy_map_one(name: str, candidates: List[str]) -> str:
    n = _normalize(name)
    # exact first
    for c in candidates:
        if _normalize(c) == n:
            return c
    # fuzzy best
    choices = [_normalize(c) for c in candidates]
    best = difflib.get_close_matches(n, choices, n=1, cutoff=0.5)
    if best:
        idx = choices.index(best[0])
        return candidates[idx]
    # fallback: return normalized as-is
    return n

def map_folder_classes_to_canonical(folder_classes: List[str], dataset_name: str) -> List[str]:
    """
    Returns a list of canonical phrases aligned to the folder class order.
    Each returned item is a canonical human phrase to use in text prompts.
    """
    canon = CANONICAL_BY_DATASET.get(dataset_name.lower())
    if not canon:
        # If we don't have a curated list, just normalize folder names
        return [_normalize(c) for c in folder_classes]
    return [_fuzzy_map_one(c, canon) for c in folder_classes]

def build_imagefolder_eval(
    dataset_name: str,
    data_root: str,
    preprocess,
    batch_size: int,
    num_workers: int,
    use_canonical: bool = True,
) -> Tuple[Tuple[Optional[DataLoader], DataLoader], List[str]]:
    """
    Returns (None, val_loader) and a classnames list for prompts.
    The classnames are in the same order as the dataset's label indices.
    If use_canonical=True, we map folder class names to canonical phrases per dataset.
    """
    eval_root = resolve_eval_root(data_root, dataset_name)
    ds = datasets.ImageFolder(str(eval_root), transform=preprocess)

    folder_classes = ds.classes  # order matches class_to_idx / labels
    if use_canonical and dataset_name.lower() in SUPPORTED_RS_DATASETS:
        classnames = map_folder_classes_to_canonical(folder_classes, dataset_name)
    else:
        classnames = [_canonicalize_classname(c) for c in folder_classes]

    val_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return (None, val_loader), classnames
