from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple
from torchvision import datasets

# All supported single-label datasets you want to call via --dataset
# We include 'eurosat' here so it is handled exactly like the others.
KNOWN_IMAGEFOLDER_DATASETS = {
    "eurosat": "EUROSAT_ROOT",
    "patternnet": "PATTERNNET_ROOT",
    "nwpu-resisc45": "NWPU_RESISC45_ROOT",
    "ucm": "UCM_ROOT",
    "whu-rs19": "WHURS19_ROOT",
    "aid": "AID_ROOT",
}

def _has_class_subdirs(p: Path) -> bool:
    try:
        return any(c.is_dir() for c in p.iterdir())
    except Exception:
        return False

def _canonicalize_classname(name: str) -> str:
    # "mobile_home_park" -> "mobile home park"
    return name.replace("_", " ").replace("-", " ").strip()

def _resolve_eval_root(dataset_key: str) -> Path:
    """
    Resolution priority (no new CLI flags, only env vars):
    1) DATASET_ROOT env for that dataset (e.g., AID_ROOT) pointing either to:
       - a folder with class subdirs (no split), or
       - a folder containing a 'test' subfolder with class subdirs.
    2) Fallback: OTPT_DATASETS_BASE/<dataset_key>/test or OTPT_DATASETS_BASE/<dataset_key>
    """
    env_key = KNOWN_IMAGEFOLDER_DATASETS[dataset_key]
    env_root = os.getenv(env_key)
    if env_root:
        env_path = Path(env_root).expanduser()
        if (env_path / "test").exists() and _has_class_subdirs(env_path / "test"):
            return env_path / "test"
        if env_path.exists() and _has_class_subdirs(env_path):
            return env_path

    base = os.getenv("OTPT_DATASETS_BASE", "")
    if base:
        base_path = Path(base).expanduser() / dataset_key
        if (base_path / "test").exists() and _has_class_subdirs(base_path / "test"):
            return base_path / "test"
        if base_path.exists() and _has_class_subdirs(base_path):
            return base_path

    raise FileNotFoundError(
        f"Could not locate eval root for dataset '{dataset_key}'. "
        f"Set {env_key} to the dataset folder (with class subfolders), "
        f"or set OTPT_DATASETS_BASE to a directory that contains '{dataset_key}/' (optionally with a 'test/' subfolder)."
    )

def build_imagefolder_dataset(
    dataset_key: str,
    preprocess,  # pass in your existing backend/model preprocess
) -> Tuple[datasets.ImageFolder, List[str]]:
    """
    Returns:
      - torchvision.datasets.ImageFolder for the evaluation set
      - normalized classnames list (used by your prompt builder)
    """
    eval_root = _resolve_eval_root(dataset_key)
    ds = datasets.ImageFolder(str(eval_root), transform=preprocess)
    classnames = [_canonicalize_classname(c) for c in ds.classes]
    return ds, classnames
