import os
import sys
from typing import Any

_DEBUG = os.getenv("OTPT_DEBUG", "0") == "1"

def set_debug(flag: bool) -> None:
    global _DEBUG
    _DEBUG = bool(flag)

def is_debug() -> bool:
    return _DEBUG

def log(*args: Any, **kwargs: Any) -> None:
    if _DEBUG:
        print(*args, **kwargs)
        sys.stdout.flush()
