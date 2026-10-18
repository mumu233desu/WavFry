"""
Unified Cross-Platform Corpus Path Resolver (wavfry.data.path_resolver)

Seamlessly adapts corpus audio and TextGrid file paths across Windows local environments
(e.g., E:/Corpus/...) and Linux cloud environments (e.g., /data/Corpus/... or /Corpus/...).
"""

import os
from pathlib import Path
from typing import Union, Optional

_CACHED_CORPUS_ROOT: Optional[str] = None


def get_active_corpus_root() -> Optional[str]:
    """
    Detects or retrieves the active root directory for corpus files.
    Priority:
      1. WAVFRY_CORPUS_ROOT environment variable (if set and non-empty)
      2. If on Linux/POSIX: /data/Corpus -> /Corpus
      3. If on Windows: E:/Corpus
    """
    global _CACHED_CORPUS_ROOT
    if _CACHED_CORPUS_ROOT is not None:
        return _CACHED_CORPUS_ROOT

    env_root = os.environ.get("WAVFRY_CORPUS_ROOT")
    if env_root and env_root.strip():
        _CACHED_CORPUS_ROOT = env_root.strip().replace("\\", "/").rstrip("/")
        return _CACHED_CORPUS_ROOT

    if os.name == "nt":
        candidates = ("E:/Corpus", "/data/Corpus", "/Corpus")
    else:
        candidates = ("/data/Corpus", "/Corpus", "E:/Corpus")

    for candidate in candidates:
        if os.path.exists(candidate):
            _CACHED_CORPUS_ROOT = candidate
            return _CACHED_CORPUS_ROOT

    return None


def resolve_corpus_path(path: Union[str, Path, None]) -> Optional[str]:
    """
    Converts a dataset path into the active platform's valid path.
    - Normalizes Windows backslashes (`\\`) to forward slashes (`/`).
    - Translates `E:/Corpus/...` or `/Corpus/...` to the active corpus root (e.g., `/data/Corpus/...`).
    - Preserves existing valid paths untouched if no remapping is needed.
    """
    if path is None:
        return None

    p_str = str(path).replace("\\", "/")

    active_root = get_active_corpus_root()
    if not active_root:
        return p_str

    active_root_norm = active_root.replace("\\", "/").rstrip("/")

    # Detect Windows drive letter prefix (e.g., E:/Corpus/...)
    p_lower = p_str.lower()
    if p_lower.startswith("e:/corpus"):
        rel = p_str[len("e:/corpus"):].lstrip("/")
        return f"{active_root_norm}/{rel}" if rel else active_root_norm
    elif p_lower.startswith("/corpus"):
        rel = p_str[len("/corpus"):].lstrip("/")
        return f"{active_root_norm}/{rel}" if rel else active_root_norm

    return p_str
