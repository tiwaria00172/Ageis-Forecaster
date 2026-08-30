# =============================================================================
# Helpers — Utility functions used across the pipeline
# =============================================================================
"""
Common helper functions: config loading, path resolution, seed setting,
device selection, and data I/O utilities.
"""

import os
import random
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(*parts: str) -> Path:
    """Resolve a path relative to the project root."""
    return get_project_root().joinpath(*parts)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the YAML configuration file.

    Args:
        config_path: Explicit path to config.yaml.
                     Defaults to <project_root>/config/config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    if config_path is None:
        config_path = str(resolve_path("config", "config.yaml"))

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> "torch.device":
    """Return the best available torch device (CUDA > CPU)."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_pickle(obj: Any, path: str) -> None:
    """Save an object to a pickle file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    """Load an object from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str) -> None:
    """Save a JSON-serialisable object to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str) -> Any:
    """Load a JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
