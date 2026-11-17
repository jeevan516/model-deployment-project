"""Save and load the trained model with skops (no pickle)."""

from __future__ import annotations

import json
from pathlib import Path

import skops.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.skops"
# Random forests store their nodes in this type; we trust it because we train the model ourselves.
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def save(model, path: str | Path = DEFAULT_MODEL_PATH, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return path


def load(path: str | Path = DEFAULT_MODEL_PATH):
    path = Path(path)
    untrusted = [t for t in sio.get_untrusted_types(file=path) if t not in TRUSTED_TYPES]
    if untrusted:
        raise ValueError(f"Refusing to load {path}: untrusted types {untrusted}")
    return sio.load(path, trusted=TRUSTED_TYPES)


def load_metadata(path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    meta = Path(path).with_suffix(".json")
    return json.loads(meta.read_text()) if meta.exists() else {}
