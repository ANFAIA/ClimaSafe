"""
climasafeai.models.registry — Plug-and-play model registry.

Scans a directory for ``*.manifest.json`` files, validates them against a
minimal schema, and returns a list of model descriptors.  The ensemble uses
this list instead of hardcoding model paths.

Manifest spec (see ``MANIFEST_SCHEMA`` for the authoritative version):

.. code-block:: json

    {
      "name": "XGBoost_calor",
      "type": "tabular",
      "class": "calor",
      "file": "XGBoost_calor.joblib",
      "description": "XGBoost for heat risk prediction",
      "enabled": true
    }

Fields
------
- **name** (str, required): unique identifier inside the ensemble.
- **type** (str, required): ``"tabular"`` | ``"lstm"`` | ``"formula"``.
  Determines which prediction backend is used.
- **class** (str, required): ``"calor"`` | ``"frio"`` | ``"both"``.
  Which risk class(es) the model covers.
- **file** (str, optional for ``lstm`` / ``"formula"``): path to the
  serialized model artifact, relative to ``MODELS_DIR``.
- **description** (str, optional): human-readable purpose.
- **enabled** (bool, default ``true``): set to ``false`` to exclude without
  deleting the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from climasafeai.utils import paths as _paths

MANIFEST_SCHEMA: dict[str, Any] = {
    "required": ["name", "type", "class"],
    "optional": {"file": "", "description": "", "enabled": True},
    "valid_types": {"tabular", "lstm", "formula"},
    "valid_classes": {"calor", "frio", "both"},
}


class ManifestError(Exception):
    """Raised when a manifest file is invalid."""


def _validate_manifest(data: dict, path: Path) -> dict:
    """Validate a single manifest dict and return the cleaned version."""
    for field in MANIFEST_SCHEMA["required"]:
        if field not in data:
            raise ManifestError(f"Missing required field '{field}' in {path}")

    if data["type"] not in MANIFEST_SCHEMA["valid_types"]:
        raise ManifestError(
            f"Invalid type '{data['type']}' in {path} — "
            f"must be one of {MANIFEST_SCHEMA['valid_types']}"
        )

    if data["class"] not in MANIFEST_SCHEMA["valid_classes"]:
        raise ManifestError(
            f"Invalid class '{data['class']}' in {path} — "
            f"must be one of {MANIFEST_SCHEMA['valid_classes']}"
        )

    # Apply defaults for optional fields
    for key, default in MANIFEST_SCHEMA["optional"].items():
        data.setdefault(key, default)

    # LSTM and formula don't need a file
    if data["type"] in ("tabular",) and not data.get("file"):
        raise ManifestError(
            f"Tabular model '{data['name']}' in {path} must specify a 'file'"
        )

    return data


def discover_models(directory: Path | None = None) -> list[dict]:
    """Scan *directory* for ``*.manifest.json`` and return validated models.

    Parameters
    ----------
    directory : Path, optional
        Folder to scan.  Defaults to ``MODELS_DIR``.

    Returns
    -------
    list[dict]
        One dict per enabled manifest, with the raw JSON fields plus the
        resolved absolute ``path`` to the model artifact (when applicable).
    """
    if directory is None:
        # Lookup at CALL time, not import time: tests (conftest patch_paths)
        # redirect paths.MODELS_DIR per test; a binding captured at import
        # would ignore the patch depending on import order.
        directory = _paths.MODELS_DIR

    models: list[dict] = []
    if not directory.is_dir():
        return models

    for manifest_path in sorted(directory.glob("*.manifest.json")):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Invalid JSON in {manifest_path}: {exc}") from exc

        data = _validate_manifest(raw, manifest_path)

        if not data.get("enabled", True):
            continue

        # Resolve artifact path relative to MODELS_DIR
        if data.get("file"):
            artifact = _paths.MODELS_DIR / data["file"]
            data["path"] = str(artifact)
        else:
            data["path"] = None

        models.append(data)

    return models


def get_model_names(directory: Path | None = None) -> list[str]:
    """Return just the names of discovered (enabled) models."""
    return [m["name"] for m in discover_models(directory)]
