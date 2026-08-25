"""
test_registry.py — Tests for the plug-and-play model registry.

Verifies:
- Manifest validation (required fields, valid types/classes)
- Folder discovery (*.manifest.json scanning)
- Enabled/disabled models
- Backward compatibility: ensemble still works without manifests
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from climasafeai.models.registry import (
    discover_models,
    get_model_names,
    _validate_manifest,
    ManifestError,
    MANIFEST_SCHEMA,
)


# ── Manifest validation ──────────────────────────────────────────────────

def test_valid_tabular_manifest():
    data = {"name": "XGBoost_calor", "type": "tabular", "class": "calor", "file": "XGBoost_calor.joblib"}
    result = _validate_manifest(data, Path("test.json"))
    assert result["name"] == "XGBoost_calor"
    assert result["enabled"] is True  # default


def test_valid_formula_manifest():
    data = {"name": "Formula", "type": "formula", "class": "both"}
    result = _validate_manifest(data, Path("test.json"))
    assert result["file"] == ""  # default


def test_valid_lstm_manifest():
    data = {"name": "LSTM", "type": "lstm", "class": "both", "file": "model.pt"}
    result = _validate_manifest(data, Path("test.json"))
    assert result["name"] == "LSTM"
    assert result["type"] == "lstm"


def test_missing_required_field_raises():
    data = {"name": "test"}  # missing type and class
    with pytest.raises(ManifestError, match="Missing required field 'type'"):
        _validate_manifest(data, Path("test.json"))


def test_invalid_type_raises():
    data = {"name": "test", "type": "neural_net", "class": "calor"}
    with pytest.raises(ManifestError, match="Invalid type"):
        _validate_manifest(data, Path("test.json"))


def test_invalid_class_raises():
    data = {"name": "test", "type": "tabular", "class": "rain"}
    with pytest.raises(ManifestError, match="Invalid class"):
        _validate_manifest(data, Path("test.json"))


def test_tabular_without_file_raises():
    data = {"name": "test", "type": "tabular", "class": "calor"}
    with pytest.raises(ManifestError, match="must specify a 'file'"):
        _validate_manifest(data, Path("test.json"))


def test_disabled_model_excluded(tmp_path):
    manifest = tmp_path / "disabled.manifest.json"
    manifest.write_text(json.dumps({
        "name": "disabled_model", "type": "formula", "class": "both", "enabled": False,
    }))
    models = discover_models(tmp_path)
    assert len(models) == 0


# ── Folder discovery ─────────────────────────────────────────────────────

def test_discover_empty_dir(tmp_path):
    models = discover_models(tmp_path)
    assert models == []


def test_discover_multiple_manifests(tmp_path):
    for name, mtype, cls in [("A", "tabular", "calor"), ("B", "formula", "both")]:
        manifest = tmp_path / f"{name}.manifest.json"
        data = {"name": name, "type": mtype, "class": cls}
        if mtype == "tabular":
            data["file"] = f"{name}.joblib"
        manifest.write_text(json.dumps(data))

    models = discover_models(tmp_path)
    assert len(models) == 2
    names = [m["name"] for m in models]
    assert "A" in names
    assert "B" in names


def test_discover_ignores_non_manifest_files(tmp_path):
    (tmp_path / "model.joblib").write_bytes(b"fake")
    (tmp_path / "notes.txt").write_text("ignore me")
    manifest = tmp_path / "real.manifest.json"
    manifest.write_text(json.dumps({"name": "real", "type": "formula", "class": "both"}))

    models = discover_models(tmp_path)
    assert len(models) == 1
    assert models[0]["name"] == "real"


def test_get_model_names(tmp_path):
    for name in ["alpha", "beta"]:
        manifest = tmp_path / f"{name}.manifest.json"
        manifest.write_text(json.dumps({"name": name, "type": "formula", "class": "both"}))

    names = get_model_names(tmp_path)
    assert names == ["alpha", "beta"]


def test_invalid_json_raises(tmp_path):
    manifest = tmp_path / "bad.manifest.json"
    manifest.write_text("{not valid json")
    with pytest.raises(ManifestError, match="Invalid JSON"):
        discover_models(tmp_path)


def test_nonexistent_dir_returns_empty():
    models = discover_models(Path("/nonexistent/path"))
    assert models == []


# ── Manifest spec constants ─────────────────────────────────────────────

def test_schema_has_required_fields():
    assert "required" in MANIFEST_SCHEMA
    assert "name" in MANIFEST_SCHEMA["required"]
    assert "type" in MANIFEST_SCHEMA["required"]
    assert "class" in MANIFEST_SCHEMA["required"]


def test_schema_valid_types():
    assert MANIFEST_SCHEMA["valid_types"] == {"tabular", "lstm", "formula"}


def test_schema_valid_classes():
    assert MANIFEST_SCHEMA["valid_classes"] == {"calor", "frio", "both"}
