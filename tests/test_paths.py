"""
test_paths.py — Tests para climasafeai/utils/paths.py
Común a todos los ml_type.
"""
from pathlib import Path
from climasafeai.utils import paths


def test_all_path_constants_are_path_objects():
    """Todas las constantes de ruta deben ser instancias de Path."""
    expected = [
        "PROJECT_DIR",
        "DATA_DIR",
        "RAW_DATA_DIR",
        "INTERIM_DATA_DIR",
        "PROCESSED_DATA_DIR",
        "MODELS_DIR",
        "ARTIFACTS_DIR",
        "REPORTS_DIR",
        "FIGURES_DIR",
    ]
    for name in expected:
        assert hasattr(paths, name), f"Falta la constante: {name}"
        assert isinstance(getattr(paths, name), Path), (
            f"{name} debe ser Path, no {type(getattr(paths, name))}"
        )





def test_project_dir_contains_pyproject(tmp_path, monkeypatch):
    """
    make_dirs() debe crear todos los subdirectorios necesarios.
    Se prueba con un PROJECT_DIR temporal para no contaminar el proyecto real.
    """
    fake_root = tmp_path / "fake_project"
    fake_root.mkdir()

    # Reemplazar PROJECT_DIR en el módulo para este test
    monkeypatch.setattr(paths, "PROJECT_DIR", fake_root)
    new_data     = fake_root / "data"
    new_raw      = new_data  / "raw"
    new_interim  = new_data  / "interim"
    new_proc     = new_data  / "processed"
    new_models   = fake_root / "models"
    new_artifacts = new_models / "artifacts"
    new_figures  = fake_root / "reports" / "figures"

    monkeypatch.setattr(paths, "DATA_DIR",            new_data)
    monkeypatch.setattr(paths, "RAW_DATA_DIR",        new_raw)
    monkeypatch.setattr(paths, "INTERIM_DATA_DIR",    new_interim)
    monkeypatch.setattr(paths, "PROCESSED_DATA_DIR",  new_proc)
    monkeypatch.setattr(paths, "MODELS_DIR",          new_models)
    monkeypatch.setattr(paths, "ARTIFACTS_DIR",       new_artifacts)
    monkeypatch.setattr(paths, "FIGURES_DIR",         new_figures)


    paths.make_dirs()

    assert new_raw.exists()
    assert new_interim.exists()
    assert new_proc.exists()
    assert new_models.exists()
    assert new_artifacts.exists()
    assert new_figures.exists()
