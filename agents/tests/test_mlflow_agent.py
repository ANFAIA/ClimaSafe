from __future__ import annotations

import pytest

from agents.agents.mlflow_agent import MLflowAgent
from agents.config import ProjectConfig
from agents.context import SharedContext

mlflow = pytest.importorskip("mlflow", reason="mlflow es un extra opcional (use_mlflow) del template, no una dependencia de agents/")


@pytest.fixture
def mlflow_context(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    return SharedContext(root=tmp_path, config=ProjectConfig(project_slug="mi_paquete"))


def _log_run(run_name: str, metric_value: float):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", "RandomForest")
        mlflow.log_metric("cv_score", metric_value)


def test_list_runs_no_experiment_yet(mlflow_context):
    agent = MLflowAgent(context=mlflow_context)
    result = agent.list_runs()
    assert result.success
    assert result.data == []


def test_list_runs_and_best_run(mlflow_context):
    mlflow.set_experiment("mi_paquete")
    _log_run("a", 0.80)
    _log_run("b", 0.90)
    _log_run("c", 0.75)

    agent = MLflowAgent(context=mlflow_context)
    result = agent.list_runs()
    assert result.success
    assert len(result.data) == 3

    best = agent.best_run(metric="cv_score")
    assert best.success
    assert best.data["metrics"]["cv_score"] == 0.90


def test_compare_latest_detects_regression(mlflow_context):
    mlflow.set_experiment("mi_paquete")
    _log_run("a", 0.90)
    _log_run("b", 0.70)  # el más reciente, peor que el anterior

    agent = MLflowAgent(context=mlflow_context)
    result = agent.compare_latest(metric="cv_score")
    assert result.success
    assert result.data["regressed"] is True
    assert result.warnings


def test_compare_latest_no_regression_has_no_warning(mlflow_context):
    mlflow.set_experiment("mi_paquete")
    _log_run("a", 0.70)
    _log_run("b", 0.90)  # el más reciente, mejor

    agent = MLflowAgent(context=mlflow_context)
    result = agent.compare_latest(metric="cv_score")
    assert result.success
    assert result.data["regressed"] is False
    assert result.warnings == []


def test_missing_project_slug_fails_without_mlflow_call(tmp_path):
    ctx = SharedContext(root=tmp_path, config=ProjectConfig(project_slug=""))
    agent = MLflowAgent(context=ctx)
    result = agent.list_runs()
    assert not result.success
