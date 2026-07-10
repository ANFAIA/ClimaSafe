"""
gstack — Git Stack: flujos de trabajo autónomos para el sistema de agentes.

Permite encadenar operaciones de múltiples agentes en una "stack" (pila) que
se ejecuta secuencialmente, con gestión automática de commits entre cada paso.

Características
---------------
- Result passing: referencias ``prev(key).data`` entre pasos
- Branching condicional: ``run_if=lambda results: ...``
- Event logging: cada acción se registra en ``workspace/gstack/events.jsonl``

Uso básico
----------
    from agents.gstack import GStack

    stack = GStack()
    stack.push("git", "analyze_diff")
    stack.push("review", "review_package", path=".")
    stack.push("git", "commit_with_changelog", message="feat: mejora tras revisión")
    result = stack.run()

Uso con auto-commit (cada paso genera un commit automático):
----------------------------------------------------------------
    from agents.gstack import auto_commit

    stack = GStack(auto_commit=True)
    stack.push("data", "eda_report", filename="dataset.csv")
    stack.push("ml", "check_overfitting")
    result = stack.run()

Result passing entre pasos
--------------------------
    stack.push("review", "review_package", path=".", result_key="review")
    stack.push("data", "eda_report", filename=prev("review").data["file"])
    #                                      ~~~~
    # Referencia al campo .data del resultado del paso con result_key="review"

Branching condicional
---------------------
    stack.push("test", "run_tests", run_if=lambda r: r[-1].success if r else True)
    # El paso solo se ejecuta si el anterior fue exitoso

Pipelines predefinidos
-----------------------
    from agents.gstack import auto_develop, auto_release, auto_fix
    from agents.gstack import auto_commit_cycle, auto_analyze, auto_data_pipeline

    result = auto_develop()            # review → test → commit
    result = auto_release("1.2.0")     # bump → changelog → commit → tag
    result = auto_fix()                # test → review → fix → commit
    result = auto_analyze()            # diagnóstico sin modificar
    result = auto_data_pipeline(filename="dataset.csv")
"""

from __future__ import annotations

from agents.gstack.stack import GStack
from agents.gstack.pipelines import (
    auto_develop,
    auto_release,
    auto_fix,
    auto_commit_cycle,
    auto_analyze,
    auto_data_pipeline,
    run_pipeline,
)

__all__ = [
    "GStack",
    "auto_develop",
    "auto_release",
    "auto_fix",
    "auto_commit_cycle",
    "auto_analyze",
    "auto_data_pipeline",
    "run_pipeline",
]
