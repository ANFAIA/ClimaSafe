"""
agents.agents.notebook_agent — Extrae salidas de notebooks e inserta
interpretaciones como celdas markdown.

Este agente es distinto a los otros seis en un punto importante: NO
interpreta nada por sí mismo — eso requeriría comprensión visual/semántica
real, que este sistema determinista no tiene (ver Filosofía en
`agents/README.md`: "no es un chatbot"). Lo que hace es la parte mecánica
(extraer imágenes/texto, insertar celdas), y deja el juicio ("esta curva
muestra overfitting a partir de la época 20") a quien lo orquesta — una
persona, o un LLM operando sobre este sistema de agentes.

Usa `agents/workspace/notebook/` para las imágenes extraídas y el manifest —
nunca escribe en la raíz del proyecto.
"""

from __future__ import annotations

import binascii
import json

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.notebook_tool import NotebookTool


@register_agent
class NotebookAgent(BaseAgent):
    name = "notebook"
    description = (
        "Extrae salidas (imágenes, texto, métricas) de un notebook ya ejecutado e inserta "
        "interpretaciones como celdas markdown. No interpreta nada él mismo — ver docstring del módulo."
    )
    capabilities = [
        "notebook", "jupyter", "ipynb", "celda", "interpretar resultados",
        "comentar notebook", "outputs del notebook",
    ]

    def actions(self) -> dict:
        return {
            "extract_outputs": self.extract_outputs,
            "insert_comments": self.insert_comments,
        }

    def _resolve_notebook(self, notebook_path: str):
        path = self.ctx.root / notebook_path
        if not path.exists():
            path = self.ctx.notebooks_dir / notebook_path
        return path if path.exists() else None

    def extract_outputs(self, *, notebook_path: str) -> AgentResult:
        """
        Extrae imágenes y texto de las salidas ya ejecutadas de `notebook_path`
        a `agents/workspace/notebook/`. Devuelve el manifest completo en
        `data["manifest"]` para que quien orqueste no tenga que releer el
        archivo — y las rutas a las imágenes para que las mire de verdad
        (este agente no puede "ver" nada, solo decodificarlo a disco).
        """
        path = self._resolve_notebook(notebook_path)
        if path is None:
            return AgentResult(False, self.name, "extract_outputs", f"No se encontró el notebook '{notebook_path}'.")
        if path.suffix != ".ipynb":
            return AgentResult(False, self.name, "extract_outputs", f"'{notebook_path}' no es un .ipynb.")

        workdir = self.ctx.agent_workspace("notebook")
        try:
            result = NotebookTool.extract(path, workdir)
        except (json.JSONDecodeError, binascii.Error) as exc:
            return AgentResult(False, self.name, "extract_outputs", f"No se pudo procesar el notebook: {exc}")

        warnings = []
        if result["n_cells_with_output"] == 0:
            warnings.append(
                "El notebook no tiene celdas con salida — o no se ejecutó, o no hay nada que interpretar."
            )
        n_errored = sum(1 for entry in result["manifest"] if entry["errored"])
        if n_errored:
            warnings.append(f"{n_errored} celda(s) terminaron en error — no interpretes un resultado que nunca se generó.")

        return AgentResult(
            True, self.name, "extract_outputs",
            f"{result['n_cells_with_output']} celda(s) con salida, {result['n_images']} imagen(es) extraída(s) a {workdir}.",
            data=result, warnings=warnings,
        )

    def insert_comments(self, *, notebook_path: str, insertions: list[dict], in_place: bool = False) -> AgentResult:
        """
        Inserta celdas markdown con las interpretaciones ya escritas (por
        quien orqueste este agente, no por él). `insertions` es una lista de
        {"cell_index": int, "markdown": str}.

        `in_place=False` por defecto (mismo patrón de seguridad que
        `DocumentationAgent.update_changelog` con `dry_run`): escribe una
        copia en `agents/workspace/notebook/<nombre>_comentado.ipynb` en vez
        de sobreescribir el original. Pasa `in_place=True` explícitamente
        para modificar el notebook original — hazlo solo si el usuario lo
        pidió así, no por defecto.
        """
        path = self._resolve_notebook(notebook_path)
        if path is None:
            return AgentResult(False, self.name, "insert_comments", f"No se encontró el notebook '{notebook_path}'.")

        if not insertions:
            return AgentResult(False, self.name, "insert_comments", "No se pasaron interpretaciones que insertar.")

        output_path = None
        if not in_place:
            workdir = self.ctx.agent_workspace("notebook")
            output_path = workdir / f"{path.stem}_comentado.ipynb"

        try:
            result = NotebookTool.insert_comments(path, insertions, output_path=output_path)
        except (json.JSONDecodeError, KeyError) as exc:
            return AgentResult(False, self.name, "insert_comments", f"No se pudo insertar: {exc}")

        warnings = []
        if result["skipped_indices"]:
            warnings.append(f"cell_index fuera de rango, omitido(s): {result['skipped_indices']}")
        if in_place:
            warnings.append(f"Se sobreescribió '{path}' — sin backup automático.")

        return AgentResult(
            True, self.name, "insert_comments",
            f"{result['n_inserted']} comentario(s) insertado(s) en {result['output_path']}.",
            data=result, warnings=warnings,
        )
