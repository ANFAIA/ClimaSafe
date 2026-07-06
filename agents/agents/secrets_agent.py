"""
agents.agents.secrets_agent — Escanea el proyecto en busca de secretos hardcodeados.

Aviso honesto, no lo suavizo: sin `detect-secrets` instalado, la cobertura
de este agente es una fracción pequeña de lo que hacen herramientas reales
del ecosistema (detect-secrets, gitleaks, TruffleHog — ver
`agents/tools/secrets_tool.py` para el detalle exacto verificado). Este
agente no sustituye ninguna de esas herramientas, es una primera capa.
"""

from __future__ import annotations

from agents.core.base_agent import AgentResult, BaseAgent
from agents.core.registry import register_agent
from agents.tools.secrets_tool import SecretsTool


@register_agent
class SecretsAgent(BaseAgent):
    name = "secrets"
    description = (
        "Escanea el proyecto en busca de secretos hardcodeados (claves, tokens, contraseñas). "
        "Usa detect-secrets si está instalado; si no, un heurístico propio mucho más limitado."
    )
    capabilities = ["secretos", "secrets", "credenciales", "contraseña hardcodeada", "detect-secrets"]

    def action_aliases(self) -> dict:
        return {"scan": ["escanea", "escanear", "busca secretos", "detecta secretos"]}

    def actions(self) -> dict:
        return {"scan": self.scan}

    def scan(self) -> AgentResult:
        findings, detector = SecretsTool.scan(self.ctx.root)

        warnings = []
        if detector == "heuristico-propio":
            warnings.append(
                "detect-secrets no está instalado — se usó el heurístico propio, mucho más limitado "
                "(no detecta tokens de Slack/Stripe/GitHub/JWT/etc., solo claves AWS, cabeceras PEM y "
                "asignaciones de alta entropía). Instala 'detect-secrets' (pip install detect-secrets) "
                "para una cobertura real."
            )

        if findings:
            warnings.append(
                "Si alguno de estos es un secreto real: NO te limites a borrarlo del código. "
                "Rota/revoca la credencial primero (ya se consideraría comprometida al haber estado "
                "en el repo), y ten en cuenta que borrar la línea no la quita del historial de git."
            )

        data = [f.__dict__ for f in findings]
        return AgentResult(
            len(findings) == 0, self.name, "scan",
            f"{len(findings)} posible(s) secreto(s) encontrado(s) (detector: {detector}).",
            data=data, warnings=warnings,
        )
