from __future__ import annotations

from agents.core.registry import agent_registry
from agents.orchestrator import Orchestrator


def test_every_capability_keyword_routes_to_its_own_agent(context):
    """
    Para cada palabra clave declarada por cada agente, usarla SOLA como
    consulta debe rutear a ese mismo agente. Si no es así, hay una colisión
    real entre agentes (exactamente el tipo de bug que encontré a mano con
    "ci" dentro de "dependencias" o "smoke" compartido entre test/api) — este
    test la encuentra sistemáticamente en vez de depender de qué frases se
    me ocurra probar.
    """
    orchestrator = Orchestrator(context=context)
    agent_registry.discover()

    problems = []
    for owner_name in agent_registry.all():
        instance = orchestrator._get_instance(owner_name)  # noqa: SLF001 — acceso intencional para el test
        for keyword in instance.capabilities:
            decision = orchestrator.select_agent(keyword)
            if decision.agent_name != owner_name:
                problems.append(
                    f"'{keyword}' (de '{owner_name}') enrutó a '{decision.agent_name}' "
                    f"en vez de a su propio agente. Candidatos: {decision.candidates}"
                )

    assert not problems, "Colisiones de capabilities encontradas:\n" + "\n".join(problems)
