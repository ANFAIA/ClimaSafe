"""Tests de los contratos de rol: todo agente tiene contrato y nadie pisa a nadie."""

from __future__ import annotations

from agents.contracts import CONTRACTS, validate_contracts
from agents.core.registry import agent_registry


def test_team_contracts_are_coherent():
    """
    La regla nº1 del equipo ("nadie se pisa") y la nº2 ("todo agente tiene un
    rol definido") no son prosa de README: se validan aquí. Si añades un
    agente sin contrato, o dos contratos declaran el mismo recurso en `owns`,
    este test te lo dice antes de que lo descubra una orden de trabajo real.
    """
    agent_registry.discover()
    problems = validate_contracts(set(agent_registry.all()))
    assert not problems, "Contratos incoherentes:\n" + "\n".join(problems)


def test_every_contract_has_role_and_limits():
    """Un contrato sin límites (`cannot`) no delimita nada — es el origen de los solapes."""
    missing_limits = [
        name for name, contract in CONTRACTS.items()
        if not contract.role.strip() or not contract.cannot
    ]
    assert not missing_limits, (
        f"Contratos sin rol o sin límites (cannot): {missing_limits}. "
        f"Todo agente debe declarar qué NO hace y a quién derivarlo."
    )
