"""Pruebas offline de la integración del flujo multiagente en ``DebugOrchestrator``.

Verifica que ``DebugOrchestrator.run_multiagent_analysis`` construye el grafo
LangGraph (explorador → orquestador → trabajadores) y ejecuta los agentes en
orden topológico respetando la profundidad, **sin** LLM real ni red.

Se usa ``DebugOrchestrator.__new__`` para saltarse el ``__init__`` pesado
(carga ``workflow_data.json`` y calcula el resumen gramatical) y se inyectan
stubs para ``load_workflow_specs``, ``TogetherChatModel`` y los métodos de
instancia.
"""

from __future__ import annotations

import pytest

from reii.ia_discursiva import DebugOrchestrator
from reii.pipelines.dependency_graph import build_workflow_spec

# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


class FakeLLM:
    """LLM falso: scout devuelve recursos vacíos; orquestador devuelve profundidad fija."""

    def __init__(self, depth: int):
        self.depth = depth

    def invoke_structured(self, messages, schema):
        props = schema.get("properties", {})
        if "resources" in props:
            return {"summary": "s", "resources": []}
        return {"depth": self.depth, "reasoning": "r", "skipped_agents": []}


class FakeAgent:
    """BaseAgent falso con ``role`` (lo único que usa ``run_multiagent_analysis``)."""

    def __init__(self, role: str):
        self.role = role


def make_spec(name: str, n_layers: int = 2):
    agents = []
    for i in range(1, n_layers + 1):
        deps = [f"{name}_a{i - 1}"] if i > 1 else []
        agents.append(
            {
                "name": f"{name}_a{i}",
                "role": f"rol {i}",
                "instructions": "i",
                "gram_cats": [],
                "disc_cats": deps,
            }
        )
    return build_workflow_spec(name, agents)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_orchestrator(monkeypatch, specs, depth: int = 2):
    """Construye un ``DebugOrchestrator`` ligero con stubs inyectados."""
    mon = DebugOrchestrator.__new__(DebugOrchestrator)
    mon.discourse_agents = {
        f"{name}_a{i}": FakeAgent(f"rol {i}")
        for name, spec in specs.items()
        for i in range(1, spec.n_layers + 1)
    }
    mon.global_grammar_summary = "resumen gramatical"
    mon.MAX_TAGGED_CHARS = 10_000
    mon._build_tagged_text_and_map = lambda: ("texto etiquetado", [])
    mon.run_discourse_agent = lambda name, **kw: [{"agent": name}]

    # Stubs de las dependencias importadas dentro del método.
    monkeypatch.setattr(
        "reii.pipelines.dependency_graph.load_workflow_specs",
        lambda config_dir: specs,
    )
    monkeypatch.setattr(
        "reii.pipelines.langchain_llm.TogetherChatModel",
        lambda: FakeLLM(depth),
    )
    return mon


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_run_multiagent_analysis_returns_graph_state(monkeypatch):
    specs = {"cohesion": make_spec("cohesion", n_layers=2)}
    mon = _make_orchestrator(monkeypatch, specs, depth=2)

    result = mon.run_multiagent_analysis(store=False)

    assert set(result["scout_reports"].keys()) == {"cohesion"}
    assert set(result["depth_plans"].keys()) == {"cohesion"}
    assert result["depth_plans"]["cohesion"].depth == 2
    # Orden topológico: capa 1 antes que capa 2.
    assert result["annotations"] == [
        {"agent": "cohesion_a1"},
        {"agent": "cohesion_a2"},
    ]


def test_run_multiagent_analysis_respects_depth(monkeypatch):
    specs = {"cohesion": make_spec("cohesion", n_layers=3)}
    mon = _make_orchestrator(monkeypatch, specs, depth=1)

    result = mon.run_multiagent_analysis(store=False)

    # Solo capa 1 se ejecuta (ahorro de agentes).
    assert result["annotations"] == [{"agent": "cohesion_a1"}]


def test_run_multiagent_analysis_multiple_workflows(monkeypatch):
    specs = {
        "cohesion": make_spec("cohesion", n_layers=2),
        "critico": make_spec("critico", n_layers=2),
    }
    mon = _make_orchestrator(monkeypatch, specs, depth=2)

    result = mon.run_multiagent_analysis(store=False)

    assert set(result["scout_reports"].keys()) == {"cohesion", "critico"}
    assert set(result["depth_plans"].keys()) == {"cohesion", "critico"}
    assert result["annotations"] == [
        {"agent": "cohesion_a1"},
        {"agent": "cohesion_a2"},
        {"agent": "critico_a1"},
        {"agent": "critico_a2"},
    ]


def test_run_multiagent_analysis_no_specs_returns_empty(monkeypatch):
    mon = _make_orchestrator(monkeypatch, {}, depth=2)

    result = mon.run_multiagent_analysis(store=False)

    assert result == {"scout_reports": {}, "depth_plans": {}, "annotations": []}
