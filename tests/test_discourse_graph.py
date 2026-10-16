"""Pruebas offline del grafo de discurso (sin LLM ni red).

Cubre ``DiscourseGraph`` y ``build_discourse_graph`` con un LLM falso y un
``executor`` falso que registra el orden de llamadas.
"""

import pytest

from reii.pipelines.dependency_graph import build_workflow_spec
from reii.pipelines.discourse_graph import DiscourseGraph, build_discourse_graph

# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


class FakeLLM:
    """LLM falso: devuelve un reporte de scout y una profundidad fija."""

    def __init__(self, depth):
        self.depth = depth

    def invoke_structured(self, messages, schema):
        # Distingue scout (SCOUT_SCHEMA) de orquestador (ORCHESTRATOR_SCHEMA)
        # por la presencia de la clave "resources" en las properties.
        props = schema.get("properties", {})
        if "resources" in props:
            return {"summary": "s", "resources": []}
        return {"depth": self.depth, "reasoning": "r", "skipped_agents": []}


class RecordingExecutor:
    """Executor falso que registra el orden de llamadas."""

    def __init__(self):
        self.calls = []

    def __call__(self, agent_name):
        self.calls.append(agent_name)
        return [{"agent": agent_name}]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_workflow(name="cohesion", n_layers=2):
    agents = []
    for i in range(1, n_layers + 1):
        deps = [f"{name}_a{i - 1}"] if i > 1 else []
        agents.append(
            {
                "name": f"{name}_a{i}",
                "role": "r",
                "instructions": "i",
                "gram_cats": [],
                "disc_cats": deps,
            }
        )
    return build_workflow_spec(name, agents)


def make_graph(depth, workflows=None):
    if workflows is None:
        workflows = {"cohesion": make_workflow("cohesion", n_layers=3)}
    executor = RecordingExecutor()
    graph = DiscourseGraph(FakeLLM(depth), workflows, {}, executor)
    return graph, executor


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_run_respects_depth_one():
    graph, executor = make_graph(depth=1)
    result = graph.run("grammar", "tagged")

    # Solo agentes de capa 1 (cohesion_a1) se ejecutan.
    assert executor.calls == ["cohesion_a1"]
    assert result["annotations"] == [{"agent": "cohesion_a1"}]


def test_run_full_depth_executes_topological_order():
    graph, executor = make_graph(depth=3)
    result = graph.run("grammar", "tagged")

    # Capa 1 antes que capa 2 antes que capa 3.
    assert executor.calls == ["cohesion_a1", "cohesion_a2", "cohesion_a3"]
    assert result["annotations"] == [
        {"agent": "cohesion_a1"},
        {"agent": "cohesion_a2"},
        {"agent": "cohesion_a3"},
    ]


def test_run_populates_reports_plans_and_annotations():
    graph, executor = make_graph(depth=2)
    result = graph.run("grammar", "tagged")

    assert set(result["scout_reports"].keys()) == {"cohesion"}
    assert set(result["depth_plans"].keys()) == {"cohesion"}
    assert result["depth_plans"]["cohesion"].depth == 2
    # depth 2 → capas 1 y 2, no la 3.
    assert executor.calls == ["cohesion_a1", "cohesion_a2"]


def test_run_multiple_workflows_each_has_entry():
    workflows = {
        "cohesion": make_workflow("cohesion", n_layers=2),
        "critico": make_workflow("critico", n_layers=2),
    }
    graph, executor = make_graph(depth=2, workflows=workflows)
    result = graph.run("grammar", "tagged")

    assert set(result["scout_reports"].keys()) == {"cohesion", "critico"}
    assert set(result["depth_plans"].keys()) == {"cohesion", "critico"}
    # Cada workflow ejecuta sus dos agentes en orden topológico.
    assert executor.calls == [
        "cohesion_a1",
        "cohesion_a2",
        "critico_a1",
        "critico_a2",
    ]


def test_layer_two_never_runs_before_its_dependency():
    graph, executor = make_graph(depth=3)
    graph.run("grammar", "tagged")

    # a2 depende de a1: a1 debe aparecer antes que a2.
    assert executor.calls.index("cohesion_a1") < executor.calls.index("cohesion_a2")
    assert executor.calls.index("cohesion_a2") < executor.calls.index("cohesion_a3")


def test_build_discourse_graph_factory():
    executor = RecordingExecutor()
    graph = build_discourse_graph(
        FakeLLM(1), {"cohesion": make_workflow("cohesion", n_layers=2)}, {}, executor
    )
    assert isinstance(graph, DiscourseGraph)
    result = graph.run("g", "t")
    assert executor.calls == ["cohesion_a1"]
