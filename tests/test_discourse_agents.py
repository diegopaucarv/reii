"""Pruebas offline de los agentes del discurso (sin LLM ni red).

Cubre ``ScoutAgent``, ``OrchestratorAgent`` y ``WorkerAgent`` usando stubs
con ``invoke_structured`` y ``analyze_corpus``.
"""

import pytest

from reii.pipelines.dependency_graph import build_workflow_spec
from reii.pipelines.discourse_agents import (
    DepthPlan,
    OrchestratorAgent,
    ScoutAgent,
    ScoutReport,
    WorkerAgent,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


class FakeLLM:
    """LLM falso que devuelve un dict fijo desde ``invoke_structured``."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke_structured(self, messages, schema):
        self.calls.append((messages, schema))
        return self.result


class RaisingLLM:
    """LLM falso que lanza una excepción en ``invoke_structured``."""

    def invoke_structured(self, messages, schema):
        raise RuntimeError("boom")


class FakeBaseAgent:
    """BaseAgent falso que registra las llamadas a ``analyze_corpus``."""

    def __init__(self, result=None):
        self.result = result if result is not None else []
        self.calls = []

    def analyze_corpus(self, tagged_text, grammar_summary, previous):
        self.calls.append((tagged_text, grammar_summary, previous))
        return self.result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_workflow(n_layers=2):
    """Workflow de ejemplo con ``n_layers`` capas encadenadas."""
    agents = []
    for i in range(1, n_layers + 1):
        deps = [f"a{i - 1}"] if i > 1 else []
        agents.append(
            {
                "name": f"a{i}",
                "role": "r",
                "instructions": "i",
                "gram_cats": [],
                "disc_cats": deps,
            }
        )
    return build_workflow_spec("cohesion", agents)


# ─────────────────────────────────────────────────────────────────────────────
# ScoutAgent
# ─────────────────────────────────────────────────────────────────────────────


def test_scout_analyze_returns_report_with_resources_and_summary():
    llm = FakeLLM(
        {
            "summary": "hay nominalizaciones",
            "resources": [
                {"name": "nominalizacion", "confidence": "alta", "evidence": "x"},
                {"name": "metafora", "confidence": "media", "evidence": "y"},
                {"name": "figura", "confidence": "baja", "evidence": "z"},
            ],
        }
    )
    workflow = make_workflow()
    agent = ScoutAgent(llm)

    report = agent.analyze(workflow, {}, "texto", "resumen")

    assert isinstance(report, ScoutReport)
    assert report.workflow == "cohesion"
    assert report.summary == "hay nominalizaciones"
    assert report.resource_names() == ["nominalizacion", "metafora", "figura"]
    assert report.present() is True


def test_scout_present_false_when_no_high_or_medium_confidence():
    llm = FakeLLM(
        {
            "summary": "nada",
            "resources": [{"name": "figura", "confidence": "baja"}],
        }
    )
    report = ScoutAgent(llm).analyze(make_workflow(), {}, "t", "g")
    assert report.present() is False


def test_scout_analyze_returns_empty_report_on_llm_exception():
    agent = ScoutAgent(RaisingLLM())
    report = agent.analyze(make_workflow(), {}, "t", "g")

    assert isinstance(report, ScoutReport)
    assert report.workflow == "cohesion"
    assert report.resources == []
    assert report.summary == ""
    assert report.present() is False


# ─────────────────────────────────────────────────────────────────────────────
# OrchestratorAgent
# ─────────────────────────────────────────────────────────────────────────────


def test_orchestrator_plan_returns_depth_and_skipped_agents():
    llm = FakeLLM({"depth": 2, "reasoning": "x", "skipped_agents": []})
    workflow = make_workflow(n_layers=3)
    scout = ScoutReport(workflow="cohesion")
    agent = OrchestratorAgent(llm)

    plan = agent.plan(workflow, scout, "g")

    assert isinstance(plan, DepthPlan)
    assert plan.depth == 2
    assert plan.reasoning == "x"
    # capas > 2 → a3
    assert plan.skipped_agents == ["a3"]


def test_orchestrator_clamps_depth_too_high():
    llm = FakeLLM({"depth": 99, "reasoning": "x", "skipped_agents": []})
    workflow = make_workflow(n_layers=3)
    plan = OrchestratorAgent(llm).plan(workflow, ScoutReport(workflow="cohesion"), "g")
    assert plan.depth == 3
    assert plan.skipped_agents == []


def test_orchestrator_clamps_depth_too_low():
    llm = FakeLLM({"depth": 0, "reasoning": "x", "skipped_agents": []})
    workflow = make_workflow(n_layers=3)
    plan = OrchestratorAgent(llm).plan(workflow, ScoutReport(workflow="cohesion"), "g")
    assert plan.depth == 1
    assert plan.skipped_agents == ["a2", "a3"]


def test_orchestrator_plan_returns_full_depth_on_llm_exception():
    workflow = make_workflow(n_layers=3)
    plan = OrchestratorAgent(RaisingLLM()).plan(
        workflow, ScoutReport(workflow="cohesion"), "g"
    )
    assert plan.depth == 3
    assert plan.skipped_agents == []


# ─────────────────────────────────────────────────────────────────────────────
# WorkerAgent
# ─────────────────────────────────────────────────────────────────────────────


def test_worker_analyze_delegates_to_base_agent():
    base = FakeBaseAgent(result=[{"agent": "a1"}])
    workflow = make_workflow()
    spec = workflow.agents[0]
    worker = WorkerAgent(spec, base)

    result = worker.analyze("texto", "resumen")

    assert result == [{"agent": "a1"}]
    assert len(base.calls) == 1
    tagged_text, grammar_summary, previous = base.calls[0]
    assert tagged_text == "texto"
    assert grammar_summary == "resumen"
    assert previous == []


def test_worker_analyze_passes_previous_through():
    base = FakeBaseAgent()
    workflow = make_workflow()
    worker = WorkerAgent(workflow.agents[0], base)

    worker.analyze("t", "g", previous=[{"x": 1}])

    assert base.calls[0][2] == [{"x": 1}]


def test_worker_properties():
    workflow = make_workflow()
    spec = workflow.agents[1]  # a2, capa 2
    worker = WorkerAgent(spec, FakeBaseAgent())

    assert worker.name == "a2"
    assert worker.layer == 2
    assert worker.disc_cats == ("a1",)
