"""Orquestación LangGraph del análisis del discurso.

Une la Fase 1 (resumen gramatical, ya calculado por el llamador) con los tres
workflows, respetando las **relaciones de dependencia** (capas) y la
**profundidad** decidida por el orquestador.

Patrones multiagente implementados:

* **Supervisor** — el orquestador (nodo ``plan``) supervisa a los trabajadores
  (nodo ``execute``).
* **Planner-Executor** — el orquestador planifica la profundidad; los
  trabajadores ejecutan.
* **Router** — el orquestador enruta a una profundidad (capas) para **ahorrar
  agentes** cuando un recurso no está presente.
* **Jerárquico** — los trabajadores se organizan en capas con dependencias
  (capa N depende de la capa N-1).
* **Map-Reduce** — los trabajadores procesan por documento (map) y las
  anotaciones se agregan (reduce).

El grafo es **inyectable**: ``llm`` (explorador/orquestador) y ``executor``
(trabajadores) se pasan por constructor, de modo que todo el flujo se puede
probar **offline** con stubs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from reii.pipelines.dependency_graph import WorkflowSpec
from reii.pipelines.discourse_agents import (
    DepthPlan,
    OrchestratorAgent,
    ScoutAgent,
    ScoutReport,
)


class DiscourseState(TypedDict, total=False):
    """Estado compartido del grafo."""

    grammar_summary: str
    tagged_text: str
    scout_reports: Dict[str, ScoutReport]
    depth_plans: Dict[str, DepthPlan]
    annotations: List[Any]


class DiscourseGraph:
    """Orquesta scout → plan → execute sobre los tres workflows."""

    def __init__(
        self,
        llm: Any,
        workflow_specs: Dict[str, WorkflowSpec],
        agent_roles: Dict[str, str],
        executor: Callable[[str], List[Any]],
    ):
        self.llm = llm
        self.workflow_specs = workflow_specs
        self.agent_roles = agent_roles
        self.executor = executor
        self.scout = ScoutAgent(llm)
        self.planner = OrchestratorAgent(llm)

    # ── nodos ──────────────────────────────────────────────────────────────
    def _scout(self, state: DiscourseState) -> DiscourseState:
        reports: Dict[str, ScoutReport] = {}
        for name, spec in self.workflow_specs.items():
            reports[name] = self.scout.analyze(
                spec,
                self.agent_roles,
                state["tagged_text"],
                state["grammar_summary"],
            )
        state["scout_reports"] = reports
        return state

    def _plan(self, state: DiscourseState) -> DiscourseState:
        plans: Dict[str, DepthPlan] = {}
        for name, spec in self.workflow_specs.items():
            report = state["scout_reports"].get(name) or ScoutReport(workflow=name)
            plans[name] = self.planner.plan(spec, report, state["grammar_summary"])
        state["depth_plans"] = plans
        return state

    def _execute(self, state: DiscourseState) -> DiscourseState:
        annotations: List[Any] = []
        for name, spec in self.workflow_specs.items():
            plan = state["depth_plans"].get(name)
            depth = plan.depth if plan else spec.n_layers
            # Orden topológico (capa a capa) → las dependencias se ejecutan
            # antes que sus dependientes. Se respeta la profundidad decidida.
            for agent_name in spec.topological_order():
                layer = spec.layer_of(agent_name)
                if layer is not None and layer > depth:
                    continue
                annotations.extend(self.executor(agent_name))
        state["annotations"] = annotations
        return state

    # ── construcción ───────────────────────────────────────────────────────
    def build(self):
        graph = StateGraph(DiscourseState)
        graph.add_node("scout", self._scout)
        graph.add_node("plan", self._plan)
        graph.add_node("execute", self._execute)
        graph.add_edge(START, "scout")
        graph.add_edge("scout", "plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", END)
        return graph.compile()

    def run(self, grammar_summary: str, tagged_text: str) -> Dict[str, Any]:
        """Ejecuta el grafo completo y devuelve el estado final."""
        initial: DiscourseState = {
            "grammar_summary": grammar_summary,
            "tagged_text": tagged_text,
            "scout_reports": {},
            "depth_plans": {},
            "annotations": [],
        }
        return self.build().invoke(initial)


def build_discourse_graph(
    llm: Any,
    workflow_specs: Dict[str, WorkflowSpec],
    agent_roles: Dict[str, str],
    executor: Callable[[str], List[Any]],
) -> DiscourseGraph:
    """Factory de conveniencia: construye un :class:`DiscourseGraph`."""
    return DiscourseGraph(llm, workflow_specs, agent_roles, executor)
