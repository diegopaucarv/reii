"""Agentes LangChain del análisis del discurso.

Tres roles, alineados con el rediseño multiagente:

* :class:`ScoutAgent` — **explorador** por workflow. Produce un reporte
  introductorio de los recursos discursivos (los que analizan los agentes de
  ese workflow) presentes en el texto, relacionándolos con los resultados
  estadísticos del resumidor gramatical. Es barato: una sola llamada por
  workflow, sin anotar spans.

* :class:`OrchestratorAgent` — **orquestador** por workflow. Decide hasta qué
  capa (profundidad) llevar el análisis a partir del reporte del explorador y
  del resumen gramatical. El objetivo es **ahorrar agentes**: si un recurso no
  está presente, se saltan las capas que dependen de él.

* :class:`WorkerAgent` — **trabajador**. Envuelve un :class:`BaseAgent`
  existente (el que anota spans) y lo asocia a su :class:`AgentSpec` (capa y
  dependencias) para que el grafo lo ejecute en orden topológico.

Todos los agentes dependen de un LLM inyectable con el método
``invoke_structured(messages, schema) -> dict`` (ver
:class:`~reii.pipelines.langchain_llm.TogetherChatModel`). En pruebas se
inyecta un stub para correr **offline**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from reii.pipelines.dependency_graph import AgentSpec, WorkflowSpec

# ─────────────────────────────────────────────────────────────────────────────
# Modelos de datos
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScoutReport:
    """Reporte introductorio del explorador para un workflow."""

    workflow: str
    resources: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def resource_names(self) -> List[str]:
        return [r.get("name", "") for r in self.resources if r.get("name")]

    def present(self) -> bool:
        """True si se detectó al menos un recurso con confianza media/alta."""
        return any(r.get("confidence") in ("alta", "media") for r in self.resources)


@dataclass
class DepthPlan:
    """Decisión de profundidad del orquestador para un workflow."""

    workflow: str
    depth: int = 1  # capa máxima a ejecutar (1..n_layers)
    reasoning: str = ""
    skipped_agents: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Esquemas JSON de salida
# ─────────────────────────────────────────────────────────────────────────────

SCOUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "resources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["alta", "media", "baja"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["name", "confidence"],
            },
        },
    },
    "required": ["summary", "resources"],
}

ORCHESTRATOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "depth": {"type": "integer", "minimum": 1},
        "reasoning": {"type": "string"},
        "skipped_agents": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["depth", "reasoning"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Explorador (scout)
# ─────────────────────────────────────────────────────────────────────────────


class ScoutAgent:
    """Explorador por workflow: reporte introductorio de recursos discursivos."""

    def __init__(self, llm: Any):
        self.llm = llm

    def analyze(
        self,
        workflow: WorkflowSpec,
        agent_roles: Dict[str, str],
        tagged_text: str,
        grammar_summary: str,
    ) -> ScoutReport:
        """Detecta qué recursos discursivos del workflow están presentes.

        ``agent_roles`` mapea nombre de agente → descripción de su rol (lo que
        analiza). Se usa para enumerar los recursos a buscar.
        """
        agent_lines = "\n".join(
            f"- {name}: {agent_roles.get(name, '')}"
            for name in workflow.topological_order()
        )
        prompt = f"""## Rol
Eres un analista del discurso que hace un barrido rápido (sin anotar spans) para
decidir qué recursos discursivos merecen un análisis profundo.

## Workflow
{workflow.display_name} — {workflow.description}

## Agentes de este workflow y qué analizan
{agent_lines}

## Resumen gramatical (resultados estadísticos)
{grammar_summary}

## Texto del corpus (con identificadores de UCE)
```
{tagged_text}
```

## Tarea
Para cada agente del workflow, indica si su recurso discursivo está presente en
el texto. Relaciona tu decisión con las estadísticas del resumen gramatical
(p. ej. "hay muchas nominalizaciones porque el resumen muestra densidad nominal
alta"). Devuelve un JSON con:
- `summary`: texto breve (máx. 200 palabras) que sintetice los hallazgos.
- `resources`: lista de objetos con `name` (nombre del recurso/agente),
  `confidence` (alta/media/baja) y `evidence` (cita o señal breve).

Si un recurso no aparece, márcalo con `confidence: "baja"` (no lo omitas).
"""
        messages: Sequence[Any] = [
            SystemMessage(
                content="Eres un lingüista computacional estricto. Devuelve "
                "únicamente JSON válido según el esquema."
            ),
            HumanMessage(content=prompt),
        ]
        try:
            data = self.llm.invoke_structured(messages, SCOUT_SCHEMA)
        except Exception as e:  # pragma: no cover - defensivo
            print(f"⚠️ Explorador de '{workflow.name}' falló: {e}")
            return ScoutReport(workflow=workflow.name)
        resources = data.get("resources", []) if isinstance(data, dict) else []
        return ScoutReport(
            workflow=workflow.name,
            resources=resources,
            summary=data.get("summary", "") if isinstance(data, dict) else "",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Orquestador
# ─────────────────────────────────────────────────────────────────────────────


class OrchestratorAgent:
    """Orquestador por workflow: decide la profundidad del análisis."""

    def __init__(self, llm: Any):
        self.llm = llm

    def plan(
        self,
        workflow: WorkflowSpec,
        scout: ScoutReport,
        grammar_summary: str,
    ) -> DepthPlan:
        """Decide hasta qué capa ejecutar, para ahorrar agentes.

        Devuelve un :class:`DepthPlan` con ``depth`` en ``[1, n_layers]``. Si el
        LLM falla, se devuelve profundidad total (comportamiento seguro).
        """
        layer_lines = "\n".join(
            f"- Capa {i}: {', '.join(a.name for a in layer)}"
            for i, layer in enumerate(workflow.layers, 1)
        )
        resources = "\n".join(
            f"- {r.get('name', '?')} ({r.get('confidence', '?')})"
            for r in scout.resources
        )
        prompt = f"""## Rol
Eres un orquestador de un sistema multiagente de análisis del discurso. Tu
objetivo es **ahorrar agentes**: decidir hasta qué capa de análisis merece la
pena llegar, sin perder hallazgos importantes.

## Workflow
{workflow.display_name} — {workflow.description}

## Capas (las capas altas dependen de las bajas)
{layer_lines}

## Reporte del explorador
{resources or "(sin recursos detectados)"}

## Resumen gramatical
{grammar_summary}

## Tarea
Elige la profundidad `depth` (1..{workflow.n_layers}) que debe ejecutarse.
Reglas:
- Si el explorador no detectó recursos relevantes, usa una profundidad baja.
- Si detectó recursos de capas altas (p. ej. figuras, valoración, mitopoética),
  sube la profundidad para cubrirlos.
- No saltes una capa intermedia si una capa superior la necesita.

Devuelve un JSON con:
- `depth`: entero (1..{workflow.n_layers}).
- `reasoning`: justificación breve.
- `skipped_agents`: lista de nombres de agentes que se omiten (capas > depth).
"""
        messages: Sequence[Any] = [
            SystemMessage(
                content="Eres un planificador estricto. Devuelve únicamente "
                "JSON válido según el esquema."
            ),
            HumanMessage(content=prompt),
        ]
        try:
            data = self.llm.invoke_structured(messages, ORCHESTRATOR_SCHEMA)
        except Exception as e:  # pragma: no cover - defensivo
            print(f"⚠️ Orquestador de '{workflow.name}' falló: {e}")
            return DepthPlan(workflow=workflow.name, depth=workflow.n_layers)

        depth = (
            data.get("depth", workflow.n_layers)
            if isinstance(data, dict)
            else workflow.n_layers
        )
        try:
            depth = int(depth)
        except (TypeError, ValueError):
            depth = workflow.n_layers
        depth = max(1, min(depth, workflow.n_layers))

        skipped = [a.name for a in workflow.agents if a.layer > depth]
        return DepthPlan(
            workflow=workflow.name,
            depth=depth,
            reasoning=data.get("reasoning", "") if isinstance(data, dict) else "",
            skipped_agents=skipped,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Trabajador
# ─────────────────────────────────────────────────────────────────────────────


class WorkerAgent:
    """Trabajador: envuelve un :class:`BaseAgent` y lo asocia a su spec."""

    def __init__(self, agent_spec: AgentSpec, base_agent: Any):
        self.agent_spec = agent_spec
        self.base_agent = base_agent

    @property
    def name(self) -> str:
        return self.agent_spec.name

    @property
    def layer(self) -> int:
        return self.agent_spec.layer

    @property
    def disc_cats(self) -> Tuple[str, ...]:
        return self.agent_spec.disc_cats

    def analyze(
        self,
        tagged_text: str,
        grammar_summary: str,
        previous: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Ejecuta el agente subyacente y devuelve sus anotaciones."""
        return self.base_agent.analyze_corpus(
            tagged_text, grammar_summary, previous or []
        )
