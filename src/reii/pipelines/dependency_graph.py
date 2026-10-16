"""Modelo explícito de dependencias entre workflows, capas y agentes.

Este módulo es la **única fuente de verdad** sobre cómo se relacionan los
agentes de análisis del discurso. Deriva las capas (``layers``) a partir de
``disc_cats`` declarado en cada JSON de workflow y expone consultas de orden
topológico, dependencias y capas para que el orquestador (LangGraph) y la UI
(``hub_tab``) respeten las relaciones de dependencia del diagrama de
referencia (Fase 1 → Workflow 1/2/3, con capas encadenadas dentro de cada
workflow).

Reglas de derivación de capas (longest-path layering sobre el DAG de
``disc_cats``):

* Un agente sin dependencias discursivas está en la **capa 1**.
* Un agente con dependencias está en la capa ``1 + max(capa de sus
  dependencias)``.

Los agentes "vacíos" (sin ``role`` ni ``instructions``, p. ej. el placeholder
``dssf``) se omiten: no son agentes reales y no deben entrar en el grafo.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class AgentSpec:
    """Descripción inmutable de un agente dentro de un workflow."""

    name: str
    workflow: str
    layer: int
    gram_cats: Tuple[str, ...] = ()
    disc_cats: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowSpec:
    """Un workflow (Cohesión / Crítico / Léxico) con sus capas ordenadas.

    ``layers`` es una tupla de tuplas: ``layers[0]`` es la capa 1, ``layers[1]``
    la capa 2, etc. Cada capa es una tupla de :class:`AgentSpec` (sin orden
    significativo dentro de la capa, ya que no hay dependencias entre ellos).
    """

    name: str
    display_name: str
    description: str
    layers: Tuple[Tuple[AgentSpec, ...], ...]

    @property
    def agents(self) -> Tuple[AgentSpec, ...]:
        """Todos los agentes, en orden topológico (capa a capa)."""
        return tuple(a for layer in self.layers for a in layer)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def layer_of(self, agent_name: str) -> Optional[int]:
        for a in self.agents:
            if a.name == agent_name:
                return a.layer
        return None

    def dependencies(self, agent_name: str) -> Set[str]:
        """Dependencias discursivas directas de ``agent_name``."""
        for a in self.agents:
            if a.name == agent_name:
                return set(a.disc_cats)
        return set()

    def agents_in_layer(self, layer: int) -> Tuple[AgentSpec, ...]:
        if 1 <= layer <= self.n_layers:
            return self.layers[layer - 1]
        return ()

    def topological_order(self) -> List[str]:
        """Nombres de agentes en orden topológico (capa a capa)."""
        return [a.name for a in self.agents]


#: Metadatos de los tres workflows (nombre visible + descripción), alineados
#: con el diagrama de referencia.
WORKFLOW_META: Dict[str, Tuple[str, str]] = {
    "cohesion": (
        "Cohesión",
        "Cómo se conecta el texto y cómo se gestiona el desacuerdo",
    ),
    "critico": (
        "Crítico",
        "Estrategias de poder y ocultamiento de la agencia",
    ),
    "lexico": (
        "Léxico",
        "El léxico como ideología y como figura",
    ),
}


def workflow_name_from_filename(filename: str) -> str:
    """Deriva el nombre corto de un workflow a partir de su nombre de archivo.

    ``discurso_cohesion.json`` → ``cohesion``, ``discurso_critico.json`` →
    ``critico``, ``discurso_lexico.json`` → ``lexico``.
    """
    base = os.path.basename(filename)
    if base.startswith("discurso_"):
        base = base[len("discurso_") :]
    if base.endswith(".json"):
        base = base[: -len(".json")]
    return base


def _is_placeholder(cfg: Dict) -> bool:
    """Un agente sin ``role`` ni ``instructions`` es un placeholder, no un agente."""
    return not cfg.get("role") and not cfg.get("instructions")


def build_workflow_spec(
    name: str,
    agents: List[Dict],
    display_name: Optional[str] = None,
    description: Optional[str] = None,
) -> WorkflowSpec:
    """Construye un :class:`WorkflowSpec` a partir de una lista de configs.

    ``agents`` es una lista de dicts con, al menos, ``name``, ``gram_cats`` y
    ``disc_cats`` (el formato de los JSON de ``ia/analisis_discursivo/``).
    Las capas se derivan con longest-path layering sobre ``disc_cats``.
    """
    display_name = display_name or WORKFLOW_META.get(name, (name, ""))[0]
    description = description or WORKFLOW_META.get(name, ("", ""))[1]

    active: List[Dict] = [cfg for cfg in agents if not _is_placeholder(cfg)]
    by_name: Dict[str, Dict] = {cfg.get("name", ""): cfg for cfg in active}

    # Capa de cada agente (longest-path). Se resuelve iterativamente porque las
    # dependencias pueden aparecer en cualquier orden dentro del JSON.
    layer: Dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for cfg in active:
            n = cfg.get("name", "")
            if n in layer:
                continue
            deps = [d for d in cfg.get("disc_cats", []) if d in by_name]
            if not deps:
                layer[n] = 1
                changed = True
            elif all(d in layer for d in deps):
                layer[n] = 1 + max(layer[d] for d in deps)
                changed = True
            # else: espera a que sus dependencias se resuelvan en otra pasada

    # Si quedan agentes sin capa (dependencia rota o ciclo), se colocan en la
    # capa 1 para no romper el grafo; el orquestador puede advertirlo.
    for cfg in active:
        n = cfg.get("name", "")
        if n not in layer:
            layer[n] = 1

    max_layer = max(layer.values(), default=0)
    layers: List[List[AgentSpec]] = [[] for _ in range(max_layer)]
    for cfg in active:
        n = cfg.get("name", "")
        layers[layer[n] - 1].append(
            AgentSpec(
                name=n,
                workflow=name,
                layer=layer[n],
                gram_cats=tuple(cfg.get("gram_cats", [])),
                disc_cats=tuple(cfg.get("disc_cats", [])),
            )
        )

    return WorkflowSpec(
        name=name,
        display_name=display_name,
        description=description,
        layers=tuple(tuple(l) for l in layers),
    )


def load_workflow_specs(config_dir: str) -> Dict[str, WorkflowSpec]:
    """Carga todos los workflows de ``config_dir`` (``ia/analisis_discursivo/``).

    Devuelve ``{nombre_corto: WorkflowSpec}``. Se ignoran los archivos que no
    son un dict de agentes (p. ej. ``resumidor_gramatical.json``, que es un
    dict único con ``name`` pero sin estructura de workflow, o archivos
    ``_``-prefijados).
    """
    specs: Dict[str, WorkflowSpec] = {}
    if not os.path.isdir(config_dir):
        return specs
    for filename in sorted(os.listdir(config_dir)):
        if not filename.endswith(".json") or filename.startswith("_"):
            continue
        path = os.path.join(config_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        # Un workflow es un dict de agentes (claves = nombres de agente, valores
        # = dicts con "name"/"role"). El resumidor gramatical es un único agente
        # con "name" como clave de nivel superior (string), no un workflow.
        if "name" in data and isinstance(data.get("name"), str):
            continue
        agents = [v for v in data.values() if isinstance(v, dict)]
        if not agents:
            continue
        name = workflow_name_from_filename(filename)
        specs[name] = build_workflow_spec(name, agents)
    return specs


def validate_dependencies(specs: Dict[str, WorkflowSpec]) -> List[str]:
    """Devuelve una lista de advertencias sobre dependencias rotas o ciclos.

    Cada advertencia es un string legible. Vacío = todo correcto.
    """
    warnings: List[str] = []
    for wf_name, spec in specs.items():
        names = {a.name for a in spec.agents}
        for a in spec.agents:
            for dep in a.disc_cats:
                if dep not in names:
                    warnings.append(
                        f"[{wf_name}] '{a.name}' depende de '{dep}', que no "
                        f"existe en el workflow."
                    )
    return warnings
