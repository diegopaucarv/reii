"""Descripción declarativa de los tres pipelines: qué leen, qué escriben y dónde se ven.

La pestaña de configuración usa este mapa para explicar cada parte del sistema.
Solo se afirma lo que el código hace hoy; los cambios que introducen los parches de
``scripts/apply_pipeline_patches.py`` están marcados con la etiqueta (tras el parche).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Pipeline:
    pid: str
    number: int
    title: str
    icon: str
    summary: str
    reads: Tuple[str, ...]
    writes: Tuple[str, ...]
    config: str
    shown_in: str
    state_key: str


PIPELINES: Tuple[Pipeline, ...] = (
    Pipeline(
        pid="batch",
        number=1,
        title="Transcripciones",
        icon="🗂️",
        summary="Procesa los archivos de transcripción con el LLM y deja las entrevistas listas para exportar al workflow.",
        reads=("Archivos de transcripción cargados",),
        writes=(
            "entrevistas.json (registros por archivo)",
            "data/txt_outputs/tmp/ (JSON exportados)",
        ),
        config="Parámetros del lote (gestor de transcripciones)",
        shown_in="Gestor de transcripciones",
        state_key="batch_worker_state",
    ),
    Pipeline(
        pid="workflow",
        number=2,
        title="Clasificación",
        icon="📊",
        summary="Segmenta en UCE/UC, clasifica en clases (CDH), calcula términos, estabilidad y, si está activo, RF/SHAP.",
        reads=("data/txt_outputs/tmp/ (JSON exportados)",),
        writes=(
            "PostgreSQL: uces, ucs, clusters, terms, kv_store",
            "data/workflow_data.json (copia JSON)",
        ),
        config="Modo classic: workflow_config.json · Modo transformer: tabla workflow_config (ConfigStore). Tras el parche ambos se escriben a la vez.",
        shown_in="Pestañas A, B, C y D",
        state_key="workflow_state",
    ),
    Pipeline(
        pid="discourse",
        number=3,
        title="Análisis del discurso (IA)",
        icon="🧠",
        summary="Flujo multiagente (LangGraph): un resumen gramatical compartido y, por workflow (Cohesión/Crítico/Léxico), un explorador, un orquestador que decide la profundidad y trabajadores que anotan en capas con dependencias.",
        reads=("data/workflow_data.json (hoy) · BD PostgreSQL (tras el parche)",),
        writes=("data/discourse_state.json", "tabla annotations (tras sincronizar)"),
        config="ia/analisis_discursivo/*.json (workflows y agentes) y resumidor gramatical; capas y dependencias en dependency_graph.py",
        shown_in="Pestaña C, última sección",
        state_key="discourse_state",
    ),
)

BY_ID: Dict[str, Pipeline] = {p.pid: p for p in PIPELINES}

#: Frase corta para el encabezado de la pestaña
HEADLINE = (
    "Esta pestaña solo configura y ejecuta. Los resultados se ven en las pestañas A–D "
    "(clasificación) y en la última sección de la pestaña C (figuras del discurso)."
)
