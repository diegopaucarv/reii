"""Gestión de transcripciones y editores de entrevistas/prompts.

Contiene la UI de procesamiento por lotes (subida, estado, cola, log,
parámetros y exportación al workflow) y los editores de configuración de
entrevistas (guía estructurada, prompts y localismos). Se separa de
``dashboard.py`` para poder probarla sin ejecutar la app completa.

Los workers ``_batch_worker`` y ``_workflow_runner`` viven aquí (antes en
``dashboard.py``) para que la UI y las pruebas puedan reutilizarlos sin
importar la app entera.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from reii.batch_processor import BatchProcessor, load_localismos
from reii.config import (
    BATCH_DB_PATH,
    GRAMATICA_PATH,
    LOCALISMOS_PATH,
    TRANSCRIPTS_DIR,
    WORKFLOW_CONFIG_PATH,
    WORKFLOW_DB_PATH,
)
from reii.guia_entrevista import guia_como_texto
from reii.ia_config_editor import (
    load_entrevista_prompt,
    load_guia_entrevista,
    normalize_guia,
    save_entrevista_prompt,
    save_guia_entrevista,
    sync_seccion_enum,
    validate_guia,
)

# Importación defensiva del módulo de gramática (lo crea otro agente).
# Si aún no existe, el editor muestra un aviso en lugar de romper la app.
try:
    from reii.gramatica import load_gramatica, save_gramatica

    _GRAMATICA_DISPONIBLE = True
except ImportError:
    _GRAMATICA_DISPONIBLE = False
    load_gramatica = None  # type: ignore[assignment]
    save_gramatica = None  # type: ignore[assignment]

# Directorio raíz del paquete ``reii`` (donde viven main_workflow*.py).
_REII_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Número de transcripciones procesadas en paralelo por el worker de lotes.
# Las llamadas al LLM son I/O-bound (red), así que un pool pequeño de hilos
# solapa la latencia sin saturar la API.
_BATCH_MAX_WORKERS = 4


def _batch_worker(
    bp: "BatchProcessor",
    queue: List[str],
    log: List[str],
    state: Dict[str, Any],
) -> None:
    """Procesa la cola en segundo plano, varios archivos en paralelo.

    Revisa ``state["paused"]`` antes de tomar cada archivo: si el usuario
    pausa, no se inician archivos nuevos y se espera a que terminen los que
    ya están en vuelo. ``state`` es un dict mutable compartido con la UI
    (evita tocar ``st.session_state`` desde el hilo).

    ``bp.save_record`` hace un read-modify-write sobre el JSON compartido,
    así que se serializa con un lock. ``log.append`` es atómico (GIL), pero
    se protege igual para mantener el orden de los mensajes por archivo.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    save_lock = threading.Lock()
    log_lock = threading.Lock()
    pop_lock = threading.Lock()

    def _log(msg: str) -> None:
        with log_lock:
            log.append(msg)

    def _process_one() -> None:
        # Pop perezoso: la cola se vacía a medida que los archivos terminan,
        # así el contador "N en cola" de la UI refleja el trabajo restante.
        while True:
            with pop_lock:
                if not queue or state["paused"]:
                    return
                path_str = queue.pop(0)
            path = Path(path_str)
            nombre = path.name
            _log(f"▶ Procesando {nombre}…")
            try:
                record = bp.process_file(path, log_cb=_log)
                with save_lock:
                    bp.save_record(record)
                icono = "✅" if record.get("estado") == "completado" else "❌"
                _log(f"{icono} {nombre}: {record.get('estado')}")
            except Exception as e:
                _log(f"❌ {nombre}: {type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=_BATCH_MAX_WORKERS) as ex:
        futures = [ex.submit(_process_one) for _ in range(_BATCH_MAX_WORKERS)]
        for f in as_completed(futures):
            f.result()

    if state["paused"]:
        _log("⏸ Lote pausado.")
    else:
        _log("✅ Lote completado.")
    state["running"] = False
    state["finished"] = True


def _workflow_runner(
    log: List[str],
    state: Dict[str, Any],
    params: Dict[str, Any],
) -> None:
    """Ejecuta el workflow principal en un subproceso y captura su salida.

    El workflow (``main_workflow.py``) lee los JSON exportados en
    ``data/txt_outputs/tmp/`` y escribe ``data/workflow_data.json``. Se corre
    como subproceso para no bloquear el hilo de Streamlit; su stdout/stderr
    se vuelcan línea a línea en ``log`` para mostrarlo en vivo.
    """
    # Persistir los parámetros elegidos para que el subproceso los lea.
    os.makedirs(os.path.dirname(WORKFLOW_CONFIG_PATH), exist_ok=True)
    with open(WORKFLOW_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    try:
        from reii.pipelines.config_sync import push_params_to_store

        _pp = push_params_to_store(params)
        log.append(f"⚙️ Configuración enviada a la BD: {len(_pp['pushed'])} claves.")
    except Exception as _qe:
        log.append(f"⚠️ No se pudo enviar la configuración a la BD: {_qe}")

    mode = params.get("workflow_mode", "classic")
    if mode == "transformer":
        script = os.path.join(_REII_DIR, "main_workflow.py")
    else:
        script = os.path.join(_REII_DIR, "main_workflow_clasico.py")
    log.append(f"🚀 Ejecutando workflow: python {script}")
    try:
        proc = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for linea in proc.stdout:
            linea = linea.rstrip()
            if linea:
                log.append(linea)
        code = proc.wait()
        if code == 0:
            log.append("✅ Workflow completado sin errores.")
        else:
            log.append(f"❌ Workflow terminó con código {code}.")
    except Exception as e:
        log.append(f"❌ Error al ejecutar el workflow: {type(e).__name__}: {e}")
    state["running"] = False
    state["finished"] = True
    state["exit_code"] = code if "code" in locals() else -1


# ════════════════════════════════════════════════════════════════════════
# Gestión de transcripciones (subida, estado, cola, log, parámetros, export)
# ════════════════════════════════════════════════════════════════════════


def render_transcripciones() -> None:
    """Gestión de transcripciones: subida, estado, cola, log, parámetros y exportación.

    Reutiliza ``BatchProcessor``, ``_batch_worker`` y ``_workflow_runner`` y
    las claves de sesión ``batch_queue``, ``batch_log``, ``batch_worker_state``,
    ``workflow_state`` y ``workflow_log``.
    """
    localismos_actuales = load_localismos()
    bp = BatchProcessor(db_path=BATCH_DB_PATH, localismos=localismos_actuales)

    # ── Estado de la cola (persistente entre reruns) ─────────────────
    if "batch_queue" not in st.session_state:
        st.session_state.batch_queue = []  # List[str] de rutas pendientes
    if "batch_log" not in st.session_state:
        st.session_state.batch_log = []  # List[str] de mensajes de progreso/error
    if "batch_worker_state" not in st.session_state:
        # dict mutable compartido con el hilo worker (no tocar session_state
        # desde el hilo): {"paused": bool, "running": bool, "finished": bool}
        st.session_state.batch_worker_state = {
            "paused": False,
            "running": False,
            "finished": False,
        }

    # ── 1. Subida y selección de archivos ────────────────────────────
    st.markdown("**1 · Archivos de transcripción**")
    uploaded = st.file_uploader(
        "Subir transcripciones (.txt, .json, .md)",
        type=["txt", "json", "md"],
        accept_multiple_files=True,
        key="reii_batch_uploader",
    )
    if uploaded:
        os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
        saved_names = []
        skipped_names = []
        for up in uploaded:
            dest = os.path.join(TRANSCRIPTS_DIR, up.name)
            if os.path.exists(dest):
                skipped_names.append(up.name)
                continue
            with open(dest, "wb") as f:
                f.write(up.getbuffer())
            saved_names.append(up.name)
        if saved_names:
            st.success(
                f"Se guardaron {len(saved_names)} archivo(s) en `{TRANSCRIPTS_DIR}`: "
                + ", ".join(saved_names)
            )
        if skipped_names:
            st.info("Ya existían y no se sobrescribieron: " + ", ".join(skipped_names))

    discovered = bp.discover_files()
    if not discovered:
        st.info(
            f"No se encontraron transcripciones en `{TRANSCRIPTS_DIR}`. "
            "Sube archivos arriba o colócalos en el directorio."
        )
        return

    file_names = [p.name for p in discovered]
    selected = st.multiselect(
        "Transcripciones disponibles",
        options=file_names,
        default=file_names,
        key="reii_batch_selected",
    )
    if not selected:
        st.warning("Selecciona al menos una transcripción para procesar.")
        return
    targets = [p for p in discovered if p.name in selected]

    # ── 2. Estado de los documentos (consciente de lo ya procesado) ────
    st.markdown("**2 · Estado de los documentos**")
    records = bp.load_all()
    by_archivo = {r.get("archivo"): r for r in records}

    filas_estado = []
    for p in discovered:
        rec = by_archivo.get(p.name)
        if rec is None:
            filas_estado.append(
                {"Archivo": p.name, "Estado": "⏳ Pendiente", "Fecha": "—"}
            )
        elif rec.get("estado") == "completado":
            filas_estado.append(
                {
                    "Archivo": p.name,
                    "Estado": "✅ Procesado",
                    "Fecha": rec.get("fecha_procesamiento", "—"),
                }
            )
        else:
            filas_estado.append(
                {
                    "Archivo": p.name,
                    "Estado": "❌ Error",
                    "Fecha": rec.get("fecha_procesamiento", "—"),
                }
            )
    if filas_estado:
        st.dataframe(
            pd.DataFrame(filas_estado),
            width="stretch",
            hide_index=True,
            key="reii_batch_status_table",
        )
        n_ok = sum(1 for f in filas_estado if f["Estado"] == "✅ Procesado")
        n_err = sum(1 for f in filas_estado if f["Estado"] == "❌ Error")
        n_pend = sum(1 for f in filas_estado if f["Estado"] == "⏳ Pendiente")
        st.caption(
            f"{n_ok} procesado(s) · {n_pend} pendiente(s) · {n_err} con error "
            f"(de {len(filas_estado)} transcripciones)."
        )

    # ── 3. Cola de procesamiento (pausar / reanudar) ─────────────────
    st.markdown("**3 · Cola de procesamiento**")

    ws = st.session_state.batch_worker_state
    running = ws["running"]
    paused = ws["paused"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(
            "▶ Procesar seleccionados",
            type="primary",
            key="reii_batch_start",
            disabled=running,
        ):
            # Encola los pendientes; los que fallaron (estado "error") se
            # reintentan automáticamente.
            pendientes = [
                p
                for p in targets
                if p.name not in by_archivo
                or by_archivo[p.name].get("estado") == "error"
            ]
            if not pendientes:
                st.info("Todos los seleccionados ya están procesados.")
            else:
                st.session_state.batch_queue = [str(p) for p in pendientes]
                st.session_state.batch_log = []
                ws["paused"] = False
                ws["finished"] = False
                ws["running"] = True
                threading.Thread(
                    target=_batch_worker,
                    args=(
                        bp,
                        st.session_state.batch_queue,
                        st.session_state.batch_log,
                        ws,
                    ),
                    daemon=True,
                ).start()
                st.rerun()
    with col2:
        if st.button(
            "⏸ Pausar",
            key="reii_batch_pause",
            disabled=not running or paused,
        ):
            ws["paused"] = True
            st.rerun()
    with col3:
        if st.button(
            "▶ Reanudar",
            key="reii_batch_resume",
            disabled=not running or not paused,
        ):
            ws["paused"] = False
            st.rerun()
    with col4:
        if st.button(
            "✖ Detener y vaciar",
            key="reii_batch_clear",
            disabled=not running and not st.session_state.batch_queue,
        ):
            ws["paused"] = True
            st.session_state.batch_queue = []
            st.rerun()

    n_cola = len(st.session_state.batch_queue)
    if running:
        if paused:
            st.warning(
                "⏸ Pausa solicitada — se detendrá al terminar el archivo actual."
            )
        else:
            st.info(f"⏳ Procesando en segundo plano… {n_cola} archivo(s) en cola.")
    elif ws["finished"]:
        st.success("✅ Lote completado.")
    elif n_cola:
        st.caption(f"Cola: {n_cola} archivo(s) pendiente(s) · pausada.")

    # ── 4. Log de procesamiento (errores en vivo) ────────────────────
    st.markdown("**4 · Log de procesamiento**")
    with st.expander(
        f"📋 Log de procesamiento ({len(st.session_state.batch_log)} líneas)",
        expanded=bool(st.session_state.batch_log) or running,
    ):
        if st.session_state.batch_log:
            st.code("\n".join(st.session_state.batch_log[-300:]))
        else:
            st.caption("Sin actividad registrada todavía.")

    # ── 5. Parámetros del análisis ────────────────────────────────────
    st.markdown("**5 · Parámetros del análisis**")
    params: Dict[str, Any] = {}

    # --- Segmentación / NLP ---
    with st.expander("⚙️ Modos de workflow", expanded=True):
        mode = st.radio(
            "Modo de procesamiento",
            options=["classic", "transformer"],
            format_func=lambda v: {
                "classic": "📊 Classic (rápido, sin GPU)",
                "transformer": "🧠 Transformer (requiere GPU)",
            }[v],
            index=0,
            key="workflow_mode_v2",  # ← key nueva fuerza init limpia
        )
        st.caption(
            "Classic: spaCy, lexecon, RF/SHAP. "
            "Transformer: embeddings fine-tuned, BERTopic, jerárquico."
        )
        params["workflow_mode"] = mode
    c1, c2, c3 = st.columns(3)
    with c1:
        params["use_bigrams"] = st.radio(
            "Bigramas",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_bigrams",
        )
        params["stem_backend"] = st.selectbox(
            "Stemming",
            ["snowball", "none"],
            index=0,
            key="reii_p_stem_backend",
        )
    with c2:
        params["min_uce_words"] = st.number_input(
            "Mín. palabras por UCE",
            min_value=1,
            value=2,  # era 3; Recomendación C del plan
            key="reii_p_min_uce_words",
            help="UCEs con menos palabras que este umbral se descartan. "
            "Reinert/IRaMuTeQ recomienda 2.",
        )
        params["uce_target_size"] = st.number_input(
            "Objetivo palabras por UCE",
            min_value=8,
            max_value=60,
            value=18,  # era 40 por getattr; Recomendación C
            step=1,
            key="reii_p_uce_target_size",
            help="Ventana objetivo al segmentar. Reinert/IRaMuTeQ usan 12–18. "
            "Subirlo fragmenta menos pero diluye la señal temática.",
        )
    with c3:
        params["tsj"] = st.number_input(
            "Frecuencia mínima (tsj)",
            min_value=1,
            value=3,
            key="reii_p_tsj",
        )
        mf0 = st.number_input(
            "min_forms_uc (mín)", min_value=1, value=10, key="reii_p_mf0"
        )
        mf1 = st.number_input(
            "min_forms_uc (máx)", min_value=1, value=14, key="reii_p_mf1"
        )
        params["min_forms_uc"] = [int(mf0), int(mf1)]

    # --- CDH / Clasificación ---
    c1, c2, c3 = st.columns(3)
    with c1:
        params["use_cdh"] = st.radio(
            "CDH",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_cdh",
        )
    with c2:
        params["pseudocount"] = st.number_input(
            "Pseudocount",
            disabled=True,
            value=0.00,
            key="reii_p_pseudocount",
        )
    with c3:
        params["use_cah_per_class"] = st.radio(
            "CAH por clase",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_cah_per_class",
            disabled=not params["use_cdh"],
        )
        params["cah_per_class_top_terms"] = st.number_input(
            "Términos CAH por clase",
            min_value=1,
            value=50,
            key="reii_p_cah_top",
            disabled=not params["use_cdh"] or not params["use_cah_per_class"],
        )

    # --- Análisis complementarios ---
    c1, c2, c3 = st.columns(3)
    with c1:
        params["use_projection"] = st.radio(
            "Proyección (AFC)",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_projection",
        )
        params["analyze_metadata"] = st.radio(
            "Metadatos",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_analyze_metadata",
        )
    with c2:
        params["use_network_analysis"] = st.radio(
            "Red semántica",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_network",
        )
        params["use_term_stability"] = st.radio(
            "Estabilidad de términos",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_term_stability",
        )
    with c3:
        params["use_llm_synthesis"] = st.radio(
            "Síntesis LLM",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_llm",
        )
        params["use_multivariate_analysis"] = st.radio(
            "Análisis multivariado",
            [True, False],
            index=1,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_multivariate",
        )
        params["multivariate_metadata"] = st.multiselect(
            "Columnas multivariado",
            ["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"],
            default=["Edad_Cat", "Sexo", "Ocupacion_Cat", "Procedencia_Cat"],
            key="reii_p_multivariate_metadata",
            disabled=not params["use_multivariate_analysis"],
        )

    # --- Random Forest + SHAP ---
    c1, c2, c3 = st.columns(3)
    with c1:
        params["use_rf_shap"] = st.radio(
            "Random Forest + SHAP",
            [True, False],
            index=1,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_use_rf_shap",
        )
    with c2:
        params["rf_n_estimators"] = st.number_input(
            "RF n_estimators",
            min_value=1,
            value=200,
            key="reii_p_rf_n",
            disabled=not params["use_rf_shap"],
        )
        params["rf_max_depth"] = st.number_input(
            "RF max_depth",
            min_value=1,
            value=8,
            key="reii_p_rf_depth",
            disabled=not params["use_rf_shap"],
        )
    with c3:
        params["rf_outlier_method"] = st.selectbox(
            "RF outliers",
            ["iqr", "zscore"],
            index=0,
            key="reii_p_rf_outlier",
            disabled=not params["use_rf_shap"],
        )
        params["rf_cat_encoding"] = st.selectbox(
            "RF cat encoding",
            ["onehot", "frequency", "target"],
            index=1,
            key="reii_p_rf_cat",
            disabled=not params["use_rf_shap"],
        )

    # --- Optimización ---
    c1, c2, c3 = st.columns(3)
    with c1:
        params["optimize"] = st.radio(
            "Optimizar",
            [True, False],
            index=0,
            format_func=lambda v: "Sí" if v else "No",
            key="reii_p_optimize",
        )
    with c2:
        params["optimize_trials"] = st.number_input(
            "Trials de optimización",
            min_value=1,
            value=100,
            key="reii_p_optimize_trials",
            disabled=not params["optimize"],
        )
    with c3:
        params["random_state"] = st.number_input(
            "random_state", value=42, key="reii_p_random_state"
        )

    # ── 6. Exportar al workflow principal ────────────────────────────
    st.markdown("**6 · Exportar al workflow**")

    # Estado del subproceso del workflow (persistente entre reruns)
    if "workflow_state" not in st.session_state:
        st.session_state.workflow_state = {
            "running": False,
            "finished": False,
            "exit_code": None,
        }

    if "workflow_log" not in st.session_state:
        st.session_state.workflow_log = []

    wf_state = st.session_state.workflow_state
    wf_log = st.session_state.workflow_log

    if st.button(
        "📤 Exportar y correr workflow",
        key="reii_batch_export",
        disabled=wf_state["running"],
    ):
        try:
            exportado = bp.exportar_para_workflow()
            n_json = len(exportado["json"])
            if not n_json:
                st.info("No hay entrevistas completadas para exportar.")
            else:
                st.session_state.workflow_log = [
                    f"📤 Se exportaron {n_json} entrevista(s) a `data/txt_outputs/tmp/`."
                ]
                wf_state["running"] = True
                wf_state["finished"] = False
                wf_state["exit_code"] = None
                threading.Thread(
                    target=_workflow_runner,
                    args=(st.session_state.workflow_log, wf_state, params),
                    daemon=True,
                ).start()
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error al exportar: {e}")

    if wf_state["running"]:
        st.info("⏳ Corriendo el workflow principal… (puede tardar varios minutos)")
        with st.expander(
            f"📋 Log del workflow ({len(wf_log)} líneas)",
            expanded=True,
        ):
            if wf_log:
                st.code("\n".join(wf_log[-300:]))
            else:
                st.caption("Iniciando…")
    elif wf_state["finished"]:
        if wf_state["exit_code"] == 0:
            st.success("✅ Workflow completado. Cargando el dashboard…")
            with st.expander(
                f"📋 Log del workflow ({len(wf_log)} líneas)",
                expanded=False,
            ):
                st.code("\n".join(wf_log[-300:]))
            # El workflow ya escribió la base SQLite: al re-renderizar la página
            # principal, _load_snapshot() encontrará las UCEs y mostrará el
            # dashboard habitual. Solo re-renderiza si la base existe; si no,
            # deja el log visible para diagnosticar.
            if os.path.exists(WORKFLOW_DB_PATH):
                st.rerun()
        else:
            st.error(
                f"❌ El workflow terminó con código {wf_state['exit_code']}. "
                "Revisa el log para ver el error."
            )
            with st.expander(
                f"📋 Log del workflow ({len(wf_log)} líneas)",
                expanded=True,
            ):
                st.code("\n".join(wf_log[-300:]))

    # ── Polling: mientras el worker o el workflow corren, re-renderiza
    # cada 2s para mostrar el log en vivo.
    if st.session_state.batch_worker_state["running"] or wf_state["running"]:
        time.sleep(2)
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# Editores de configuración de entrevistas (localismos, prompts, guía)
# ════════════════════════════════════════════════════════════════════════


def _render_localismos_editor() -> None:
    st.markdown("**🌎 Localismos**")
    with st.expander(
        "🌎 Localismos (palabras que pudieron mal transcribirse)", expanded=False
    ):
        st.caption(
            "Lista de palabras que pudieron mal transcribirse (incluidos "
            "localismos). Sepáralas por comas. El modelo las usará como pistas "
            "para corregir la transcripción según el contexto."
        )
        texto_localismos = st.text_area(
            "Localismos (separados por comas)",
            value=", ".join(load_localismos()),
            height=200,
            key="reii_localismos_editor",
        )
        if st.button("💾 Guardar localismos", key="reii_localismos_save"):
            try:
                palabras = [
                    t.strip().lower() for t in texto_localismos.split(",") if t.strip()
                ]
                # Dedupe preservando el orden.
                vistos = set()
                unicas = []
                for p in palabras:
                    if p not in vistos:
                        vistos.add(p)
                        unicas.append(p)
                os.makedirs(os.path.dirname(LOCALISMOS_PATH), exist_ok=True)
                with open(LOCALISMOS_PATH, "w", encoding="utf-8") as f:
                    json.dump(unicas, f, ensure_ascii=False, indent=2)
                st.success(f"Localismos guardados en `{LOCALISMOS_PATH}`.")
                st.rerun()
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"❌ Error: {e}")


def _parse_lista(texto: str) -> List[str]:
    """Convierte texto separado por comas en una lista de strings limpios."""
    return [t.strip() for t in texto.split(",") if t.strip()]


def _parse_json_dict(texto: str, campo: str) -> Dict:
    """Convierte texto JSON en un dict; valida que sea un objeto."""
    data = json.loads(texto)
    if not isinstance(data, dict):
        raise ValueError(f"'{campo}' debe ser un objeto JSON.")
    return data


def _render_gramatica_editor() -> None:
    st.markdown("**🧩 Configuración gramatical**")
    with st.expander("🧩 Configuración gramatical (ALCESTE)", expanded=False):
        if not _GRAMATICA_DISPONIBLE:
            st.info("Módulo de gramática no disponible aún.")
            return

        st.caption(
            "Edita la configuración de gramática para el análisis ALCESTE. "
            "Los campos de lista se separan por comas; `palabras_activas` y "
            "`sinonimos` se editan como JSON."
        )

        cfg = load_gramatica() or {}

        content_pos = st.text_area(
            "content_pos (categorías gramaticales, separadas por comas)",
            value=", ".join(cfg.get("content_pos", [])),
            key="reii_gramatica_content_pos",
        )
        stopwords_extra = st.text_area(
            "stopwords_extra (separadas por comas)",
            value=", ".join(cfg.get("stopwords_extra", [])),
            key="reii_gramatica_stopwords_extra",
        )
        expresiones = st.text_area(
            "expresiones (separadas por comas)",
            value=", ".join(cfg.get("expresiones", [])),
            key="reii_gramatica_expresiones",
        )
        palabras_suplementarias = st.text_area(
            "palabras_suplementarias (separadas por comas)",
            value=", ".join(cfg.get("palabras_suplementarias", [])),
            key="reii_gramatica_palabras_suplementarias",
        )
        palabras_excluir = st.text_area(
            "palabras_excluir (separadas por comas)",
            value=", ".join(cfg.get("palabras_excluir", [])),
            key="reii_gramatica_palabras_excluir",
        )
        palabras_activas = st.text_area(
            "palabras_activas (JSON)",
            value=json.dumps(
                cfg.get("palabras_activas", {}), ensure_ascii=False, indent=2
            ),
            key="reii_gramatica_palabras_activas",
        )
        sinonimos = st.text_area(
            "sinonimos (JSON)",
            value=json.dumps(cfg.get("sinonimos", {}), ensure_ascii=False, indent=2),
            key="reii_gramatica_sinonimos",
        )

        if st.button("💾 Guardar", key="reii_gramatica_save"):
            try:
                data = {
                    "content_pos": _parse_lista(content_pos),
                    "stopwords_extra": _parse_lista(stopwords_extra),
                    "expresiones": _parse_lista(expresiones),
                    "palabras_suplementarias": _parse_lista(palabras_suplementarias),
                    "palabras_excluir": _parse_lista(palabras_excluir),
                    "palabras_activas": _parse_json_dict(
                        palabras_activas, "palabras_activas"
                    ),
                    "sinonimos": _parse_json_dict(sinonimos, "sinonimos"),
                }
                save_gramatica(data)
                st.success(f"Configuración gramatical guardada en `{GRAMATICA_PATH}`.")
                st.rerun()
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"❌ Error: {e}")


def _render_prompts_editor() -> None:
    st.markdown("**📝 Prompts de entrevista**")
    with st.expander(
        "📝 Prompts de entrevista (segmentación y extracción)", expanded=False
    ):
        st.caption(
            "Edita las instrucciones y el esquema JSON de cada ruta. "
            "El esquema de extracción define el formato de la metadata "
            "(datos_sociodemograficos) y de las respuestas. Formato: "
            '`{"name", "role", "instructions", "json_schema", "few_shot_examples"}`.'
        )
        for _pname, _plabel in (
            ("segmentacion", "Ruta 1 · Segmentación estructural"),
            ("extraccion", "Ruta 2 · Extracción y mapeo"),
        ):
            _pcfg = load_entrevista_prompt(_pname)
            st.markdown(f"**{_plabel}** — `ia/entrevistas/{_pname}.json`")
            _ptexto = st.text_area(
                f"Config JSON ({_pname})",
                value=json.dumps(_pcfg, ensure_ascii=False, indent=2),
                height=320,
                key=f"reii_prompt_{_pname}",
            )
            if st.button(
                f"💾 Guardar prompt {_pname}", key=f"reii_prompt_save_{_pname}"
            ):
                try:
                    _pdata = json.loads(_ptexto)
                    if not isinstance(_pdata, dict):
                        raise ValueError("El JSON debe ser un objeto.")
                    if not isinstance(_pdata.get("instructions"), str):
                        raise ValueError("Falta 'instructions' (string).")
                    if not isinstance(_pdata.get("json_schema"), dict):
                        raise ValueError("Falta 'json_schema' (objeto).")
                    save_entrevista_prompt(_pname, _pdata)
                    st.success(
                        f"Prompt `{_pname}` guardado en `ia/entrevistas/{_pname}.json`."
                    )
                    st.rerun()
                except (json.JSONDecodeError, ValueError) as e:
                    st.error(f"❌ JSON inválido: {e}")


def _collect_guia_from_widgets() -> Dict[str, Any]:
    """Reconstruye la guía desde los widgets del editor estructurado."""
    guia = st.session_state.get("guia_working", {})
    secciones = list(guia.get("secciones", []))
    preguntas = list(guia.get("preguntas", []))
    new_secciones = []
    for i, sec in enumerate(secciones):
        new_secciones.append(st.session_state.get(f"guia_sec_{i}", sec).strip())
    new_preguntas = []
    for i, sec in enumerate(secciones):
        new_sec_name = new_secciones[i]
        for j, p in enumerate(preguntas):
            if p.get("seccion") == sec:
                texto = st.session_state.get(
                    f"guia_q_{i}_{j}", p.get("pregunta", "")
                ).strip()
                if texto:
                    new_preguntas.append(
                        {"numero": 0, "seccion": new_sec_name, "pregunta": texto}
                    )
    return {
        "secciones": [s for s in new_secciones if s],
        "preguntas": new_preguntas,
    }


def _render_guia_editor() -> None:
    """Editor estructurado de la guía de entrevista.

    Un expander por sección; dentro, un campo de texto por pregunta y botones
    para añadir/eliminar preguntas y añadir/renombrar/eliminar secciones. Al
    guardar se valida, normaliza, persiste y sincroniza el enum de secciones.
    """
    if "guia_working" not in st.session_state:
        st.session_state.guia_working = load_guia_entrevista()
    guia = st.session_state.guia_working
    secciones = list(guia.get("secciones", []))
    preguntas = list(guia.get("preguntas", []))

    st.markdown("**📋 Guía de preguntas**")
    st.caption(
        "Edita las secciones y las preguntas que se inyectan en los prompts. "
        "Al guardar se renumera y se sincroniza el enum `seccion_entrevista` "
        "del esquema de segmentación."
    )

    for i, seccion in enumerate(secciones):
        with st.expander(f"📂 {seccion}", expanded=False):
            st.text_input("Nombre de la sección", value=seccion, key=f"guia_sec_{i}")
            # Preguntas de esta sección
            for j, p in enumerate(preguntas):
                if p.get("seccion") == seccion:
                    c1, c2 = st.columns([5, 1])
                    c1.text_input(
                        f"Pregunta {p.get('numero')}",
                        value=p.get("pregunta", ""),
                        key=f"guia_q_{i}_{j}",
                    )
                    if c2.button("🗑", key=f"guia_q_del_{i}_{j}"):
                        preguntas.remove(p)
                        st.session_state.guia_working["preguntas"] = preguntas
                        st.rerun()
            if st.button("➕ Añadir pregunta", key=f"guia_q_add_{i}"):
                preguntas.append({"numero": 0, "seccion": seccion, "pregunta": ""})
                st.session_state.guia_working["preguntas"] = preguntas
                st.rerun()
            if st.button("🗑 Eliminar sección", key=f"guia_sec_del_{i}"):
                secciones.remove(seccion)
                preguntas = [p for p in preguntas if p.get("seccion") != seccion]
                st.session_state.guia_working["secciones"] = secciones
                st.session_state.guia_working["preguntas"] = preguntas
                st.rerun()

    # ── Añadir sección ──
    nueva_seccion = st.text_input("Nueva sección", key="guia_new_sec")
    if st.button("➕ Añadir sección", key="guia_sec_add"):
        if nueva_seccion.strip():
            secciones.append(nueva_seccion.strip())
            st.session_state.guia_working["secciones"] = secciones
            st.rerun()

    # ── Guardar ──
    if st.button("💾 Guardar guía", key="guia_save"):
        data = _collect_guia_from_widgets()
        ok, err = validate_guia(data)
        if not ok:
            st.error(f"❌ {err}")
        else:
            norm = normalize_guia(data)
            save_guia_entrevista(norm)
            rep = sync_seccion_enum()
            added = len(rep.get("added", []))
            removed = len(rep.get("removed", []))
            st.success(
                f"Guía guardada en `ia/entrevistas/guia.json`. "
                f"Secciones sincronizadas: +{added} añadidas, -{removed} eliminadas."
            )
            st.session_state.guia_working = norm
            st.rerun()

    st.caption("Vista previa (lo que recibe el modelo):")
    st.code(guia_como_texto(), language=None)


def render_entrevistas_prompts() -> None:
    """Editores de configuración de entrevistas: localismos, prompts y guía."""
    _render_localismos_editor()
    _render_gramatica_editor()
    _render_prompts_editor()
    _render_guia_editor()
