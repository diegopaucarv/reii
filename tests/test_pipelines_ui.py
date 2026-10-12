"""Pruebas de la pestaña de configuración y de la vista de figuras del discurso (AppTest)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _hub_app():
    import streamlit as st

    from reii.pipelines.hub_tab import HubContext, render_pipeline_hub

    st.session_state.setdefault("calls", [])
    calls = st.session_state["calls"]
    st.session_state.setdefault("workflow_state", {"running": False, "finished": False})
    st.session_state.setdefault(
        "discourse_state", {"running": False, "finished": False}
    )

    def status():
        return {
            "batch": {"state": "al_dia", "metrics": [("Documentos", 28)]},
            "workflow": {"state": "al_dia", "metrics": [("UCEs", 5809), ("Clases", 8)]},
            "discourse": {
                "state": "desactualizado",
                "metrics": [("Anotaciones (BD)", 0)],
            },
            "checks": [
                {
                    "id": "annotations_sync",
                    "label": "Anotaciones en la BD",
                    "level": "warn",
                    "detail": "faltan",
                    "fix": "sync_annotations",
                },
                {
                    "id": "classification",
                    "label": "Clasificación",
                    "level": "ok",
                    "detail": "ok",
                    "fix": None,
                },
            ],
        }

    ctx = HubContext(
        get_status=status,
        session=st.session_state,
        get_effective_config=lambda: {
            "use_rf_shap": (False, "ultima_corrida"),
            "min_uce_words": (3, "override"),
            "tsj": (3, "ultima_corrida"),
            "pseudocount": (0.1, "ultima_corrida"),
        },
        config_categories={
            "rf_shap": ["use_rf_shap"],
            "frecuencias": ["tsj", "pseudocount"],
        },
        save_config=lambda u: calls.append(("save", dict(u))) or {"pushed": list(u)},
        reset_config=lambda: calls.append(("reset",)),
        start_workflow=lambda p: calls.append(("workflow", dict(p))),
        list_workflows=lambda: [
            {
                "file": "1.json",
                "name": "Figuras",
                "agents": ["metafora", "anclaje"],
                "progress": 0.5,
            }
        ],
        rename_workflow=lambda f, n: calls.append(("rename", f, n)),
        start_discourse=lambda *a: calls.append(("discourse", a)),
        run_fix=lambda fid: calls.append(("fix", fid)) or "hecho",
        render_agents_crud=lambda: None,
    )
    render_pipeline_hub(ctx)


def test_hub_renders_map_checks_and_two_steps():
    at = AppTest.from_function(_hub_app, default_timeout=30).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert (
        "Transcripciones" in text
        and "Clasificación" in text
        and "Análisis del discurso" in text
    )
    labels = [t.label for t in at.tabs]
    for want in ("2 · Clasificación", "3 · Análisis del discurso (IA)"):
        assert want in labels, labels
    assert not any("1 · Transcripciones" in l for l in labels), labels


def test_fix_button_calls_the_requested_action():
    at = AppTest.from_function(_hub_app, default_timeout=30).run()
    at.button(key="hub_fix_annotations_sync_sync_annotations").click().run()
    assert ("fix", "sync_annotations") in at.session_state["calls"]


def test_config_edit_then_save_and_run_pass_the_changed_values():
    at = AppTest.from_function(_hub_app, default_timeout=30).run()
    at.radio(key="hub_ess_use_rf_shap").set_value(True).run()
    assert at.button(key="hub_cfg_save").disabled is False
    at.button(key="hub_cfg_save").click().run()
    assert ("save", {"use_rf_shap": True}) in at.session_state["calls"]
    at.radio(key="hub_wf_mode").set_value("transformer").run()
    at.button(key="hub_run_workflow").click().run()
    assert (
        "workflow",
        {"workflow_mode": "transformer", "use_rf_shap": True},
    ) in at.session_state["calls"]


def test_save_is_disabled_without_changes_and_discourse_run_uses_db_source():
    at = AppTest.from_function(_hub_app, default_timeout=30).run()
    assert at.button(key="hub_cfg_save").disabled is True
    at.button(key="hub_wfrun_1.json").click().run()
    kind, args = [c for c in at.session_state["calls"] if c[0] == "discourse"][0]
    assert args == ("1.json", True, None, False, "db")


def test_run_buttons_are_disabled_while_running():
    at = AppTest.from_function(_hub_app, default_timeout=30)
    at.session_state["workflow_state"] = {"running": True}
    at.session_state["discourse_state"] = {"running": True}
    at.run()
    assert not at.exception, at.exception
    assert (
        at.button(key="hub_run_workflow").disabled
        and at.button(key="hub_wfrun_1.json").disabled
    )


def _results_app(empty=False):
    from reii.pipelines.discourse_results import render_discourse_results

    uces = [
        {
            "uce_id": "u1",
            "texto": "El agua es vida y la vida es agua.",
            "cluster_id": 0,
            "doc_id": 1,
        },
        {
            "uce_id": "u2",
            "texto": "Nada cambia en el pueblo.",
            "cluster_id": 1,
            "doc_id": 2,
        },
    ]
    ann = (
        {}
        if empty
        else {
            "u1": [
                {
                    "agent": "A",
                    "trait": "metáfora",
                    "confidence": "alta",
                    "spans": [
                        {"quote": "El agua es vida", "start_char": 0, "end_char": 15}
                    ],
                }
            ],
            "u2": [
                {
                    "agent": "B",
                    "trait": "anclaje",
                    "confidence": "baja",
                    "spans": [{"quote": "Nada cambia"}],
                }
            ],
        }
    )
    render_discourse_results(ann, uces, class_id=0, class_color="#7C6BF8")


def _results_empty_app():
    from reii.pipelines.discourse_results import render_discourse_results

    render_discourse_results(
        {}, [{"uce_id": "u1", "texto": "x", "cluster_id": 0, "doc_id": 1}], class_id=0
    )


def test_results_show_only_the_selected_class_by_default():
    at = AppTest.from_function(_results_app, default_timeout=30).run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Figuras (citas)"] == "1" and metrics["UCEs anotadas"] == "1"
    html = " ".join(m.value for m in at.markdown)
    assert "metáfora" in html and "anclaje" not in html and "<mark" in html


def test_results_all_classes_toggle_and_filters():
    at = AppTest.from_function(_results_app, default_timeout=30).run()
    at.toggle(key="dr_all").set_value(True).run()
    assert {m.label: m.value for m in at.metric}["Figuras (citas)"] == "2"
    at.multiselect(key="dr_conf").unselect("baja").run()
    assert {m.label: m.value for m in at.metric}[
        "Figuras (citas)"
    ] == "2"  # métricas del alcance, filtros afectan la lista
    assert "anclaje" not in " ".join(m.value for m in at.markdown)


def test_results_empty_state_points_to_last_tab():
    at = AppTest.from_function(_results_empty_app, default_timeout=30).run()
    assert not at.exception, at.exception
    assert any("última pestaña" in i.value for i in at.info)


def _results_wf_app():
    from reii.pipelines.discourse_results import render_discourse_results

    uces = [
        {"uce_id": "u1", "texto": "El agua es vida.", "cluster_id": 0, "doc_id": 1},
        {"uce_id": "u2", "texto": "Nada cambia hoy.", "cluster_id": 0, "doc_id": 1},
        {"uce_id": "u3", "texto": "Otro texto suelto.", "cluster_id": 0, "doc_id": 2},
    ]
    ann = {
        "u1": [
            {
                "agent": "metafora",
                "trait": "metáfora",
                "confidence": "alta",
                "spans": [{"quote": "El agua"}],
            }
        ],
        "u2": [
            {
                "agent": "anclaje",
                "trait": "anclaje",
                "confidence": "alta",
                "spans": [{"quote": "Nada"}],
            }
        ],
        "u3": [
            {
                "agent": "viejo",
                "trait": "x",
                "confidence": "media",
                "spans": [{"quote": "Otro"}],
            }
        ],
    }
    wfs = [
        {"file": "1.json", "name": "Figuras", "agents": ["metafora"]},
        {"file": "2.json", "name": "Marcos", "agents": ["anclaje"]},
    ]
    render_discourse_results(ann, uces, class_id=0, workflows=wfs)


def test_workflow_selector_switches_between_discourse_pipelines():
    at = AppTest.from_function(_results_wf_app, default_timeout=30).run()
    assert not at.exception, at.exception
    sel = at.selectbox(key="dr_wf")
    assert sel.options[0].startswith("Todos los pipelines · 3 citas")
    assert any(o.startswith("Figuras · 1 citas") for o in sel.options)
    assert any(
        o.startswith("Sin pipeline") for o in sel.options
    )  # agente que ya no está en ningún JSON
    assert {m.label: m.value for m in at.metric}["Figuras (citas)"] == "3"
    sel.select("1.json").run()
    assert {m.label: m.value for m in at.metric}["Figuras (citas)"] == "1"
    html = " ".join(m.value for m in at.markdown)
    assert "metáfora" in html and "anclaje" not in html
    at.selectbox(key="dr_wf").select("2.json").run()
    assert "anclaje" in " ".join(m.value for m in at.markdown)


def test_agent_filter_is_limited_to_the_selected_pipeline():
    at = AppTest.from_function(_results_wf_app, default_timeout=30).run()
    at.selectbox(key="dr_wf").select("1.json").run()
    assert list(at.multiselect(key="dr_agents_1.json").options) == ["metafora"]
