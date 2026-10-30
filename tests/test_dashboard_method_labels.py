"""Tests del selector de particion por metodo de `reii.dashboard`.

`dashboard.py` es un script de Streamlit: al importarlo ejecuta la UI completa y
carga el snapshot con `DashboardAdapter` (Postgres). En este entorno no hay base
de datos accesible, asi que se sustituyen `version_hash`/`snapshot` por un
snapshot minimo en memoria JUSTO antes del import: sin red y sin datos reales.
Si `streamlit` no estuviera instalado, el modulo no se podria importar y los
tests se saltan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("streamlit", reason="reii.dashboard importa streamlit")


def _uce(idx: int, cluster_id: int, doc_id: str = "0") -> dict:
    """UCE minima con los campos que consume el dashboard al importarse."""
    uce_id = f"{doc_id}_{idx}_0"
    return {
        "id": uce_id,
        "uce_id": uce_id,
        "doc_id": doc_id,
        "local_idx": idx,
        "seccion": "a",
        "texto": f"texto de prueba {idx}",
        "cluster_id": cluster_id,
        "is_stable": True,
        "phi_score": 0.5,
        "projected_cluster_id": None,
        "projection_distance": None,
        "projection_margin": None,
        "projection_ratio": None,
        "metadata": {"uce_local_idx": idx, "section_id": "a"},
        "lemmas": ["prueba"],
        "verbos": [],
        "negaciones": [],
        "pronombres": [],
        "marcadores_discursivos": [],
        "coref_chains": [],
        "registro": "formal",
        "metricas_lexicas": {"num_tokens": 3, "mean_zipf": 4.0},
    }


def _import_dashboard():
    """Importa `reii.dashboard` con el adaptador parcheado (sin Postgres)."""
    from unittest.mock import patch

    import reii.backend.dashboard_adapter as adapter_mod

    uces = [_uce(idx, 0 if idx < 4 else 1) for idx in range(8)]
    snapshot = {
        "uces": uces,
        "terminos": [],
        "ucs": [],
        "clusters": [],
        "config": {},
        "doc_metadata": {"0": {}},
        "uce_phi": [{"uce_id": u["uce_id"], "phi_score": u["phi_score"]} for u in uces],
    }
    with (
        patch.object(adapter_mod.DashboardAdapter, "version_hash", lambda self: "test"),
        patch.object(
            adapter_mod.DashboardAdapter, "snapshot", lambda self, *a, **kw: snapshot
        ),
    ):
        import importlib

        return importlib.import_module("reii.dashboard")


dashboard = _import_dashboard()


def _data(**overrides) -> dict:
    """Snapshot de juguete: ganador persistido = cluster 0/1 de `uces`."""
    data = {
        "uces": [_uce(0, -1), _uce(1, 1), _uce(2, 0)],
        "ucs": [{"id": "uc0", "cluster_label_double": 0}],
        "terminos": [{"termino": "prueba", "cluster": 0, "phi": 1.0}],
        "method_selection": {"winner": "A"},
    }
    data.update(overrides)
    return data


def test_options_sin_method_labels():
    assert dashboard.method_labels_options(_data()) == (
        ["__stored__"],
        "__stored__",
    )


def test_options_con_etiquetas_vacias():
    data = _data(method_labels={"A": {}, "B": {}, "C": {}})
    assert dashboard.method_labels_options(data) == (
        ["__stored__"],
        "__stored__",
    )


def test_options_default_es_el_ganador_con_etiquetas():
    data = _data(
        method_labels={"A": {"0_0_0": 2}, "C": {"0_1_0": 5}},
        method_selection={"winner": "C"},
    )
    options, default = dashboard.method_labels_options(data)
    assert options == ["__stored__", "A", "C"]
    assert default == "C"


def test_options_default_stored_si_el_ganador_no_tiene_etiquetas():
    data = _data(
        method_labels={"A": {"0_0_0": 2}, "B": {}},
        method_selection={"winner": "B"},
    )
    options, default = dashboard.method_labels_options(data)
    assert options == ["__stored__", "A"]
    assert default == "__stored__"


def test_apply_stored_es_identidad():
    data = _data(method_labels={"A": {"0_0_0": 2}})
    assert dashboard.apply_method_labels(data, "__stored__") is data


def test_apply_metodo_sin_etiquetas_es_identidad():
    data = _data(method_labels={"A": {"0_0_0": 2}})
    assert dashboard.apply_method_labels(data, "C") is data


def test_apply_remepea_sin_mutar_el_snapshot():
    data = _data(method_labels={"A": {"0_0_0": 2}})
    original_uces = data["uces"]
    original_first = data["uces"][0]

    view = dashboard.apply_method_labels(data, "A")

    # Lista y dicts nuevos; la UCE de A que no estaba en el ganador se clasifica.
    assert view is not data
    assert view["uces"] is not original_uces
    assert view["uces"][0] is not original_first
    assert view["uces"][0]["cluster_id"] == 2

    # Las UCEs ausentes de la particion A quedan sin clasificar.
    assert view["uces"][1]["cluster_id"] == -1
    assert view["uces"][2]["cluster_id"] == -1

    # El snapshot original no se toca.
    assert data["uces"] is original_uces
    assert data["uces"][0]["cluster_id"] == -1
    assert data["uces"][1]["cluster_id"] == 1
    assert data["uces"][2]["cluster_id"] == 0

    # El resto del snapshot se comparte tal cual (particion del ganador).
    assert view["terminos"] is data["terminos"]
    assert view["ucs"] is data["ucs"]
    assert view["method_selection"] is data["method_selection"]


def test_summary_de_la_particion_elegida():
    data = _data(method_labels={"A": {"0_0_0": 2, "0_2_0": 2}})
    summary = dashboard.method_labels_summary(data, "A")
    assert summary["method"] == "A"
    assert summary["n_total"] == 3
    assert summary["n_classified"] == 2
    assert summary["k"] == 1
    assert summary["sizes"] == {2: 2}
    assert summary["coverage"] == pytest.approx(2 / 3)


def test_summary_stored_usa_la_particion_persistida():
    summary = dashboard.method_labels_summary(_data(), "__stored__")
    assert summary["n_classified"] == 2
    assert summary["k"] == 2
    assert summary["sizes"] == {0: 1, 1: 1}
