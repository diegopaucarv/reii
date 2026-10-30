"""Tests de `reii.diag term-metrics`: Yates, phi sin corregir, FDR y MCA.

No tocan datos reales: construyen la tabla 2x2 desde recuentos conocidos y un
artefacto minimo, y comprueban las identidades estadisticas, la verificacion de
coherencia y el contrato del comando.
"""

from __future__ import annotations

import json

import pytest

from reii.diag import term_metrics as tm

# -- estadistica -----------------------------------------------------------


def test_phi_sin_yates_matches_definition():
    a, b, c, d = 2, 2, 1, 5
    assert abs(tm.phi_signed(a, b, c, d) - 0.356348) < 1e-5


def test_chi2_yates_value():
    a, b, c, d = 2, 2, 1, 5
    assert abs(tm.chi2_yates(a, b, c, d) - 0.178571) < 1e-5


def test_phi_with_yates_value():
    a, b, c, d = 2, 2, 1, 5
    assert abs(tm.phi_yates_signed(a, b, c, d) - 0.133631) < 1e-5


def test_yates_zeroes_weak_association():
    a, b, c, d = 1, 2, 3, 4
    assert abs(tm.phi_signed(a, b, c, d) + 0.089087) < 1e-5
    assert tm.chi2_yates(a, b, c, d) == 0.0
    assert tm.phi_yates_signed(a, b, c, d) == 0.0


def test_phi_keeps_sign():
    # ad - bc > 0  ->  asociacion positiva
    assert tm.phi_signed(5, 1, 2, 4) > 0
    # ad - bc < 0  ->  asociacion negativa (el termino evita la clase)
    assert tm.phi_signed(1, 5, 4, 2) < 0
    # misma tabla, phi yates nunca cambia el signo, solo lo encoge
    assert tm.phi_yates_signed(1, 5, 4, 2) < 0
    assert abs(tm.phi_yates_signed(1, 5, 4, 2)) < abs(tm.phi_signed(1, 5, 4, 2))


def test_class_sizes_skips_unassigned():
    assert tm.class_sizes([0, 0, 1, -1, 2, 2, 2]) == {0: 2, 1: 1, 2: 3}


# -- artefacto de prueba ---------------------------------------------------

_LABELS = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
# (termino, cluster, a, b, c, d)
_CELLS = [("a", 0, 2, 2, 1, 5), ("b", 2, 1, 2, 3, 4)]


def _doc(overrides=None):
    overrides = overrides or {}
    terms = []
    for name, cluster, a, b, c, d in _CELLS:
        row = {
            "termino": name,
            "cluster": cluster,
            "frecuencia_global": a + b,
            "frecuencia_cluster": a,
            "chi2_yates": tm.chi2_yates(a, b, c, d),
            "phi": tm.phi_yates_signed(a, b, c, d),
            "p_valor": 0.01,
            "p_adj": 0.20,
            "significativo": True,
        }
        row.update(overrides.get(name, {}))
        terms.append(row)
    return {
        "terminos": terms,
        "multivariate": {
            "cluster_labels": _LABELS,
            "terms": ["a", "b"],
            "col_coords": [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]],
            "eigenvalues": [0.4, 0.2],
            "explained_inertia": [0.6, 0.4],
        },
    }


def test_derive_terms_validates_and_flags():
    rows, meta = tm.derive_terms(_doc(), 0.05)
    assert meta["n_uces_con_clase"] == 10
    assert meta["class_sizes"] == {0: 3, 1: 3, 2: 4}
    assert meta["max_abs_diff_chi2_yates"] == 0
    assert meta["max_abs_diff_phi"] == 0
    assert len(rows) == 2

    first = rows[0]
    assert abs(first["phi_sin_yates"] - 0.356348) < 1e-4
    assert abs(first["phi"] - 0.133631) < 1e-4
    assert first["significativo"] is True
    assert first["significativo_fdr"] is False


def test_derive_terms_skips_impossible_cells():
    doc = _doc({"a": {"frecuencia_cluster": 99}})
    rows, meta = tm.derive_terms(doc, 0.05)
    assert meta["n_rows_skipped"] == 1
    assert len(rows) == 1


def test_mca_inertia_over_total():
    info = tm.mca_meta(_doc())
    assert info["total_inertia"] == 1.0
    assert info["explained_inertia_corrected"] == [0.4, 0.2]
    assert info["explained_inertia_stored"] == [0.6, 0.4]


# -- CLI -------------------------------------------------------------------


def test_cli_writes_json(tmp_path):
    art = tmp_path / "wf.json"
    art.write_text(json.dumps(_doc()), encoding="utf-8")
    out = tmp_path / "out.json"
    rc = tm.main(["--artifact", str(art), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_uces_con_clase"] == 10
    assert len(payload["terms"]) == 2


def test_cli_rejects_incoherent_artifact(tmp_path):
    art = tmp_path / "wf.json"
    art.write_text(json.dumps(_doc({"a": {"chi2_yates": 999.0}})), encoding="utf-8")
    out = tmp_path / "out.json"
    rc = tm.main(["--artifact", str(art), "--out", str(out)])
    assert rc == 1
    assert not out.exists()
