"""Bootstrap real de estabilidad: `_stability_spaces` (patch 20).

El bug que cubre: ``run()`` pasaba ``getattr(self, "_stability_spaces", None)``
a ``_stability_blocks``, pero NADA asignaba ese atributo. El espacio llegaba
siempre ``None``, todos los ``bootstrap_ari`` caian a ``nan`` y
``MethodMetrics.stability_kind`` caia siempre a ``"perturbation"``: el
"criterio principal: bootstrap (remuestreo real)" que documenta
``reii/method_selection.py`` nunca estuvo operativo.

Contrato que se fija aqui:

  (a) flag apagado -> no se construye espacio y el tipo sigue "perturbation".
  (b) flag encendido + espacio valido -> ``bootstrap_ari`` finito y
      ``stability_kind == "bootstrap"``.
  (c) espacio malformado -> ``{}`` para ese metodo, sin lanzar.
  (d) un metodo malo no impide evaluar al resto.
  (e) ``_build_stability_spaces`` degrada POR METODO, no propaga excepciones.

No se instancia el pipeline: ``DoubleClassifier.__new__`` evita
Segmentador/UCBuilder (Stanza). El import del modulo cuesta lo mismo que en
``test_cdh_min_leaf_floor.py`` (spacy/prince/transformers).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from reii.main_workflow import (
    _CLI_FLAGS_SIN_VALOR,
    UC,
    Config,
    DoubleClassifier,
    _cli_overrides,
)
from reii.method_selection import MethodMetrics


# ── Helpers ────────────────────────────────────────────────────────────
def _dc(**cfg) -> DoubleClassifier:
    """``DoubleClassifier`` sin ``__init__`` (no necesita Stanza)."""
    dc = DoubleClassifier.__new__(DoubleClassifier)
    dc.config = Config(**cfg)
    dc.ppmi_builder = None
    dc.quiet_selection = True
    return dc


def _blobs(n: int = 120, d: int = 4, seed: int = 0):
    """Tres nubes separadas en la 1a coordenada + particion de referencia."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.3, size=(n, d))
    third = n // 3
    X[:third, 0] += 6.0
    X[2 * third :, 0] -= 6.0
    y = np.repeat([0, 1, 2], [third, third, n - 2 * third])
    return X, y


def _predict_thirds(X_sub):
    """Refit trivial y determinista: reasigna por la 1a coordenada."""
    v = np.asarray(X_sub, dtype=float)[:, 0]
    return np.where(v > 3.0, 0, np.where(v < -3.0, 2, 1))


def _spec(X, y):
    ids = [f"u{i}" for i in range(len(y))]
    return ids, {u: int(c) for u, c in zip(ids, y)}


# ── (a) flag apagado: nada cambia ──────────────────────────────────────
def test_flag_off_builds_no_space_and_keeps_perturbation():
    assert Config().stability_bootstrap is True  # default desde v6
    dc = _dc(stability_bootstrap=False)  # opt-out explicito

    # Corta ANTES de validar argumentos: no toca ucs/voc/labels.
    assert dc._build_stability_spaces(None, None, None) is None
    assert dc._build_stability_spaces({"ucs": []}, {}, {}) is None

    # Asi queda en run(): el atributo vale None y el bloque no existe.
    dc._stability_spaces = dc._build_stability_spaces(None, None, None)
    assert dc._stability_spaces is None
    assert dc._stability_blocks(dc._stability_spaces) == {}

    metric = MethodMetrics(method="A", k=3, ari=0.42, coverage=0.4)
    assert metric.stability_kind == "perturbation"
    assert metric.stability == pytest.approx(0.42)


# ── (b) espacio valido: bootstrap real, finito ─────────────────────────
def test_valid_space_yields_finite_bootstrap_ari():
    dc = _dc()
    X, y = _blobs()
    ids, cmap = _spec(X, y)
    spaces = {"A": (X, ids, cmap, _predict_thirds)}

    out = dc._stability_blocks(spaces, n_boot=10, n_perm=10, min_n=30)
    block = out["A"]
    assert block, "un espacio valido no puede producir un bloque vacio"

    value = float(block["bootstrap_ari"])
    assert np.isfinite(value)
    assert value > 0.9, "las tres nubes deben reproducirse en la replica"
    assert block["stability_kind"] == "bootstrap"
    assert np.isfinite(float(block["bootstrap_ari_ci_low"]))
    assert np.isfinite(float(block["bootstrap_ari_ci_high"]))

    metric = MethodMetrics(
        method="A",
        k=3,
        ari=0.5,
        coverage=0.5,
        bootstrap_ari=value,
    )
    assert metric.stability_kind == "bootstrap"
    assert metric.stability == pytest.approx(value)


# ── (c) espacio malformado: {} sin lanzar ──────────────────────────────
def test_malformed_spaces_are_dropped_without_raising():
    dc = _dc()
    X, y = _blobs()
    ids, cmap = _spec(X, y)

    # (i) X con shape[0] != len(ids)
    wrong_rows = (np.zeros((3, X.shape[1])), ids, cmap, _predict_thirds)
    assert dc._stability_blocks({"A": wrong_rows}) == {"A": {}}

    # (ii) predict_fn no callable
    assert dc._stability_blocks({"A": (X, ids, cmap, None)}) == {"A": {}}

    # (iii) nada asignado (< min_n)
    unassigned = {u: -1 for u in ids}
    spec = (X, ids, unassigned, _predict_thirds)
    assert dc._stability_blocks({"A": spec}, min_n=30) == {"A": {}}


# ── (d) un metodo malo no bloquea al resto ─────────────────────────────
def test_one_bad_space_does_not_block_the_rest():
    dc = _dc()
    X, y = _blobs()
    ids, cmap = _spec(X, y)
    bad = (np.zeros((1, X.shape[1])), ids, cmap, _predict_thirds)
    good = (X, ids, cmap, _predict_thirds)

    out = dc._stability_blocks({"A": bad, "B": good}, n_boot=5, n_perm=5, min_n=30)
    assert set(out) == {"A", "B"}
    assert out["A"] == {}
    assert np.isfinite(float(out["B"]["bootstrap_ari"]))


# ── (e) construccion: degrada POR METODO, nunca lanza ──────────────────
def _uc_corpus(n: int = 40):
    """UCs sinteticas: dos bloques lexicos + un termino comun."""
    ucs, labels = [], []
    for i in range(n):
        head = "a" if i < n // 2 else "b"
        terms = [f"{head}1", f"{head}2", "comun"]
        ucs.append(
            UC(
                texto=" ".join(terms),
                lemmas=list(terms),
                stems=list(terms),
                uce_ids=[f"uce{i}"],
            )
        )
        labels.append(0 if head == "a" else 1)
    return {
        "ucs": ucs,
        "labels": np.asarray(labels, dtype=int),
        "voc": sorted({"a1", "a2", "b1", "b2", "comun"}),
    }


def test_build_spaces_off_by_default_and_degrades_per_method():
    # flag off (opt-out): ni siquiera se mira el resultado
    assert (
        _dc(stability_bootstrap=False)._build_stability_spaces(_uc_corpus(), None, None)
        is None
    )

    dc = _dc(stability_bootstrap=True)
    spaces = dc._build_stability_spaces(_uc_corpus(), None, None, min_n=30)

    # A se construye; B y C no tienen resultado -> se omiten con motivo.
    assert set(spaces) == {"A"}
    X, ids, cmap, predict = spaces["A"]
    assert X.shape[0] == len(ids) == 40
    assert set(cmap) == set(ids)
    assert callable(predict)
    assert len(predict(X[:32])) == 32

    # El motor completo tampoco lanza sobre un espacio degenerado.
    out = dc._stability_blocks(spaces, n_boot=3, n_perm=3, min_n=30)
    assert "A" in out


# ── Method C: solo las UCEs con embedding ──────────────────────────────
def _emb_result(n_all: int = 60, n_emb: int = 54, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_all, 4))
    y = np.repeat([0, 1, 2], [20, 20, 20])
    mask = np.zeros(n_all, dtype=bool)
    mask[:n_emb] = True
    return {
        "uce_ids": [f"e{i}" for i in range(n_all)],
        "labels_array": y,
        "embedding_matrix": X,
        "embeddable_mask": mask,
        "backend": "kmeans_spherical",
        "target_k": 3,
        "min_cluster_size": 5,
    }


def test_method_c_space_uses_embeddable_subset():
    dc = _dc(stability_bootstrap=True)
    spaces = dc._build_stability_spaces(None, None, _emb_result(), min_n=30)

    assert set(spaces) == {"C"}
    X, ids, cmap, predict = spaces["C"]
    assert X.shape[0] == len(ids) == 54
    assert len(cmap) == 54
    assert ids == [f"e{i}" for i in range(54)]
    assert callable(predict)


def test_method_c_omitted_when_too_few_embeddable():
    dc = _dc(stability_bootstrap=True)
    res = _emb_result()
    res["embeddable_mask"] = np.array([True] * 10 + [False] * 50)
    assert "C" not in dc._build_stability_spaces(None, None, res, min_n=30)


def test_method_c_omitted_when_mask_missing():
    dc = _dc(stability_bootstrap=True)
    res = _emb_result()
    res.pop("embeddable_mask")
    assert "C" not in dc._build_stability_spaces(None, None, res, min_n=30)


# ── CLI: opt-in declarado y aplicado ───────────────────────────────────
def test_cli_flag_declared_and_applied(monkeypatch):
    # Declarado: si no, `_warn_unknown_cli_flags` lo reporta como ignorado.
    assert "--stability-bootstrap" in _CLI_FLAGS_SIN_VALOR

    monkeypatch.setattr(sys, "argv", ["reii", "--stability-bootstrap"])
    assert _cli_overrides(Config()).stability_bootstrap is True

    # default True desde v6: sin flag sigue True
    monkeypatch.setattr(sys, "argv", ["reii"])
    assert _cli_overrides(Config()).stability_bootstrap is True


def test_optimize_only_forces_bootstrap_off(monkeypatch):
    # Con default True, --optimize-only debe forzar False: el optimizador no
    # puede correr bootstrap real en cada trial de etapa 2.
    monkeypatch.setattr("reii.main_workflow._OPTIMIZE_ONLY", True)
    monkeypatch.setattr(sys, "argv", ["reii", "--optimize-only"])
    assert _cli_overrides(Config()).stability_bootstrap is False


def test_no_bootstrap_flag_declared_and_applied(monkeypatch):
    # Declarado: si no, `_warn_unknown_cli_flags` lo reporta como ignorado.
    assert "--no-stability-bootstrap" in _CLI_FLAGS_SIN_VALOR

    # Opt-out explicito del default True: recupera el informe por perturbacion.
    monkeypatch.setattr(sys, "argv", ["reii", "--no-stability-bootstrap"])
    assert _cli_overrides(Config()).stability_bootstrap is False


def test_no_stability_bootstrap_conflicts_with_stability_bootstrap(monkeypatch):
    # Los dos flags se contradicen: abortar (exit 2) en vez de que gane el
    # ultimo de argv, que dejaria la decision a merced del orden de los flags.
    monkeypatch.setattr(
        sys, "argv", ["reii", "--stability-bootstrap", "--no-stability-bootstrap"]
    )
    with pytest.raises(SystemExit) as exc:
        _cli_overrides(Config())
    assert exc.value.code == 2
