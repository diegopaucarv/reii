"""_is_sentinel: el valor centinela real es (-2.0, -1.0, 0.0), no -999.0
(error que tenía el script 3_diagnose_optuna_db.py original).
_bottleneck: decide cuál de los 3 métodos (WC/COREF/EMB) es el cuello de
botella de un trial, promediando los ARI cruzados entre pares — hoy se
calcula y se guarda como user_attr en cada trial pero nadie lo agrega ni lo
reporta (ver conversación: comparative_results.json ya lo tiene).

Ambas funciones viven en main_workflow.py; se extraen vía AST para no
necesitar el stack pesado de NLP solo para probar dos funciones puras.
"""
import numpy as np
import pytest

from _ast_extract import extract_function


@pytest.fixture(scope="module")
def is_sentinel():
    # _is_sentinel referencia _SENTINEL_TRIAL como variable libre (no es
    # un parámetro) — hay que inyectarla en el namespace.
    return extract_function("_is_sentinel", {"_SENTINEL_TRIAL": (-2.0, -1.0, 0.0)})


@pytest.fixture(scope="module")
def bottleneck():
    return extract_function("_bottleneck", {"np": np})


# ── _is_sentinel ────────────────────────────────────────────────────────

def test_sentinel_exact_tuple_is_detected(is_sentinel):
    assert is_sentinel((-2.0, -1.0, 0.0)) is True


def test_sentinel_accepts_list_too(is_sentinel):
    # save_pareto_front compara t.values (puede venir como list de Optuna)
    assert is_sentinel([-2.0, -1.0, 0.0]) is True


@pytest.mark.parametrize(
    "values",
    [
        (0.5, 0.5, 2.0),
        (-999.0, -999.0, -999.0),  # el valor que asumía (mal) el script 3 original
        (-2.0, -1.0, 0.1),  # casi el centinela, no exacto
        (),
    ],
)
def test_non_sentinel_values_are_not_flagged(is_sentinel, values):
    assert is_sentinel(values) is False


# ── _bottleneck ─────────────────────────────────────────────────────────
# _bottleneck promedia los ARI cruzados en pares que SÍ comparten cada
# método: wc→(ab,ac), coref→(ab,bc), emb→(bc,ac). El método con el promedio
# más bajo es el reportado como cuello de botella.

def test_low_ari_ab_and_ac_flags_wc_as_bottleneck(bottleneck):
    # wc aparece en ab y ac — si ambos son bajos, wc debe ser el cuello.
    assert bottleneck(ari_ab=0.1, ari_bc=0.9, ari_ac=0.1) == "wc"


def test_low_ari_ab_and_bc_flags_coref_as_bottleneck(bottleneck):
    assert bottleneck(ari_ab=0.1, ari_bc=0.1, ari_ac=0.9) == "coref"


def test_low_ari_bc_and_ac_flags_emb_as_bottleneck(bottleneck):
    assert bottleneck(ari_ab=0.9, ari_bc=0.1, ari_ac=0.1) == "emb"


def test_all_equal_ari_returns_one_of_the_three_methods(bottleneck):
    # Empate exacto: min() de Python devuelve el primero en orden de
    # inserción del dict ("wc"). Se documenta el desempate, no se asume
    # que "no importa cuál".
    assert bottleneck(ari_ab=0.5, ari_bc=0.5, ari_ac=0.5) == "wc"
