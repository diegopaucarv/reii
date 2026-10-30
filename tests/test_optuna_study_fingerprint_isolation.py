"""_normalize_storage es la garantía de que reii_optuna.db puede acumular,
sin mezclarse, un estudio Optuna por cada corpus distinto que se haya
corrido alguna vez (confirmado: study_name = f"{name}{suffix}_{fingerprint}").
Eso es lo que hace viable minar el histórico multi-corpus para calibrar
rangos de búsqueda a futuro (atlas de parámetros) — si esta aislación se
rompiera silenciosamente, trials de corpus distintos terminarían
compitiendo en el mismo frente de Pareto sin que nada lo avisara.

Se extrae la función real vía AST; se stubea _corpus_fingerprint (no nos
interesa CÓMO se calcula el fingerprint acá, solo que su valor de retorno
SIEMPRE termine en el study_name).
"""
import os
from typing import Optional

import pytest

from _ast_extract import extract_function


class _Cfg:
    db_local_path = "data/workflow_data.json"


class _StubSelf:
    config_base = _Cfg()


def _make_normalize_storage(fingerprint_values):
    """fingerprint_values: lista de valores que _corpus_fingerprint() va
    devolviendo en llamadas sucesivas (uno por cada llamada a normalize)."""
    calls = iter(fingerprint_values)

    def fake_fingerprint():
        return next(calls)

    namespace = {"os": os, "Optional": Optional, "_corpus_fingerprint": fake_fingerprint}
    return extract_function("_normalize_storage", namespace)


def test_falsy_storage_returns_memory_study_unchanged():
    normalize = _make_normalize_storage([])
    storage, study_name = normalize(_StubSelf(), None, "reii_search")
    assert storage is None
    assert study_name == "reii_search"  # sin fingerprint — no hay archivo que aislar


def test_relative_sqlite_path_remapped_and_fingerprint_appended():
    normalize = _make_normalize_storage(["fpA"])
    storage, study_name = normalize(
        _StubSelf(), "sqlite:///reii_optuna.db", "reii_search"
    )
    assert storage == "sqlite:///data/reii_optuna.db"
    assert study_name == "reii_search_fpA"


def test_two_different_corpora_never_share_a_study_name():
    """La propiedad central: mismo study_name base + mismo storage, pero
    dos corpus (dos fingerprints) distintos → dos study_name distintos.
    Si esto fallara, un estudio nuevo sobre un corpus nuevo podría
    cargar/mezclar trials de un corpus viejo (load_if_exists=True)."""
    normalize = _make_normalize_storage(["corpus_A_fp", "corpus_B_fp"])
    _, study_name_a = normalize(_StubSelf(), "sqlite:///reii_optuna.db", "reii_search")
    _, study_name_b = normalize(_StubSelf(), "sqlite:///reii_optuna.db", "reii_search")
    assert study_name_a != study_name_b
    assert study_name_a.endswith("corpus_A_fp")
    assert study_name_b.endswith("corpus_B_fp")


def test_same_corpus_rerun_reuses_the_same_study_name():
    """Caso espejo del anterior: mismo fingerprint dos veces (mismo
    corpus, dos corridas) → mismo study_name, para que load_if_exists=True
    sí reutilice/acumule trials entre corridas del MISMO corpus."""
    normalize = _make_normalize_storage(["same_fp", "same_fp"])
    _, study_name_1 = normalize(_StubSelf(), "sqlite:///reii_optuna.db", "reii_search")
    _, study_name_2 = normalize(_StubSelf(), "sqlite:///reii_optuna.db", "reii_search")
    assert study_name_1 == study_name_2


def test_absolute_sqlite_path_is_left_as_is():
    normalize = _make_normalize_storage(["fpZ"])
    storage, _ = normalize(_StubSelf(), "sqlite:////already/absolute/path.db", "s")
    assert storage == "sqlite:////already/absolute/path.db"


@pytest.mark.parametrize("stage_suffix", ["_wc_only", "_all", ""])
def test_fingerprint_appends_after_whatever_stage_suffix_the_caller_used(stage_suffix):
    # optimizar() arma study_name como f"{base}{stage_suffix}" ANTES de
    # llamar a _normalize_storage — confirma que el fingerprint se agrega
    # al final de eso, no que lo reemplaza ni se inserta en medio.
    normalize = _make_normalize_storage(["fp1"])
    _, study_name = normalize(
        _StubSelf(), "sqlite:///reii_optuna.db", f"reii_search{stage_suffix}"
    )
    assert study_name == f"reii_search{stage_suffix}_fp1"
