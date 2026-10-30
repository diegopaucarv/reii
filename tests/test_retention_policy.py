"""Tests de la política de retención de formas activas.

NO importa ``reii.main_workflow``: el knob, la decisión por token y la tabla de
tokens se testean como funciones puras / escritor de fichero.
"""

from __future__ import annotations

import pytest
from reii.retention_policy import (
    DEFAULT_RETENTION_POLICY,
    RETENTION_POLICIES,
    TOOL_ADVERB_LEMMAS,
    TokenTableWriter,
    policy_content_pos,
    policy_filters_tool_adverbs,
    policy_uses_is_stop,
    read_token_table,
    resolve_retention_policy,
    token_decision,
    token_rows,
)


class _FakeToken:
    """Misma interfaz que un token de spaCy (sólo lo que usa ``token_rows``)."""

    def __init__(self, text, lemma, pos, is_stop, is_alpha):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.is_stop = is_stop
        self.is_alpha = is_alpha


def test_knob_maps_to_three_policies():
    assert set(RETENTION_POLICIES) == {
        "pos_only",
        "pos_plus_propn",
        "tool_adverb_filtered",
    }

    # pos_only: is_stop manda y PROPN/X no son activos.
    assert policy_uses_is_stop("pos_only")
    assert "PROPN" not in policy_content_pos("pos_only")
    assert not policy_filters_tool_adverbs("pos_only")

    # pos_plus_propn: PROPN/X activos; is_stop deja de ser criterio de marcador.
    assert not policy_uses_is_stop("pos_plus_propn")
    assert "PROPN" in policy_content_pos("pos_plus_propn")
    assert "X" in policy_content_pos("pos_plus_propn")
    assert not policy_filters_tool_adverbs("pos_plus_propn")

    # tool_adverb_filtered: como la agresiva + lista cerrada de adverbios.
    assert not policy_uses_is_stop("tool_adverb_filtered")
    assert "PROPN" in policy_content_pos("tool_adverb_filtered")
    assert policy_filters_tool_adverbs("tool_adverb_filtered")


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        resolve_retention_policy("aggressive")
    with pytest.raises(ValueError):
        resolve_retention_policy(None, "no_existe")


def test_effective_default_and_precedence():
    # (1) Sin Config ni gramática: default efectivo = conservadora.
    assert DEFAULT_RETENTION_POLICY == "tool_adverb_filtered"
    assert resolve_retention_policy(None, None) == "tool_adverb_filtered"

    # (2) ia/gramatica.json manda sobre el default del código.
    assert resolve_retention_policy(None, "pos_only") == "pos_only"
    assert (
        resolve_retention_policy(None, "tool_adverb_filtered")
        == "tool_adverb_filtered"
    )

    # (3) Config manda sobre ia/gramatica.json: no repetimos el footgun de
    # content_pos (que load_gramatica pisa siempre).
    assert resolve_retention_policy("pos_plus_propn", "pos_only") == "pos_plus_propn"
    assert resolve_retention_policy("pos_only", "tool_adverb_filtered") == "pos_only"


def test_default_is_the_same_in_code_gramatica_and_json():
    # Las tres definiciones del default (código, DEFAULT_GRAMATICA, JSON)
    # deben coincidir; si divergen, la precedencia miente.
    from reii.gramatica import DEFAULT_GRAMATICA, load_gramatica

    assert DEFAULT_GRAMATICA["retention_policy"] == DEFAULT_RETENTION_POLICY
    assert load_gramatica()["retention_policy"] == DEFAULT_RETENTION_POLICY


def test_policies_keep_their_semantics():
    kwargs = dict(pos="ADV", lemma="no", is_stop=True, is_alpha=True)
    # Baseline: el stopword de contenido cae en marcadores (predicado muerto).
    assert token_decision(policy="pos_only", **kwargs) == "marker"
    # Agresiva: se rescata.
    assert token_decision(policy="pos_plus_propn", **kwargs) == "active"
    # Conservadora: el adverbio-herramienta sigue en marcadores.
    assert token_decision(policy="tool_adverb_filtered", **kwargs) == "marker"

    # Un NOUN no-stop es activo en las tres políticas.
    for policy in RETENTION_POLICIES:
        assert (
            token_decision(
                policy=policy, pos="NOUN", lemma="agua", is_stop=False, is_alpha=True
            )
            == "active"
        )

    # PROPN sólo entra donde la política lo habilita.
    assert (
        token_decision(
            policy="pos_only", pos="PROPN", lemma="lima", is_stop=False, is_alpha=True
        )
        == "drop"
    )
    assert (
        token_decision(
            policy="pos_plus_propn",
            pos="PROPN",
            lemma="lima",
            is_stop=False,
            is_alpha=True,
        )
        == "active"
    )

    # Clase cerrada y tokens no alfa: marcador / descarte, como antes.
    assert (
        token_decision(
            policy="pos_plus_propn", pos="DET", lemma="el", is_stop=False, is_alpha=True
        )
        == "marker"
    )
    assert (
        token_decision(
            policy="pos_plus_propn",
            pos="NOUN",
            lemma="x",
            is_stop=False,
            is_alpha=False,
        )
        == "drop"
    )


def test_tool_adverb_list_only_applies_to_adverbs():
    lemma = sorted(TOOL_ADVERB_LEMMAS)[0]
    assert (
        token_decision(
            policy="tool_adverb_filtered",
            pos="ADV",
            lemma=lemma,
            is_stop=False,
            is_alpha=True,
        )
        == "marker"
    )
    # El mismo lema con POS adjetivo es contenido legítimo ("solo" = alone).
    assert (
        token_decision(
            policy="tool_adverb_filtered",
            pos="ADJ",
            lemma="solo",
            is_stop=False,
            is_alpha=True,
        )
        == "active"
    )


def test_token_table_is_complete_and_invalidates_by_fingerprint(tmp_path):
    path = tmp_path / "token_table.jsonl"
    doc = [
        _FakeToken("Hola", "hola", "INTJ", True, True),
        _FakeToken("mundo", "mundo", "NOUN", False, True),
        _FakeToken(".", ".", "PUNCT", False, False),
    ]

    rows = token_rows(doc)
    assert len(rows) == len(doc) == 3
    assert rows[1] == {
        "forma": "mundo",
        "lemma": "mundo",
        "pos": "NOUN",
        "is_stop": False,
        "is_alpha": True,
    }

    writer = TokenTableWriter(str(path), "fp-v1")
    writer.add("0_0_1", rows, words=2)
    writer.add("0_0_1", rows)  # idempotente por uce_id
    writer.close()

    table = read_token_table(str(path), "fp-v1")
    assert table is not None
    assert list(table) == ["0_0_1"]
    # nº de filas de la tabla == nº de tokens de la UCE.
    assert len(table["0_0_1"]["tokens"]) == len(doc)
    assert table["0_0_1"]["words"] == 2

    # Corpus distinto => cache inválida. None distingue "inválida" de "vacía".
    assert read_token_table(str(path), "fp-v2") is None

    # Reabrir con la huella correcta continúa la tabla sin duplicar filas.
    writer2 = TokenTableWriter(str(path), "fp-v1")
    writer2.add("0_0_1", rows)
    writer2.add("0_0_2", rows[:1], words=1)
    writer2.close()

    table2 = read_token_table(str(path), "fp-v1")
    assert table2 is not None
    assert len(table2) == 2
    assert len(table2["0_0_1"]["tokens"]) == 3

    # Una huella distinta reescribe la tabla (no mezcla corpus).
    writer3 = TokenTableWriter(str(path), "fp-v2")
    writer3.add("9_9_9", rows[:1])
    writer3.close()
    stale = read_token_table(str(path), "fp-v1")
    assert stale is None
    fresh = read_token_table(str(path), "fp-v2")
    assert fresh is not None
    assert list(fresh) == ["9_9_9"]
