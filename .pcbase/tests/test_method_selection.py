"""Tests de la regla de selección automática de método (``select-v1``).

Los casos usan números REALES medidos en logs del proyecto:

* ``output_2.log`` (DUAL WC+SIM, 528 UCE): A 320 UCEs (60.6 %) ARI=0.350;
  B 219 (41.5 %) ARI=0.212.
* ``output.log`` (DUAL WC+SIM): A 141 (26.7 %) ARI=0.110; B 198 (37.5 %)
  ARI=0.073 — aquí el ARI y la cobertura apuntan a métodos distintos.
* Group ``spec=e3ddf421 mode=wc_only`` (18/22/26): A K=7, coverage 0.39.

La regla es pura: no importa numpy, ni el pipeline, ni la BD.
"""

import json

import pytest

from reii.method_selection import (
    RULE_VERSION,
    MethodMetrics,
    Selection,
    audit_history,
    audit_text,
    load_selection,
    persist_selection,
    select_method,
)


def _m(method, k, ari, coverage, noise=0.0, perturbation=None, available=True):
    return MethodMetrics(
        method=method,
        k=k,
        ari=ari,
        coverage=coverage,
        noise_frac=noise,
        perturbation=perturbation or f"{method}1<->{method}2",
        available=available,
    )


class TestRule:
    def test_clear_winner_is_not_ambiguous(self):
        """output_2.log: A gana por ARI (0.350 vs 0.212) y también por K."""
        sel = select_method([_m("A", 7, 0.350, 0.606), _m("B", 9, 0.212, 0.415)])
        assert sel.winner == "A"
        assert sel.runner_up == "B"
        assert sel.delta_ari == pytest.approx(0.138)
        assert sel.ambiguous is False
        assert sel.criteria_conflict is False
        assert sel.rule_version == RULE_VERSION

    def test_criteria_conflict_and_ambiguity(self):
        """output.log: A gana por ARI (0.110 vs 0.073) pero B cubre más.

        Con margin=0.05 el delta (0.037) cae dentro de la banda -> AMBIGUO.
        Con el default margin=0.02 no es ambiguo, pero el conflicto de
        criterios se declara igual.
        """
        sel = select_method(
            [_m("A", 6, 0.110, 0.267), _m("B", 8, 0.073, 0.375)], margin=0.05
        )
        assert sel.winner == "A"
        assert sel.ambiguous is True
        assert sel.criteria_conflict is True

        strict = select_method(
            [_m("A", 6, 0.110, 0.267), _m("B", 8, 0.073, 0.375)], margin=0.02
        )
        assert strict.winner == "A"
        assert strict.ambiguous is False
        assert strict.criteria_conflict is True

    def test_coverage_floor_discards_method(self):
        """C con ARI alto pero coverage por debajo del piso no compite."""
        sel = select_method(
            [
                _m("A", 7, 0.400, 0.394),
                _m("C", 12, 0.900, 0.050),  # ARI envidiables, 5 % de material
            ]
        )
        assert sel.winner == "A"
        assert [d["method"] for d in sel.discarded] == ["C"]
        assert "coverage" in sel.discarded[0]["why"]

    def test_k_bounds_discard(self):
        sel = select_method(
            [_m("A", 1, 0.900, 0.900), _m("B", 7, 0.300, 0.400)],
            k_bounds=(2, 20),
        )
        assert sel.winner == "B"
        assert sel.discarded[0]["method"] == "A"
        assert "K=1" in sel.discarded[0]["why"]

    def test_unavailable_method_never_wins(self):
        sel = select_method(
            [_m("A", 7, 0.40, 0.40), _m("C", 9, 0.99, 0.90, available=False)]
        )
        assert sel.winner == "A"

    def test_no_viable_method(self):
        sel = select_method([_m("A", 1, 0.5, 0.5), _m("B", 30, 0.4, 0.5)])
        assert sel.winner is None
        assert sel.winner_k == 0
        assert len(sel.discarded) == 2
        assert "ningun metodo" in sel.reason
        assert "[X]" in sel.explain_text()

    def test_exact_tie_is_deterministic(self):
        """Empate total: gana A por orden canónico, y se declara ambiguo."""
        sel = select_method([_m("B", 7, 0.30, 0.40), _m("A", 7, 0.30, 0.40)])
        assert sel.winner == "A"
        assert sel.ambiguous is True
        assert sel.delta_ari == pytest.approx(0.0)

    def test_tie_broken_by_coverage(self):
        sel = select_method([_m("A", 7, 0.30, 0.40), _m("B", 7, 0.30, 0.45)])
        assert sel.winner == "B"
        assert "coverage" in sel.reason
        assert sel.criteria_conflict is False

    def test_empty_metrics(self):
        sel = select_method([])
        assert sel.winner is None
        assert sel.abstained is True
        assert sel.reason == "no se recibieron metricas"


class TestSelectV2:
    """select-v2: K minimo 3, coverage util (sin clases diminutas) y abtencion.

    Casos tomados del log real del optimizador (528 UCEs):
      A K=17 ARI=0.087 cov=0.252 | C K=2 ARI=1.000 cov=0.288  -> antes ganaba C
    """

    def test_k2_no_es_viable(self):
        sel = select_method([_m("A", 17, 0.087, 0.252), _m("C", 2, 1.000, 0.288)])
        assert sel.winner == "A"
        assert [d["method"] for d in sel.discarded] == ["C"]
        assert "K=2" in sel.discarded[0]["why"]

    def test_tiny_reduce_coverage_util(self):
        """Clases diminutas no cuentan como material analizable."""
        m = MethodMetrics("A", 16, 0.900, 0.300, tiny_frac=0.200)
        assert m.usable_coverage == pytest.approx(0.100)
        sel = select_method([m, _m("C", 4, 0.200, 0.300)])
        assert sel.winner == "C"
        assert "coverage util" in sel.discarded[0]["why"]
        assert "tiny" in sel.discarded[0]["why"]

    def test_abstencion_cuando_nada_es_viable(self):
        sel = select_method([_m("A", 2, 0.900, 0.050), _m("C", 1, 0.900, 0.020)])
        assert sel.winner is None
        assert sel.abstained is True
        text = sel.explain_text()
        assert "SIN CLASIFICACION" in text
        text.encode("cp1252")  # consola del host

    def test_desempate_por_coverage_util(self):
        a = MethodMetrics("A", 7, 0.300, 0.400, tiny_frac=0.100)  # usable 0.300
        c = MethodMetrics("C", 7, 0.300, 0.320)  # usable 0.320
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert "coverage util" in sel.reason

    def test_noise_alto_descarta(self):
        """Mas de medio corpus sin clasificar no es una particion, por ARI alto
        que tenga (HDBSCAN con mucho -1)."""
        c = MethodMetrics("C", 5, 0.990, 0.400, noise_frac=0.600)
        a = _m("A", 7, 0.300, 0.400)
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert "ruido" in sel.discarded[0]["why"]

    def test_noise_bajo_no_descarta(self):
        """El ruido se reporta y se ve en la tabla; 0.40 < 0.50 es viable."""
        c = MethodMetrics("C", 5, 0.990, 0.200, noise_frac=0.400)
        a = _m("A", 7, 0.300, 0.400)
        sel = select_method([a, c])
        row_c = next(r for r in sel.all_metrics if r["method"] == "C")
        assert row_c["noise_frac"] == pytest.approx(0.400)
        assert row_c["usable"] == pytest.approx(0.200)
        assert sel.winner == "C"  # ARI 0.99 con 20% util > ARI 0.30 con 40%

    def test_dominancia_092_descarta(self):
        """Particion C real medida: {442, 25, 12} de 479 asignadas = 0.92.

        Una clase gigante + satelites es degenerada y su ARI 0.52-0.78 esta
        inflado por construccion: el gate la descarta aunque lidere el ARI.
        """
        c = MethodMetrics("C", 3, 0.785, 0.371, max_class_share=442 / 479)
        a = _m("A", 4, 0.250, 0.574)
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert "dominante" in sel.discarded[0]["why"]
        assert "0.92" in sel.discarded[0]["why"]

    def test_dominancia_089_con_ruido_no_descarta(self):
        """0.89 de las asignadas con 30% de ruido = 0.62 del corpus: viable.

        Reemplaza a ``test_dominancia_089_no_descarta``: el umbral ya no es
        0.90 sobre lo asignado sino 0.70 sobre el corpus, asi que 0.89 sin
        ruido SI se descarta; con ruido, no.
        """
        c = MethodMetrics(
            "C", 3, 0.785, 0.371, noise_frac=0.30, max_class_share=0.89
        )
        a = _m("A", 4, 0.250, 0.574)
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert not any("dominante" in d["why"] for d in sel.discarded)
        # c = 0.89 * (1 - 0.30) = 0.623 del corpus
        assert sel.winner_dominance_corpus == pytest.approx(0.623)

    def test_dominancia_gate_aparece_en_explain(self):
        """El umbral se imprime en unidades del corpus (0.70)."""
        c = MethodMetrics("C", 3, 0.785, 0.371, max_class_share=0.60)
        sel = select_method([c])
        text = sel.explain_text()
        assert "clase mayor=0.60" in text
        assert "= 0.60 del corpus" in text
        assert "gate: >0.70 del corpus" in text


class TestSerialization:
    def test_to_dict_is_strict_json(self):
        sel = select_method([_m("A", 7, 0.35, 0.60)])
        # allow_nan=False: NaN/inf romperían cualquier consumidor JSON.
        json.dumps(sel.to_dict(), allow_nan=False)

    def test_roundtrip_preserves_decision(self):
        sel = select_method([_m("A", 7, 0.350, 0.606), _m("B", 9, 0.212, 0.415)])
        again = Selection.from_dict(sel.to_dict())
        assert again.winner == sel.winner
        assert again.winner_ari == pytest.approx(sel.winner_ari)
        assert again.k_bounds == sel.k_bounds
        assert again.rule_version == RULE_VERSION

    def test_explain_is_ascii_only(self):
        """La consola del host es cp1252: nada de cajas ni flechas exóticas."""
        sel = select_method([_m("A", 7, 0.35, 0.60), _m("B", 9, 0.15, 0.40)])
        text = sel.explain_text()
        text.encode("cp1252")
        assert "AMBIGUO" not in text  # delta grande: no hay casi-empate


class TestPersistence:
    def test_persist_and_load_with_fingerprint(self, tmp_path):
        snap = tmp_path / "method_selection.json"
        hist = tmp_path / "method_selection_log.jsonl"
        sel = select_method([_m("A", 7, 0.35, 0.60)])
        persist_selection(
            sel,
            str(snap),
            history_path=str(hist),
            context={
                "corpus_fingerprint": "fp1",
                "uce_target_size": 26,
                "mode": "select",
            },
        )
        assert snap.exists() and hist.exists()

        loaded = load_selection(str(snap), corpus_fingerprint="fp1", uce_target_size=26)
        assert loaded is not None and loaded.winner == "A"

        # Otro corpus -> no se reutiliza la elección (invalidación explícita).
        assert load_selection(str(snap), corpus_fingerprint="fp2") is None
        # Otro tamaño de UCE -> tampoco.
        assert (
            load_selection(str(snap), corpus_fingerprint="fp1", uce_target_size=30)
            is None
        )

    def test_history_is_append_only(self, tmp_path):
        snap = tmp_path / "s.json"
        hist = tmp_path / "log.jsonl"
        for target, winner_ari in ((18, 0.42), (26, 0.39)):
            sel = select_method([_m("A", 7, winner_ari, 0.39)])
            persist_selection(
                sel,
                str(snap),
                history_path=str(hist),
                context={
                    "corpus_fingerprint": "fp1",
                    "uce_target_size": target,
                    "mode": "wc_only",
                },
            )
        lines = hist.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["rule_version"] == RULE_VERSION for line in lines)

    def test_load_missing_file_returns_none(self, tmp_path):
        assert load_selection(str(tmp_path / "nope.json")) is None


class TestAudit:
    def _record(self, fp, target, winner, ari=0.4, coverage=0.4, ambiguous=False):
        return {
            "rule_version": RULE_VERSION,
            "context": {
                "corpus_fingerprint": fp,
                "uce_target_size": target,
                "mode": "wc_only",
                "written_at": f"2026-10-27T00:00:0{target % 10}",
            },
            "selection": {
                "winner": winner,
                "winner_ari": ari,
                "winner_coverage": coverage,
                "winner_k": 7,
                "ambiguous": ambiguous,
                "criteria_conflict": False,
            },
        }

    def test_stable_winner_across_targets(self):
        report = audit_history(
            [self._record("fp1", 18, "A"), self._record("fp1", 26, "A")]
        )
        assert report["instability"] == []
        assert "[ok]" in audit_text(
            [self._record("fp1", 18, "A"), self._record("fp1", 26, "A")]
        )

    def test_winner_flip_is_reported(self):
        recs = [self._record("fp1", 30, "C"), self._record("fp1", 18, "A")]
        report = audit_history(recs)
        assert len(report["instability"]) == 1
        assert report["instability"][0]["winners"] == ["A", "C"]
        text = audit_text(recs)
        text.encode("cp1252")
        assert "no es identificable" in text

    def test_ambiguous_rows_are_listed(self):
        report = audit_history([self._record("fp1", 18, "A", ambiguous=True)])
        assert len(report["ambiguous"]) == 1
