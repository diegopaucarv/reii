"""Contrato de ``select-v4`` (AGENTS: no se toca ``select-v3`` desde aqui).

Este fichero es la especificacion ejecutable de ``docs/design_v5/SELECT_V4.md``.
Se activa solo cuando el diff de la seccion 7 del documento esta aplicado:

    RULE_VERSION == "select-v4"

Hasta entonces todos los tests se **saltan** (la suite queda verde) en vez de
fallar: el diff lo aplica el usuario a mano sobre ``src/**``, que es un scope de
escritura prohibido para este agente.

Cubre los seis requisitos del encargo:

(a) C denso ya NO gana por construccion cuando su ``bootstrap_ari``/``null_p``
    son malos                        -> ``TestDensoNoGanaPorConstruccion``
(ii) C con ``hdbscan`` nunca gana    -> ``TestDiagnosticoC``
(iii) 0.837 con nu=0 se descarta     -> ``TestDominanciaCorpus``
(iv) abstención                      -> ``TestAbstencion``
(v) ``criteria_conflict`` con kinds  -> ``test_kinds_distintos_marcan_conflicto``
(vi) snapshot con ``rule_version``
     vieja se ignora                 -> ``TestSnapshotVersion``
(vii) métricas minimas no lanzan     -> ``TestMetricasMinimas``

La regla es pura: no importa numpy, ni el pipeline, ni la BD. Tiempo esperado
~0.2 s.
"""

import json

import pytest

from reii import method_selection as _ms

# ``import *``-equivalente tolerante: con ``select-v3`` aplicado el modulo no
# tiene las señales nuevas y este fichero debe poder COLECTARSE (para que el
# ``skipif`` de abajo lo marque como saltado) en vez de fallar al importar.
RULE_VERSION = _ms.RULE_VERSION
MethodMetrics = _ms.MethodMetrics
Selection = _ms.Selection
load_selection = _ms.load_selection
persist_selection = _ms.persist_selection
select_method = _ms.select_method
last_load_rejection = getattr(_ms, "last_load_rejection", lambda: None)

pytestmark = pytest.mark.skipif(
    RULE_VERSION != "select-v4",
    reason=(
        "select-v4 no aplicado en src/reii/method_selection.py; "
        "aplicar el diff de docs/design_v5/SELECT_V4.md seccion 7"
    ),
)

NAN = float("nan")


def _m(method, k, ari, coverage, **kw):
    """``MethodMetrics`` con perturbacion por defecto (A1<->A2, ...)."""
    kw.setdefault("perturbation", f"{method}1<->{method}2")
    return MethodMetrics(method=method, k=k, ari=ari, coverage=coverage, **kw)


def _c(**kw):
    """C denso: los defaults NO son competitivos (bootstrap y nulo malos)."""
    defaults = dict(
        backend="dense_cdh",
        bootstrap_ari=0.30,
        null_p=0.50,
        coverage_stable=0.40,
        max_class_share=0.35,
    )
    defaults.update(kw)
    return _m(
        "C",
        defaults.pop("k", 4),
        defaults.pop("ari", 0.90),
        defaults.pop("coverage", 1.0),
        **defaults,
    )


# ══════════════════════════════════════════════════════════════════════════
# (a) estabilidad real: C denso no gana por construccion
# ══════════════════════════════════════════════════════════════════════════
class TestDensoNoGanaPorConstruccion:
    def test_denso_no_gana_si_su_bootstrap_es_peor(self):
        """C con ``ari=1.0`` (tautologico) y bootstrap 0.25 pierde con A 0.55."""
        a = _m("A", 5, 0.550, 0.600, bootstrap_ari=0.55, null_p=0.010)
        c = _c(k=4, ari=1.000, bootstrap_ari=0.25, null_p=0.010)
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert sel.winner_stability == pytest.approx(0.550)
        assert sel.winner_stability_kind == "bootstrap"
        # C (bootstrap 0.25) no llega a runner-up: STABILITY_MEAN_MIN=0.30
        # (STABILITY_NULL.md) lo descarta por falta de senal de estructura.
        assert sel.runner_up is None
        assert [d["method"] for d in sel.discarded] == ["C"]
        assert "piso 0.30" in sel.discarded[0]["why"]
        # Sin runner-up no hay conflicto de criterios que declarar.
        # (El conflicto aparece cuando el runner-up cubre mas o cuando
        # los kinds difieren: tests dedicados abajo.)
        assert sel.criteria_conflict is False

    def test_denso_no_gana_sin_evidencia_contra_el_nulo(self):
        """C con bootstrap 0.70 pero ``null_p=0.61``: descartado por el gate."""
        a = _m("A", 5, 0.400, 0.500, bootstrap_ari=0.40, null_p=0.010)
        c = _c(bootstrap_ari=0.700, coverage_stable=0.900, null_p=0.610)
        # El gate del nulo ya no es el default (SELECT_V4.md 3.7.4: el
        # default es "diagnostic"); este test verifica el GATE, asi que lo
        # pide explicito.
        sel = select_method([a, c], null_mode="gate")
        assert sel.winner == "A"
        why = [d["why"] for d in sel.discarded if d["method"] == "C"]
        assert why and "nulo" in why[0]
        assert "p=0.610" in why[0]

    def test_denso_legitimo_si_gana(self):
        """Control: la regla no es anti-C. bootstrap 0.72 + p=0.004 gana."""
        a = _m("A", 4, 0.300, 0.400, bootstrap_ari=0.300, null_p=0.020)
        c = _c(
            k=4,
            ari=0.960,
            backend="ward",
            bootstrap_ari=0.720,
            coverage_stable=0.900,
            max_class_share=0.410,
            null_p=0.004,
        )
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert sel.winner_stability == pytest.approx(0.720)


# ══════════════════════════════════════════════════════════════════════════
# (ii) C con hdbscan nunca gana
# ══════════════════════════════════════════════════════════════════════════
class TestDiagnosticoC:
    def test_hdbscan_con_flag_no_compite(self):
        a = _m("A", 7, 0.300, 0.400)
        c = _m(
            "C",
            5,
            0.990,
            0.900,
            noise_frac=0.12,
            backend="hdbscan",
            diagnostic_only=True,
            bootstrap_ari=0.60,
            max_class_share=0.40,
            null_p=0.004,
        )
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert sel.discarded[0]["method"] == "C"
        assert "diagnostico" in sel.discarded[0]["why"]
        assert "hdbscan" in sel.discarded[0]["why"]

    def test_hdbscan_sin_flag_tampoco_compite(self):
        """Fail-closed: no se confia en que el productor ponga el flag."""
        a = _m("A", 7, 0.300, 0.400)
        c = _m(
            "C",
            5,
            0.990,
            0.900,
            backend="hdbscan",
            diagnostic_only=False,
            bootstrap_ari=0.90,
            coverage_stable=0.95,
            max_class_share=0.20,
            null_p=0.004,
        )
        assert c.diagnostic_only is False
        assert select_method([a, c]).winner == "A"

    def test_hdbscan_elegible_solo_si_se_desactiva_explicitamente(self):
        a = _m("A", 7, 0.300, 0.400)
        c = _m(
            "C",
            5,
            0.990,
            0.900,
            backend="hdbscan",
            diagnostic_only=False,
            bootstrap_ari=0.90,
            coverage_stable=0.95,
            max_class_share=0.20,
            null_p=0.004,
        )
        sel = select_method([a, c], diagnostic_backends=())
        assert sel.winner == "C"


# ══════════════════════════════════════════════════════════════════════════
# (iii) dominancia sobre el corpus
# ══════════════════════════════════════════════════════════════════════════
class TestDominanciaCorpus:
    def test_dominancia_corpus_0837_descarta(self):
        """{442,45,30,11} sobre 528: a=0.837 y nu=0 => c=0.837 > 0.70."""
        a = _m("A", 4, 0.250, 0.574)
        c = _c(
            k=3,
            ari=0.785,
            coverage=0.850,
            coverage_stable=0.800,
            bootstrap_ari=0.620,
            max_class_share=442 / 528,
            null_p=0.004,
        )
        assert c.dominance_corpus_effective == pytest.approx(442 / 528)
        sel = select_method([a, c])
        assert sel.winner == "A"
        why = [d["why"] for d in sel.discarded if d["method"] == "C"][0]
        assert "dominante" in why and "0.84" in why
        assert "0.70" in why

    def test_dominancia_ignora_el_ruido_en_el_denominador(self):
        """Misma c=0.837 con nu=0 y con nu=0.119: mismo veredicto."""
        a = _m("A", 4, 0.250, 0.574)
        limpia = _c(
            k=3,
            ari=0.785,
            coverage=0.850,
            max_class_share=442 / 528,
            bootstrap_ari=0.620,
            null_p=0.004,
        )
        ruidosa = _c(
            k=3,
            ari=0.785,
            coverage=0.850,
            noise_frac=63 / 528,
            max_class_share=442 / 465,
            bootstrap_ari=0.620,
            null_p=0.004,
        )
        for m in (limpia, ruidosa):
            assert m.dominance_corpus_effective == pytest.approx(0.8371, abs=1e-3)
            sel = select_method([a, m])
            assert sel.winner == "A"
            assert any("dominante" in d["why"] for d in sel.discarded)

    def test_dominancia_089_con_ruido_no_descarta(self):
        """0.89 de las asignadas con 30% de ruido = 0.62 del corpus: valida."""
        a = _m("A", 4, 0.250, 0.574)
        c = _c(
            k=3,
            ari=0.785,
            coverage=0.850,
            noise_frac=0.30,
            max_class_share=0.89,
            bootstrap_ari=0.620,
            null_p=0.004,
        )
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert not any("dominante" in d["why"] for d in sel.discarded)
        assert sel.winner_dominance_corpus == pytest.approx(0.623)

    def test_dominancia_prefiere_los_conteos(self):
        """``max_class_size/n_total`` manda sobre la identidad a(1-nu)."""
        m = _c(
            k=3,
            ari=0.785,
            coverage=0.850,
            max_class_share=0.20,
            n_total=528,
            max_class_size=442,
            n_assigned=528,
            n_noise=0,
        )
        assert m.dominance_corpus_effective == pytest.approx(442 / 528)


# ══════════════════════════════════════════════════════════════════════════
# (iv) abstención
# ══════════════════════════════════════════════════════════════════════════
class TestAbstencion:
    def test_abstiene_sin_viables(self):
        sel = select_method([_m("A", 2, 0.900, 0.050), _m("C", 1, 0.900, 0.020)])
        assert sel.winner is None
        assert sel.abstained is True
        assert len(sel.discarded) == 2
        assert "ningun metodo" in sel.reason

    def test_abstiene_si_nadie_supera_el_nulo(self):
        a = _m("A", 5, 0.120, 0.300, null_p=0.620)
        b = _m("B", 6, 0.100, 0.350, null_p=0.550)
        c = _c(bootstrap_ari=0.200, coverage_stable=0.500, null_p=0.400)
        # El gate del nulo ya no es el default (SELECT_V4.md 3.7.4: el
        # default es "diagnostic"); este test verifica el GATE, asi que lo
        # pide explicito.
        sel = select_method([a, b, c], null_mode="gate")
        assert sel.winner is None
        assert sel.abstained is True
        assert len(sel.discarded) == 3

    def test_explain_de_abstencion_es_imprimible(self):
        sel = select_method([_m("A", 1, 0.900, 0.900)])
        text = sel.explain_text()
        assert "SIN CLASIFICACION" in text
        text.encode("cp1252")  # consola del host

    def test_unico_candidato_tautologico_no_gana(self):
        """C ward sin bootstrap: no hay estabilidad valida, se abstiene."""
        c = _m(
            "C",
            4,
            1.000,
            1.000,
            backend="ward",
            coverage_stable=0.950,
            max_class_share=0.400,
        )
        sel = select_method([c])
        assert sel.winner is None
        assert "tautologica" in sel.discarded[0]["why"]


# ══════════════════════════════════════════════════════════════════════════
# (v) conflicto de criterios por familias distintas
# ══════════════════════════════════════════════════════════════════════════
class TestFamiliasDeEstabilidad:
    def test_kinds_distintos_marcan_conflicto(self):
        """A mide perturbacion y C bootstrap con delta 0.01: no es comparable."""
        a = _m("A", 6, 0.550, 0.500)  # kind perturbation
        c = _c(
            k=4,
            ari=0.540,
            bootstrap_ari=0.540,
            coverage_stable=0.600,
            max_class_share=0.300,
            null_p=0.004,
        )
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert sel.winner_stability_kind == "perturbation"
        assert sel.runner_up_stability_kind == "bootstrap"
        assert sel.criteria_conflict is True
        assert sel.ambiguous is True  # 0.010 <= margin 0.02
        assert "misma escala" in sel.explain_text()

    def test_kinds_iguales_y_mas_coverage_marcan_conflicto(self):
        a = _m("A", 5, 0.550, 0.600, bootstrap_ari=0.550, null_p=0.010)
        c = _c(
            bootstrap_ari=0.450,  # >= STABILITY_MEAN_MIN (0.30): viable
            coverage_stable=0.850,
            max_class_share=0.450,
            null_p=0.010,
        )
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert sel.criteria_conflict is True
        assert "coverage util" in sel.explain_text()

    def test_kinds_iguales_y_menos_coverage_no_marcan_conflicto(self):
        a = _m("A", 5, 0.550, 0.700, bootstrap_ari=0.550, null_p=0.010)
        c = _c(
            bootstrap_ari=0.450,  # >= STABILITY_MEAN_MIN (0.30): viable
            coverage_stable=0.400,
            max_class_share=0.450,
            null_p=0.010,
        )
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert sel.criteria_conflict is False

    def test_tautologico_no_compite_pero_con_bootstrap_si(self):
        a = _m("A", 4, 0.450, 0.400)
        taut = _m(
            "C",
            4,
            1.000,
            1.000,
            backend="ward",
            coverage_stable=0.950,
            max_class_share=0.400,
            null_p=0.004,
        )
        assert select_method([a, taut]).winner == "A"
        assert "tautologica" in select_method([a, taut]).discarded[0]["why"]

        real = _m(
            "C",
            4,
            1.000,
            1.000,
            backend="ward",
            bootstrap_ari=0.800,
            coverage_stable=0.950,
            max_class_share=0.400,
            null_p=0.004,
        )
        assert select_method([a, real]).winner == "C"

        escaped = select_method([a, taut], reject_tautological=False)
        assert escaped.winner == "C"


# ══════════════════════════════════════════════════════════════════════════
# Cobertura efectiva: ``coverage`` de un denso es 1.0 por construccion
# ══════════════════════════════════════════════════════════════════════════
class TestCoberturaEfectiva:
    def test_coverage_stable_manda_sobre_coverage(self):
        a = _m("A", 4, 0.300, 0.400)
        c = _c(
            ari=0.700,
            bootstrap_ari=0.700,
            coverage=1.000,
            coverage_stable=0.100,
            max_class_share=0.400,
            null_p=0.004,
        )
        sel = select_method([a, c])
        assert sel.winner == "A"
        why = [d["why"] for d in sel.discarded if d["method"] == "C"][0]
        assert "coverage util 0.100" in why
        assert "coverage estable 0.100" in why

    def test_desempate_por_coverage_efectiva(self):
        a = _m("A", 5, 0.400, 0.500, bootstrap_ari=0.400, coverage_stable=0.500)
        b = _m("B", 5, 0.400, 0.500, bootstrap_ari=0.400, coverage_stable=0.600)
        sel = select_method([a, b])
        assert sel.winner == "B"
        assert "coverage util" in sel.reason


# ══════════════════════════════════════════════════════════════════════════
# (vi) el snapshot de otra regla se ignora
# ══════════════════════════════════════════════════════════════════════════
class TestSnapshotVersion:
    def _write(self, path, rule_version, fingerprint="fp1", target=26):
        payload = {
            "rule_version": rule_version,
            "context": {
                "corpus_fingerprint": fingerprint,
                "uce_target_size": target,
                "mode": "select",
                "written_at": "2026-10-27T00:00:00",
            },
            "selection": {
                "winner": "C",
                "winner_ari": 0.785,
                "winner_coverage": 0.371,
                "winner_k": 3,
                "rule_version": rule_version,
                "max_class_share_max": 0.90,
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_snapshot_v3_se_ignora_y_se_registra_el_motivo(self, tmp_path):
        snap = self._write(tmp_path / "method_selection.json", "select-v3")
        assert load_selection(snap, corpus_fingerprint="fp1") is None
        why = last_load_rejection()
        assert why is not None
        assert "select-v3" in why["why"]
        assert "select-v4" in why["why"]

    def test_motivo_tambien_se_acumula_en_explain(self, tmp_path):
        snap = self._write(tmp_path / "method_selection.json", "select-v3")
        notes = []
        assert load_selection(snap, explain=notes) is None
        assert notes and "snapshot ignorado" in notes[0]
        assert "rule_version" in notes[0]

    def test_snapshot_de_la_regla_vigente_se_carga(self, tmp_path):
        snap = tmp_path / "method_selection.json"
        sel = select_method([_m("A", 7, 0.350, 0.606)])
        persist_selection(
            sel,
            str(snap),
            context={"corpus_fingerprint": "fp1", "uce_target_size": 26},
        )
        loaded = load_selection(str(snap), corpus_fingerprint="fp1", uce_target_size=26)
        assert loaded is not None
        assert loaded.winner == "A"
        assert loaded.rule_version == RULE_VERSION
        assert last_load_rejection() is None

    def test_escape_para_leer_historia(self, tmp_path):
        """``expected_rule_version=""`` desactiva el chequeo a proposito."""
        snap = self._write(tmp_path / "method_selection.json", "select-v3")
        loaded = load_selection(snap, expected_rule_version="")
        assert loaded is not None and loaded.winner == "C"

    def test_fingerprint_sigue_invalidando(self, tmp_path):
        snap = self._write(tmp_path / "method_selection.json", RULE_VERSION)
        assert load_selection(snap, corpus_fingerprint="otro") is None
        assert "corpus_fingerprint" in last_load_rejection()["why"]


# ══════════════════════════════════════════════════════════════════════════
# (vii) métricas minimas: nunca excepcion
# ══════════════════════════════════════════════════════════════════════════
class TestMetricasMinimas:
    def test_metricas_de_tres_argumentos_no_lanzan(self):
        sel = select_method([MethodMetrics("A", 1, 0.9, 0.9)])
        assert sel.winner is None
        assert sel.abstained is True
        assert sel.discarded[0]["method"] == "A"
        assert "K=1" in sel.discarded[0]["why"]

    def test_metricas_minimas_viables_no_lanzan(self):
        sel = select_method([MethodMetrics("A", 7, 0.35, 0.60)])
        assert sel.winner == "A"
        assert sel.winner_stability_kind == "perturbation"
        assert sel.null_mode == "diagnostic"
        assert sel.dominance_scale == "corpus"
        sel.explain_text().encode("cp1252")

    def test_lista_vacia_y_nan_no_lanzan(self):
        empty = select_method([])
        assert empty.winner is None and empty.reason == "no se recibieron metricas"

        weird = MethodMetrics("C", 5, NAN, NAN, noise_frac=NAN, bootstrap_ari=NAN)
        sel = select_method([weird])
        assert sel.winner is None
        assert sel.abstained is True


# ══════════════════════════════════════════════════════════════════════════
# Nulo: gate (default) vs rank vs off
# ══════════════════════════════════════════════════════════════════════════
class TestNullModes:
    def _only_bad_null(self):
        return [
            _c(
                ari=0.700,
                bootstrap_ari=0.700,
                coverage_stable=0.900,
                max_class_share=0.400,
                null_p=0.610,
            )
        ]

    def test_gate_abstiene(self):
        # El default es "diagnostic"; el gate se prueba pasandolo
        # explicito (si no, este assert dejaria de verificar el gate).
        sel = select_method(self._only_bad_null(), null_mode="gate")
        assert sel.winner is None

    def test_off_ignora_el_nulo(self):
        sel = select_method(self._only_bad_null(), null_mode="off")
        assert sel.winner == "C"

    def test_rank_no_descarta(self):
        sel = select_method(self._only_bad_null(), null_mode="rank")
        assert sel.winner == "C"
        assert sel.null_mode == "rank"

    def test_rank_antepone_la_evidencia(self):
        a = _m("A", 5, 0.300, 0.400, bootstrap_ari=0.300, null_p=0.010)
        c = _c(
            ari=0.700,
            bootstrap_ari=0.700,
            coverage_stable=0.900,
            max_class_share=0.400,
            null_p=0.610,
        )
        sel = select_method([a, c], null_mode="rank")
        assert sel.winner == "A"
        assert "nulo" in sel.reason

    def test_nan_nunca_descarta(self):
        a = _m("A", 5, 0.300, 0.400)  # null_p = nan
        assert select_method([a]).winner == "A"

    def test_modo_invalido_es_error_de_programacion(self):
        with pytest.raises(ValueError):
            select_method([_m("A", 5, 0.3, 0.4)], null_mode="ranking")


# ══════════════════════════════════════════════════════════════════════════
# Presentacion y serializacion
# ══════════════════════════════════════════════════════════════════════════
class TestExplain:
    def test_imprime_el_umbral_efectivo_no_el_090(self):
        c = _m(
            "C",
            3,
            0.785,
            0.371,
            max_class_share=0.60,
            bootstrap_ari=0.62,
            coverage_stable=0.40,
            null_p=0.004,
        )
        text = select_method([c]).explain_text()
        assert "clase mayor=0.60" in text
        assert "= 0.60 del corpus" in text
        assert "gate: >0.70 del corpus" in text
        assert "0.90" not in text

    def test_umbral_configurable_se_imprime(self):
        # 0.50 queda por debajo del umbral 0.55 => viable, y el explain debe
        # imprimir el umbral EFECTIVO, no el 0.70 por defecto.
        c = _m(
            "C",
            3,
            0.500,
            0.371,
            max_class_share=0.50,
            bootstrap_ari=0.62,
            coverage_stable=0.40,
            null_p=0.004,
        )
        text = select_method([c], max_class_share_max=0.55).explain_text()
        assert "gate: >0.55 del corpus" in text
        assert "= 0.50 del corpus" in text
        assert "0.70" not in text

    def test_columnas_nuevas(self):
        c = _m(
            "C",
            4,
            0.700,
            1.000,
            backend="ward",
            bootstrap_ari=0.700,
            coverage_stable=0.900,
            max_class_share=0.400,
            null_p=0.004,
        )
        text = select_method([c]).explain_text()
        assert "domC" in text and "stab" in text
        text.encode("cp1252")

    def test_explain_ascii_puro(self):
        sel = select_method([_m("A", 7, 0.35, 0.60), _m("B", 9, 0.15, 0.40)])
        sel.explain_text().encode("cp1252")
        assert "AMBIGUO" not in sel.explain_text()


class TestSerializacion:
    def test_to_dict_es_json_estricto(self):
        sel = select_method([_m("A", 7, 0.35, 0.60)])
        json.dumps(sel.to_dict(), allow_nan=False)

    def test_roundtrip_conserva_las_senales_nuevas(self):
        sel = select_method([_m("A", 7, 0.350, 0.606)])
        again = Selection.from_dict(sel.to_dict())
        assert again.winner == sel.winner
        assert again.rule_version == RULE_VERSION
        assert again.dominance_scale == "corpus"
        assert again.null_mode == "diagnostic"
        assert again.null_alpha == pytest.approx(0.05)
        assert again.max_class_share_max == pytest.approx(0.70)
        assert again.winner_stability_kind == "perturbation"

    def test_roundtrip_con_bootstrap(self):
        c = _m(
            "C",
            4,
            1.000,
            1.000,
            backend="ward",
            bootstrap_ari=0.720,
            coverage_stable=0.900,
            max_class_share=0.400,
            null_p=0.004,
        )
        sel = select_method([c])
        again = Selection.from_dict(sel.to_dict())
        assert again.winner_stability == pytest.approx(0.720)
        assert again.winner_stability_kind == "bootstrap"
        assert again.winner_null_p == pytest.approx(0.004)
        assert again.winner_dominance_corpus == pytest.approx(0.400)


class TestDeterminismo:
    def test_dos_llamadas_identicas(self):
        rows = [
            _m("A", 5, 0.400, 0.500, bootstrap_ari=0.40, null_p=0.010),
            _m("B", 5, 0.400, 0.500, bootstrap_ari=0.40, coverage_stable=0.600),
            _c(bootstrap_ari=0.400, coverage_stable=0.400, null_p=0.010),
        ]
        first = select_method(rows).to_dict()
        second = select_method(list(reversed(rows))).to_dict()
        # ``all_metrics`` conserva el orden de entrada (igual que en v3): el
        # contrato de determinismo es sobre la decision, no sobre el listado.
        for payload in (first, second):
            payload["all_metrics"] = sorted(
                payload["all_metrics"], key=lambda r: r["method"]
            )
        assert first == second


# ══════════════════════════════════════════════════════════════════════════
# Backends no conmensurables: ``dense_cdh`` solo mueve un multiplicador
# ══════════════════════════════════════════════════════════════════════════
class TestBackendsNoConmensurables:
    def test_dense_cdh_por_perturbacion_no_compite(self):
        """El pase tight/loose solo mueve ``dense_cdh_loose_factor``."""
        a = _m("A", 5, 0.307, 0.400)
        c = _m(
            "C",
            4,
            0.931,
            1.000,
            backend="dense_cdh",
            coverage_stable=0.479,
            max_class_share=0.400,
            null_p=0.004,
        )
        assert c.stability_kind == "perturbation"
        why = _ms._feasibility(c, 0.15, (3, 20))
        assert why is not None
        assert "dense_cdh" in why
        assert "dense_cdh_loose_factor" in why
        assert "bootstrap_ari" in why
        sel = select_method([a, c])
        assert sel.winner == "A"
        assert [d["method"] for d in sel.discarded] == ["C"]
        assert "dense_cdh" in sel.discarded[0]["why"]

    def test_dense_cdh_con_bootstrap_si_compite(self):
        """Con ``bootstrap_ari`` finito la regla no lo descarta."""
        c = _m(
            "C",
            4,
            0.700,
            1.000,
            backend="dense_cdh",
            bootstrap_ari=0.620,
            coverage_stable=0.900,
            max_class_share=0.400,
            null_p=0.004,
        )
        assert c.stability_kind == "bootstrap"
        assert _ms._feasibility(c, 0.15, (3, 20)) is None
        a = _m("A", 5, 0.307, 0.400)
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert sel.discarded == []
        assert sel.winner_stability_kind == "bootstrap"

    def test_ward_y_kmeans_conservan_el_mensaje_tautologico(self):
        """Los backends tautologicos no cambian su motivo de descarte."""
        for backend in ("ward", "kmeans_spherical"):
            m = _m(
                "C",
                4,
                1.000,
                1.000,
                backend=backend,
                coverage_stable=0.950,
                max_class_share=0.400,
                null_p=0.004,
            )
            why = _ms._feasibility(m, 0.15, (3, 20))
            assert why == (
                f"estabilidad tautologica: backend {backend} no tiene grados"
                " de libertad en el pase tight/loose (ARI=1.0 por construccion);"
                " falta bootstrap_ari"
            )

    def test_reject_tautological_sigue_gobernando(self):
        """El escape cubre los dos casos, no solo la tautologia."""
        c = _m(
            "C",
            4,
            0.931,
            1.000,
            backend="dense_cdh",
            coverage_stable=0.479,
            max_class_share=0.400,
            null_p=0.004,
        )
        assert _ms._feasibility(c, 0.15, (3, 20)) is not None
        assert (
            _ms._feasibility(c, 0.15, (3, 20), reject_tautological=False)
            is None
        )


class TestExplainAlineacion:
    def test_cabecera_y_filas_tienen_el_mismo_ancho(self):
        """Incluye ``stability_kind="perturbation"`` (12 caracteres)."""
        rows = [
            _m("A", 5, 0.310, 0.450),
            _m("B", 6, 0.280, 0.400),
            _m(
                "C",
                4,
                1.000,
                1.000,
                backend="ward",
                bootstrap_ari=0.620,
                coverage_stable=0.950,
                max_class_share=0.400,
                null_p=0.004,
            ),
        ]
        sel = select_method(rows)
        lines = sel.explain_lines()
        header_i = next(
            i for i, line in enumerate(lines) if line.startswith("  met")
        )
        header = lines[header_i]
        # El veredicto/tag es texto libre: el ancho fijo termina en la
        # columna ``perturbacion``, y ahi debe cuadrar la cabecera.
        width = len(header) - len("veredicto")
        method_rows = lines[header_i + 1 : header_i + 1 + len(rows)]
        assert len(method_rows) == len(rows)
        for line in method_rows:
            body = line
            for tag in ("<- runner-up", "<- winner", "no disponible"):
                if body.endswith(tag):
                    body = body[: -len(tag)]
                    break
            assert len(body) == width, f"{len(body)} != {width}: {line!r}"
