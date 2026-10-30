"""Regresión: `min_cluster_size_cdh` debe ser un piso DURO del tamaño de clase.

Antes del fix, `_min_cluster_size` era solo una regla de parada de la recursión
(`if n < min: hoja`) y `_corte_optimo` elegía libremente entre k=1..n-1. Efecto:
clases de 1-2 UCs, K inflado (16 con min=39 sobre n=392) y K inestable entre
corridas con segmentación idéntica (3 / 7 / 16).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from reii.main_workflow import (
    ClasificadorDescendente,
    Config,
    Optimizador,
    UCBuilderConfig,
    _cdh_frac_from_params,
)

FLOOR_FRAC = 0.10
N_UC = 400


def _cfg(**kw):
    base = dict(
        min_cluster_size_cdh=FLOOR_FRAC,
        random_state=0,
        n_perm_cdh=100,
        swap_iterations=1,
    )
    base.update(kw)
    return Config(**base)


def _blocks(n_uc=400, n_blocks=8, terms_per_block=10):
    """n_uc UCs en n_blocks bloques de vocabulario disjunto."""
    per = n_uc // n_blocks
    rows, cols = [], []
    for b in range(n_blocks):
        for i in range(b * per, (b + 1) * per):
            for t in range(b * terms_per_block, (b + 1) * terms_per_block):
                rows.append(i)
                cols.append(t)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n_uc, n_blocks * terms_per_block))


def test_corte_optimo_no_isola_outlier_cuando_hay_piso():
    """Sin min_side, el mejor corte aísla el outlier (1 vs n-1). Con piso, no."""
    coord = np.array([0.0] * 19 + [1.0] * 20 + [100.0])
    cdh = ClasificadorDescendente(_cfg())

    var_libre, lab_libre = cdh._corte_optimo(coord)
    assert var_libre > 0.0
    assert int(lab_libre.sum()) == 1  # corta 39 vs 1: clase de una sola UC

    var_piso, lab_piso = cdh._corte_optimo(coord, min_side=5)
    assert var_piso > 0.0
    # Con piso, el corte más extremo admisible deja 5 UCs al lado del outlier.
    assert int(np.bincount(lab_piso).min()) >= 5
    assert int(lab_piso.sum()) == 5


def test_ninguna_clase_bajo_el_piso():
    mat = _blocks()
    floor = max(5, int(N_UC * FLOOR_FRAC))
    cdh = ClasificadorDescendente(_cfg())
    labels, _ = cdh.clasificar(mat)

    sizes = np.bincount(labels[labels >= 0])
    assert sizes.size >= 2, "el corpus sintético debe particionarse"
    assert int(sizes.min()) >= floor, f"clases={sizes.tolist()} piso={floor}"
    assert int(sizes.sum()) == N_UC


def test_piso_desactivado_recupera_comportamiento_laxo():
    """Con cdh_enforce_min_leaf=False se permite partir con n >= floor (no 2*floor)."""
    mat = _blocks()
    floor = max(5, int(N_UC * FLOOR_FRAC))
    cdh = ClasificadorDescendente(_cfg(cdh_enforce_min_leaf=False))
    labels, _ = cdh.clasificar(mat)

    sizes = np.bincount(labels[labels >= 0])
    assert int(sizes.sum()) == N_UC
    # El modo laxo no garantiza el piso; este corpus no debe romperlo, pero
    # tampoco debe reventar ni dejar UCs sin etiqueta.
    assert sizes.size >= 2


def test_piso_infactible_no_divide():
    """n < 2*floor en la raíz: una sola clase con todo el corpus.

    El piso absoluto es 5 (max(5, frac*n)), así que n=8 < 2*5=10.
    """
    mat = _blocks(n_uc=8, n_blocks=2, terms_per_block=5)
    cdh = ClasificadorDescendente(_cfg())
    labels, _ = cdh.clasificar(mat)

    assert len(np.unique(labels)) == 1
    assert int((labels >= 0).sum()) == 8


# ── Techo de `cdh_target_k` (patch 17) ─────────────────────────────────
# `cdh_target_k` NO es un objetivo: `_cdh_frac_from_params` lo convierte en
# `min_cluster_size_cdh = 1/(2K)`, así que subirlo REDUCE el piso de hoja y
# el CDH particiona MÁS. Medido en `reii_probe_v5_stage1_wc` (n=528):
# kT<=12 -> K<=18 (viable); kT>=16 -> K=25..34, fuera del gate. El techo del
# espacio se acota a `_CDH_TARGET_K_HI` (patch 17).


def _search_space(total_uces):
    """Espacio de búsqueda sin segmentar corpus (evita `Optimizador.__init__`)."""
    opt = Optimizador.__new__(Optimizador)
    opt.config_base = Config()
    opt.uc_config = UCBuilderConfig()
    opt.retrofitted_vectors = None
    opt.total_uces = total_uces
    return opt.get_search_space()


def test_cdh_target_k_ceiling_on_probe_corpus():
    """n=528 (reii_probe_v5_stage1_wc): el techo es el explícito, no 20."""
    lo, hi = _search_space(528)["cdh_target_k"]
    assert hi <= 12, f"cdh_target_k hi={hi} viola el techo medido"
    assert lo == 3


def test_cdh_target_k_ceiling_respects_small_corpus():
    """Corpus pequeño: manda n//10 (8) y el mínimo nunca baja de 3."""
    lo, hi = _search_space(80)["cdh_target_k"]
    assert hi == 8 and lo == 3
    lo2, hi2 = _search_space(10)["cdh_target_k"]
    assert lo2 == 3 and hi2 == 3


def test_cdh_target_k_ceiling_unknown_corpus_size():
    """total_uces <= 0: techo nuevo (12), no el 20 anterior."""
    lo, hi = _search_space(0)["cdh_target_k"]
    assert hi == 12 and lo == 3


def test_cdh_target_k_frac_at_ceiling_keeps_nonzero_floor():
    """kT=12 -> frac=1/24; sobre n=528 el piso de hoja no degenera en 0."""
    frac = _cdh_frac_from_params({"cdh_target_k": 12})
    assert frac == pytest.approx(1.0 / 24.0)
    assert int(round(frac * 528)) >= 1
