# Cableado del bootstrap de estabilidad (patch 20)

**El bug.** `run` pasaba `self._stability_blocks(getattr(self, "_stability_spaces",
None))` (`main_workflow.py:6496-6497`) pero **nadie asignaba** el atributo: era
siempre `None`, `_stability_blocks` (`:5689`) devolvia `{}`, todo `bootstrap_ari`
quedaba `nan` y `stability_kind` (`method_selection.py:213-224`) caia siempre a
`"perturbation"`: el criterio principal del punto 4 nunca estuvo operativo.

**Que se cablea** (opt-in `--stability-bootstrap`, default `False`):

| metodo  | unidades                                     | `X`                                        | `predict_fn`                                          |
| ------- | -------------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| A wc    | UCs tight (`main_workflow.py:4472-4477`)     | UC x terminos binaria->PPMI, la del CDH    | CDH reajustado, piso **absoluto** del ajuste completo |
| B coref | UCs tight (`main_workflow.py:5108-5125`)     | idem                                       | idem, sin warm-start (mismo punto fijo)               |
| C emb   | UCEs con embedding (`main_workflow.py:5240`) | `embedding_matrix` post-PCA (`:5267-5270`) | `guarded_fit`, mismo `target_k`, piso re-escalado     |

Con `False` el builder corta en su primera linea y devuelve `None`: mismo `{}`
que antes. Siempre corre `embeddable_mask` (mismo valor que `zero_rows`) y una clave extra en C.

**Decision sobre C.** C **si** se cablea: `BACKEND_C.md` §4 mide a n=528/d=30
dense_cdh 6.3-39.1 ms, ward 4.1-20.9 ms y kmeans 29.6-218.4 ms, y las replicas
usan ~124 filas. Solo 154/528 UCEs tienen embedding (`PAPER_GAPS.md:164`): se
usan esas (superan `min_n=30`) y si quedan <30 C se omite. `hdbscan` fuera (no
controla K). Cablear C desbloquea `ward`/`kmeans_spherical`, hoy descartados por
`reject_tautological` (`method_selection.py:514-522`: ARI tight/loose = 1.0).

**Coste.** `n_boot=10` => 10 refits por metodo + 50 permutaciones sin refit. A/B
dominan y no estan medidos a esta escala: ~10^2 ms por CDH sobre 200-400 UCs x
5k terminos => **~1-10 s por corrida `select`**. No usarlo en optimizacion:
`select` corre una vez por trial (`main_workflow.py:5964-5966`, >1000 veces).

**Riesgos residuales.**
- A/B remuestrean UCs, no UCEs: `bootstrap_ari` es comparable, pero su
  `coverage_stable` es fraccion de UCs (C la mide sobre UCEs) y alimenta el gate
  `coverage_floor` (`method_selection.py:502-504`).
- A/B miden el clasificador, no la segmentacion (determinista: no se re-segmenta,
  exigiria Stanza por replica).
- `spec_hash` (`main_workflow.py:6224`) hashea `asdict(config)`: cambia por el
  campo nuevo aunque el flag este apagado; solo lo lee `9_selection_audit.py`.
- Memoria: A/B densifican, acotado por `_STABILITY_MAX_DENSE_CELLS` (20M celdas).
