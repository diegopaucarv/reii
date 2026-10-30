# VERIFY + CLEANUP v6

Verificación estática de `docs/patches/16..20` sobre el árbol de trabajo
(`C:\Users\Julia\Downloads\reii`, 2026-10-28) y plan de limpieza en 4 bloques.
No se aplicó ningún patch ni se modificó `src/`, `tests/`, `scripts/`, `docs/patches/`.

## 0. Digest de verificación

`git apply --check` individual y en serie 16+17+18+19 → rc=0:

| #   | patch                           |     B | check     | offsets                    |
| --- | ------------------------------- | ----: | --------- | -------------------------- |
| 16  | `16_select_noncommensurable`    |  8669 | OK        | ninguno                    |
| 17  | `17_cdh_target_k_ceiling`       |  4208 | OK        | 2 (internos)               |
| 18  | `18_diag_sentinel_signflip`     | 18991 | OK        | ninguno                    |
| 19  | `19_search_space_precommit`     | 17011 | OK        | ninguno (regenerado 17:02) |
| 20  | `20_stability_spaces_bootstrap` |     — | no existe | —                          |

**Los offsets de 17 son intra-patch, no drift del árbol.** En cada fichero el
hunk #1 desplaza al hunk #2:

- `main_workflow.py`: el hunk #1 inserta 10 líneas en L8358 (7 de comentario +
  `_CDH_TARGET_K_HI = 12` + 2 en blanco) → el hunk #2 cae en 8982, no en 8972.
- `tests/test_cdh_min_leaf_floor.py`: el hunk #1 cambia 1 import por 7 (+6) →
  el hunk #2 cae en 99, no en 93.

Anclas reales comprobadas: `main_workflow.py:8358` `return n_gate...`,
`main_workflow.py:8972` `# Ahora el sampler...`, test L10 `import pytest`,
test L93 línea en blanco. Las cabeceras `+8972`/`+93` del patch están stale;
el injerto es el que se pretende.

**18 sobre `3_diagnose_optuna_db.py` (M en git):** los 7 hunks casan byte a byte
con la base del patch (−28, −40, −83, −145, −169, −220, −264); los cambios
locales no tocan esas zonas.

**Consumidores del reporte de select (`knd` 6→12):** ninguno parsea por
posición de columna. Emisores: `src/reii/method_selection.py:89,324,419` y
`src/reii/main_workflow.py:6134` (solo `print`). `src/9_selection_audit.py` lee
snapshots/historial JSON y compara `why` por substring (`_reason_family`);
`3_diagnose_optuna_db.py` no lo lee; `scripts/playground/select_v4_arena.py`
tiene su propia copia de `_feasibility`; los tests solo usan substrings.
**El cambio de ancho no rompe a nadie**: `TestExplainAlineacion` espera cabecera
de 96 caracteres (2+76+2+14+2), coherente con las filas nuevas.

**`Optimizador.get_search_space` (L8954–9039) lee exactamente:**
`config_base.min_forms_uc`, `config_base.tsj`, `config_base.optimize_freeze_inert`
(getattr, True), `uc_config.window_size` (getattr, 3), `total_uces` (getattr, 0)
y `retrofitted_vectors` (getattr, None). El helper `_search_space` de los tests
define esos 4 atributos → sin `AttributeError`.

**Nombres de test:** 17 cita `tests/test_cdh_min_leaf_floor.py` (**existe**;
`test_cdh_min_floor.py` no existe). 16 → `test_select_v4.py` (existe). 18 crea
`test_diag_signflip.py` (aún ausente: correcto). 19 amplía `test_v5_space.py`
(existe). 20 crearía `test_stability_bootstrap.py` (ausente → condicional).

## a) Borrables YA

| Elemento                                                                            |             Tamaño | Justificación                                               |
| ----------------------------------------------------------------------------------- | -----------------: | ----------------------------------------------------------- |
| 10 × `__pycache__` (42 `.pyc`)                                                      |            1.82 MB | bytecode regenerable, fuera de git                          |
| `.pytest_cache/` (5 ficheros)                                                       |              24 KB | caché de pytest, se recrea                                  |
| `docs/patches/*.orig` / `*.rej`                                                     |         0 ficheros | no hay restos de rechazo                                    |
| `data/coref_chains_cache.json.lock`                                                 | 0 B (23-oct 20:19) | lock huérfano; solo sin corrida activa                      |
| `data/best_params.json.bak-20261027_210140`                                         |           10,938 B | byte-idéntico a `..._210920` (sha256 `9694833…`): uno sobra |
| `data/archive_logs/` (`output.log` 4.82 MB, `output_2.log` 70.5 KB, `p.txt` 9.7 KB) |            4.90 MB | logs de corrida Oct-24/-28, superados                       |
| `timing.tsv`                                                                        |              403 B | scratch de tiempos, sin consumidores                        |

Los `data/coref_seg_checkpoint.*.json` NO van aquí (ver b).

## b) Borrables DESPUÉS de la corrida v6

- `data/method_selection.json` y `data/method_selection_log.jsonl`: **hoy no
  existen** (0 ficheros) → el paso 6 del runbook es no-op. Solo habría que
  borrarlos si quedan restos pre-v6, porque `9_selection_audit --strict`
  compara replay vigente contra payload viejo y fallaría.
- `data/coref_seg_checkpoint.ctx3.json` (583,151 B) y `.ctx5.json` (549,014 B)
  (checkpoints del segmenter; borrarlos **antes** fuerza recómputo de Stanza en
  v6) y `data/checkpoints/` (vacío hoy).
- Motivo general: v6 corre con `--force` y reescribe `workflow_data.json`; hasta
  que termine, los artefactos previos son la única referencia válida.
- **Contradicción a resolver:** `data/archive_v5/best_params.v5_knee_rejected.json`
  (4,598 B, el knee artefacto) y `data/archive_v5/comparative_results.json`
  (27,000 B, idéntico a `data/comparative_results.json`, sha256 `967bcb3…`) están
  dentro de `archive_v5/`, que el bloque (c) declara protegido → **no borrar sin
  aprobación explícita del usuario**.

## c) Protegidos — no borrar nunca

- `data/diag/bench_uce{18,22,26,30}.log` (81,819 / 82,222 / 82,169 / 23,644 B)
  y `data/diag/uce_size_bench.json` (950 B): inputs del Bloque 1;
  `4_measure_uce_target_size.py --parse-only` los consume (glob
  `bench_uce*.log`, `--out` por defecto `uce_size_bench.json`).
- `data/archive_v4/` (3 ficheros, incl. `reii_optuna.db` 393,216 B),
  `data/archive_v5/` (2 ficheros) y `data/reii_optuna.db` (503,808 B): histórico.
- `.pcbase/` y `backup_patchset_*` (incluido el nuevo del paso 1).
- `8_probe_b_permissive.py` (15,722 B, raíz): está ahí a propósito;
  `src/reii/main_workflow.py:8371` lo cita **por nombre**.

## d) Dudosos — preguntar antes de borrar

Solo **2** de los 3 ficheros de 0 B existen (corrección a la premisa):

| Fichero                | Tamaño | mtime                   | Nota                                                                   |
| ---------------------- | -----: | ----------------------- | ---------------------------------------------------------------------- |
| `reii` (raíz)          |    0 B | 2026-10-28 16:44:39.560 | untracked                                                              |
| `reii.db` (raíz)       |    0 B | 2026-10-28 16:44:39.315 | untracked; inofensivo: `find_db` lo rechaza (sin tabla `trial_values`) |
| `reii_b_permissive.py` |      — | —                       | **no existe** en el árbol                                              |

Ambos son anteriores a los patches 16–18 (16:47+) y ningún patch los produce;
parecen restos de un comando mal escrito. **Recomendación: no borrar sin
confirmación** (`reii.db` ya está descartado por el código).

También dudoso: `data/diag.bak-20261027_210140.zip` vs `..._210920.zip`
(271,102 B cada uno, **sha256 distinto**) → no son duplicados; preguntar.

Opcional (scratch de raíz; pytest no los recoge porque `pyproject.toml:21` fija
`testpaths = ["tests"]`): `test_optuna_study_fingerprint_isolation.py` (4,074 B),
`test_sentinel_and_bottleneck.py` (3,036 B),
`test_svd_embedding_shape_bug.py` (4,867 B), `_ast_extract.py` (2,353 B),
`srt_analysis.json` (3.14 MB), `clean_manifest.json` (8,808 B).
