# Auditor de selección de método (`audit-v4`)

Estado: **propuesta cerrada; el diff no está aplicado** (hay una corrida en
curso). Los write scopes de este encargo son este documento,
`scripts/playground/audit_smoke.py` y `scripts/playground/audit_fixtures/*.json`.

Alcance: `src/9_selection_audit.py`. Se asume (no se rediseña) el contrato de
otros dos agentes:

- `docs/design_v5/BACKEND_C.md` — `method_c_backend in
{"hdbscan","dense_cdh","ward","kmeans_spherical"}`, `MethodMetrics.diagnostic_only`,
  y `hdbscan` como **diagnóstico** (no puede ganar).
- `docs/design_v5/GATE_NOISE.md` — gate de dominancia sobre la fracción del
  **corpus** `c = a·(1−ν)` con umbral `C0 = 0.70` (`select-v4`).

---

## 1. Inventario del estado actual

`src/9_selection_audit.py` tiene 122 líneas. Todo el cálculo vive en
`reii/method_selection.py`; el script es un _renderer_ + un `--json` a fichero.

### 1.1 Qué lee

| línea    | qué hace                                                                                                                                           |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| L37-38   | importa `DATA_DIR`, `audit_history`, `audit_text`, `read_history`                                                                                  |
| L40-41   | rutas por defecto: `data/method_selection_log.jsonl`, `data/method_selection.json`                                                                 |
| L65-67   | `--history`, `--snapshot`, `--json PATH`                                                                                                           |
| L75-83   | `read_history()`; si no hay registros -> `[X]` y `exit 2`                                                                                          |
| L85      | imprime `audit_text(records)` (tabla de estabilidad por corpus)                                                                                    |
| L87-100  | snapshot: `context.written_at`, `corpus_fingerprint`, `uce_target_size`, `n_uces`; `selection.winner`, `winner_k`, `winner_ari`, `winner_coverage` |
| L102-110 | vuelca `audit_history(records)` al fichero de `--json`                                                                                             |
| L112-118 | `unstable = bool(report["instability"])` -> mensaje y `exit 1` sólo con `--strict`                                                                 |

`audit_history` (`method_selection.py` L538-595) agrupa por
`corpus_fingerprint` (o `spec`, L524-527) y ordena por
`(uce_target_size, written_at)` (L529-535). Sólo mira 11 claves de `selection`
(L560-572).

### 1.2 Qué exige

Una sola cosa: que el ganador no cambie entre `uce_target_size` vecinos del
mismo corpus (L578-588). `--strict` (L68-72) convierte esa inestabilidad en
`exit 1`. Nada más.

### 1.3 Qué NO verifica (y por qué importa)

| invariante           | dónde debería mirar                                              | estado hoy                                                                                                                                                                                                                                                                                         |
| -------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 reproducibilidad   | `selection.all_metrics` + `select_method`                        | **no se llama nunca** `select_method` ni `_feasibility` (0 referencias en el script)                                                                                                                                                                                                               |
| 2 motivo de descarte | `selection.discarded[*].why` vs `_feasibility`                   | se imprime el texto (vía `audit_text`) y no se contrasta con nada                                                                                                                                                                                                                                  |
| 3 denominador        | `all_metrics[*].max_class_share / noise_frac / assigned_frac`    | se imprimen `max_class_share` y `usable`, nunca se comprueba aritmética                                                                                                                                                                                                                            |
| 4 backend            | `method_c_backend`, `diagnostic_only`                            | las claves **no existen** en el payload (grep vacío en `src/**`); el script no las lee                                                                                                                                                                                                             |
| 5 abstención         | `selection.winner` vs `abstained`                                | L98 imprime `winner` pero no compara con `abstained`. Peor: `audit_history` **normaliza** la contradicción en `method_selection.py` **L571** (`abstained = bool(sel.get("abstained")) or not sel.get("winner")`), así que un payload `abstained=false` + `winner=null` sale como abstención limpia |
| 6 no intersección    | `agreement_sym`, `triple_intersection`, `mode` legado            | no se escanea ninguna clave legada                                                                                                                                                                                                                                                                 |
| 7 trazabilidad       | `written_at`, `n_uces`, `corpus_fingerprint`, `best_params.json` | imprime `written_at` (L92) y nada más; no toca `best_params.json` ni hay hash de spec                                                                                                                                                                                                              |

Además, el umbral de dominancia está **cableado como texto**: el único rastro
es el `'>0.90'` de `Selection.explain_lines` (`method_selection.py` L248-252),
que este script no usa; y `--json` **exige una ruta** (L67), no hay modo
máquina a stdout.

### 1.4 Payload real observado (2026-10-28)

`data/method_selection.json`: `rule_version=select-v3`,
`context={corpus_fingerprint, uce_target_size, mode, n_uces, seed, written_at}`,
`selection.max_class_share_max=0.90` y filas de `all_metrics` con
`{method, k, ari, coverage, noise_frac, tiny_frac, perturbation, available,
assigned_frac, min_class_size, median_class_size, max_class_share, usable}`.
Historial: 1 línea. No hay `method_c_backend`, ni `backend`, ni
`diagnostic_only`, ni `n_assigned`, ni `spec_hash`.

Concretamente, la única corrida persistida descartó C por `K=2 fuera de
[3, 20]` y eligió B (`a=0.189`, `ν=0.000`); C tenía `a=0.964`, `ν=0.377`,
`c=0.600`.

---

## 2. Las siete invariantes

Cada una devuelve `ok` / `warn` (`[!]`) / `fail` (`[X]`). `--strict` sale con 1
si hay **algún** `fail`.

| #   | id                  | qué se comprueba                                                                                                                                                  | `[X]` cuando                                                                                                                                                |
| --- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `reproducibility`   | reconstruir `MethodMetrics` desde `all_metrics` y recalcular `select_method` con los umbrales persistidos                                                         | el ganador recalculado difiere y el payload declara la **misma** `rule_version` que el módulo. Si declara otra, es `[!]` (drift de regla) con el mismo diff |
| 2   | `discard_reason`    | para cada método: `_feasibility(...)` vigente vs `selection.discarded`; y si el motivo persistido es de dominancia, que el valor medido supere el umbral efectivo | dice "descartado por X" y la regla vigente lo considera viable                                                                                              |
| 3   | `denominator`       | `assigned_frac ≈ 1−noise_frac`; `n_assigned+n_noise = n_total`; `n_assigned ≈ n_total(1−ν)`; `a = s_max/n_assigned`; `c = a(1−ν) = s_max/n_total`                 | off-by-one (`n_total−1`), `a` calculado sobre el corpus, o identidad `c` rota (**`[!]`** si faltan las claves nuevas)                                       |
| 4   | `backend_coherente` | `method_c_backend` en el conjunto admitido; `diagnostic_only` coherente; ganador C con backend de diagnóstico                                                     | backend fuera del conjunto, o **ganador C con `hdbscan`/`diagnostic_only=true`**                                                                            |
| 5   | `abstention`        | `winner=None` + `abstained=true` es legítimo; se exige `reason`                                                                                                   | `abstained=true` con `winner`                                                                                                                               |
| 6   | `no_intersection`   | escaneo de `agreement_sym`, `triple_intersection`, `stable_cross*`, `mode ∈ {all, two_of_three, ...}`                                                             | nunca: es `[!]` (aviso de payload legado)                                                                                                                   |
| 7   | `traceability`      | `corpus_fingerprint` (o `spec`), `n_uces`, `written_at`, `spec_hash`, `best_params.json` (mtime)                                                                  | falta fingerprint/spec (la corrida no es atribuible)                                                                                                        |

Decisiones de diseño que evitan duplicar la regla:

- **No se reimplementa el gate.** El umbral efectivo se lee de
  `selection.max_class_share_max` y la viabilidad se delega en
  `reii.method_selection._feasibility`. La _escala_ del umbral (`corpus` = `c`
  o `assigned` = `a`) se deduce por **capacidad del módulo**
  (`hasattr(MethodMetrics, "dominance_corpus")`), no por una copia de la
  fórmula. Con `select-v3` da `assigned`; en cuanto se aplique el diff de
  `GATE_NOISE.md` §12.1 da `corpus`, sin tocar la auditoría.
- **No se cablea `0.90` ni `0.70`.** `_default_gate()` instancia `Selection` y
  lee su propio `max_class_share_max`.
- **Umbral por registro:** si el payload declara la misma `rule_version` que el
  módulo se usa el umbral persistido (replay exacto); si no, el umbral por
  defecto del módulo (la regla vigente manda).

---

## 3. Contrato de `last_selection` (payload persistido)

Lo escribe `DoubleClassifier.persist_method_selection`
(`main_workflow.py` L5951-5973) -> `reii.method_selection.persist_selection`
(L446-473), que además deja `self.db.data["method_selection"]` (L9959-9965).

### 3.1 Lo que ya se escribe (no se renombra nada)

| clave                                                                                       | tipo                            | origen                                   |
| ------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------- |
| `rule_version`                                                                              | `str`                           | `Selection.rule_version` (`"select-v3"`) |
| `context.corpus_fingerprint`                                                                | `str`                           | `_corpus_fingerprint()` (L507)           |
| `context.uce_target_size`                                                                   | `int`                           | `config.uce_target_size`                 |
| `context.mode`                                                                              | `str`                           | constante `"select"`                     |
| `context.n_uces`                                                                            | `int`                           | `n_total`                                |
| `context.seed`                                                                              | `int \| null`                   | `config.random_state`                    |
| `context.written_at`                                                                        | `str` ISO-8601 UTC              | `persist_selection` L459                 |
| `selection.winner`                                                                          | `str \| null`                   | `Selection`                              |
| `selection.winner_ari` / `winner_coverage` / `winner_usable_coverage`                       | `float \| null`                 | idem                                     |
| `selection.winner_k`                                                                        | `int`                           | idem                                     |
| `selection.runner_up*`, `delta_ari`, `delta_coverage`                                       | `str`/`float \| null`           | idem                                     |
| `selection.ambiguous`, `criteria_conflict`, `abstained`                                     | `bool`                          | idem                                     |
| `selection.reason`                                                                          | `str`                           | idem                                     |
| `selection.discarded`                                                                       | `list[{method: str, why: str}]` | idem                                     |
| `selection.margin`, `coverage_floor`, `min_class_share`, `noise_max`, `max_class_share_max` | `float`                         | umbrales efectivos de la corrida         |
| `selection.k_bounds`                                                                        | `[int, int]`                    | idem                                     |
| `selection.all_metrics[i].method`                                                           | `str` `"A"\|"B"\|"C"`           | `MethodMetrics.as_row()`                 |
| `selection.all_metrics[i].k`                                                                | `int`                           | idem                                     |
| `selection.all_metrics[i].ari` / `coverage` / `tiny_frac` / `noise_frac`                    | `float`                         | idem                                     |
| `selection.all_metrics[i].usable`                                                           | `float`                         | `coverage − tiny_frac`                   |
| `selection.all_metrics[i].perturbation`                                                     | `str`                           | `"A1<->A2"` etc.                         |
| `selection.all_metrics[i].available`                                                        | `bool`                          | idem                                     |
| `selection.all_metrics[i].assigned_frac`                                                    | `float \| null`                 | `1 − noise_frac` (L5686)                 |
| `selection.all_metrics[i].min_class_size` / `median_class_size`                             | `int` / `float`                 | L5687-5688                               |
| `selection.all_metrics[i].max_class_share`                                                  | `float`                         | `s_max / n_assigned` (L5665-5674)        |

### 3.2 Lo que `main_workflow.py` debe **empezar** a escribir

Todas son **aditivas**; ninguna clave existente cambia de nombre ni de tipo.

| clave                                       | tipo            | de dónde sale                                                                          | invariante que desbloquea |
| ------------------------------------------- | --------------- | -------------------------------------------------------------------------------------- | ------------------------- |
| `context.method_c_backend`                  | `str`           | `config.method_c_backend`                                                              | 4                         |
| `context.spec_hash`                         | `str`           | hash del spec efectivo (p.ej. `sha256` de `Config` serializado + `corpus_fingerprint`) | 7                         |
| `selection.method_c_backend`                | `str`           | copia para autodescripción del payload                                                 | 4                         |
| `selection.all_metrics[i].backend`          | `str`           | `res_emb1["backend"]` (sólo fila `C`)                                                  | 4                         |
| `selection.all_metrics[i].diagnostic_only`  | `bool`          | `MethodMetrics.diagnostic_only` (fila `C`)                                             | 4                         |
| `selection.all_metrics[i].n_total`          | `int`           | `n_total`                                                                              | 3                         |
| `selection.all_metrics[i].n_assigned`       | `int`           | `sum(partition_sizes)` (excluye `-1`)                                                  | 3                         |
| `selection.all_metrics[i].n_noise`          | `int`           | `n_total − n_assigned`                                                                 | 3                         |
| `selection.all_metrics[i].max_class_size`   | `int`           | `max(partition_sizes)` = `s_max`                                                       | 3                         |
| `selection.all_metrics[i].dominance_corpus` | `float`         | `MethodMetrics.dominance_corpus` (`as_row`, `GATE_NOISE.md` §12.1)                     | 2, 3                      |
| `selection.winner_dominance_corpus`         | `float \| null` | `Selection` (`GATE_NOISE.md` §12.1)                                                    | 2                         |

Con `backend`/`diagnostic_only`/`dominance_corpus` aplicados, la auditoría pasa
de `[!]` "no verificable" a `[ok]`/`[X]` real en las invariantes 2, 3 y 4.

---

## 4. Diff completo de `src/9_selection_audit.py`

Reemplazo del fichero (122 -> 998 líneas, 23 funciones; el grueso es el motor
de invariantes y el formateo de motivos, no lógica nueva). Cambios frente al actual:

- `--json` pasa a `nargs="?"`: `--json` solo (o `--json -`) emite **sólo** JSON
  a stdout; `--json RUTA` se comporta como hoy. `--strict`, `--history`,
  `--snapshot` intactos; se añaden `--best-params` y `--limit`.
- Se añade el bloque `INVARIANTES` (matriz + detalles) y el motor
  `_check_record`, que importa `_feasibility`/`select_method` del módulo.
- La sección heredada (`audit_text`) y el veredicto de estabilidad se mantienen
  literalmente; el `--json` conserva `groups`, `instability`, `ambiguous`,
  `abstained`, `n_records` y **añade** `records`, `snapshot`, `summary`,
  `audit_contract`, `rule_version`, `threshold_scale`, `effective_threshold`.
- `exit`: 2 sin historial; 1 con `--strict` si hay inestabilidad **o** `[X]`;
  0 en el resto.

El bloque siguiente es el fichero **completo**; `scripts/playground/audit_smoke.py`
lo extrae y ejecuta mientras el diff no esté aplicado (marcador
`<!-- audit-v4-source -->`).

<!-- audit-v4-source -->

```diff
--- a/src/9_selection_audit.py
+++ b/src/9_selection_audit.py
@@ -1,122 +1,470 @@
+#!/usr/bin/env python3
+"""
+Script 9 - Auditor de la seleccion de metodo (A / B / C) persistida.
+
+Pregunta 1 (heredada): la eleccion automatica de metodo, SE MANTIENE? Si con el
+mismo corpus el ganador cambia al mover uce_target_size, la eleccion no es
+identificable: es un artefacto de la segmentacion, no una propiedad de los
+datos.
+
+Pregunta 2 (contrato audit-v4): el payload persistido es COHERENTE con la regla
+vigente y con las metricas que lo justifican? Se verifican siete invariantes y
+cada una se marca [ok] / [!] / [X]:
+
+  1. reproducibilidad   recalcular select_method desde all_metrics da el ganador
+  2. motivo de descarte el motivo persistido coincide con _feasibility vigente
+  3. denominador        a sobre UCEs asignadas; n_assigned/n_total cuadran
+  4. backend            method_c_backend valido; hdbscan (diagnostico) no gana
+  5. abstencion         winner=None es legitimo; abstained=true con winner no
+  6. no interseccion    sin campos de estabilidad conjunta (agreement_sym, ...)
+  7. trazabilidad       fingerprint + n_uces + written_at + best_params.json
+
+La regla NO se reimplementa aqui: la viabilidad y el umbral de dominancia se
+importan de reii.method_selection (_feasibility / select_method). El umbral
+efectivo se lee del payload y su escala ('corpus' = c, 'assigned' = a) se
+deduce por capacidad del modulo (MethodMetrics.dominance_corpus), no por una
+copia de la formula. Ver docs/design_v5/AUDIT_V4.md.
+
+Fuente de datos (no recalcula particiones):
+  * data/method_selection_log.jsonl  -> una linea JSON por corrida.
+  * data/method_selection.json       -> snapshot de la ultima corrida.
+  * data/best_params.json            -> solo mtime/size, para trazar el origen.
+
+Uso (dentro del contenedor: necesita data/ montado, no la raiz):
+  docker compose --profile batch run --rm workflow python -u src/9_selection_audit.py
+  ... --history data/method_selection_log.jsonl --json data/diag/selection_audit.json
+  ... --json              # maquina: solo JSON a stdout (exit != 0 con --strict)
+  ... --strict            # sale con 1 si hay [X] o el ganador cambia por corpus
+
+Salida: tabla ASCII + veredicto. Sin emojis ni box-drawing (consola cp1252).
+"""
+
+from __future__ import annotations
+
+import argparse
+import dataclasses
+import json
+import math
+import sys
+from datetime import datetime, timezone
+from pathlib import Path
+from typing import Any, Dict, List, Optional, Sequence, Tuple
+
+from reii.config import BEST_PARAMS_PATH, DATA_DIR
+from reii.method_selection import (
+    RULE_VERSION,
+    MethodMetrics,
+    Selection,
+    _feasibility,
+    audit_history,
+    audit_text,
+    read_history,
+    select_method,
+)
+
+#: Marca de contrato: audit_smoke.py la usa para saber si el diff esta aplicado.
+AUDIT_CONTRACT = "audit-v4"
+
+DEFAULT_HISTORY = DATA_DIR / "method_selection_log.jsonl"
+DEFAULT_SNAPSHOT = DATA_DIR / "method_selection.json"
+DEFAULT_BEST_PARAMS = Path(BEST_PARAMS_PATH)
+
+#: Backends del paso denso de Method C (docs/design_v5/BACKEND_C.md, seccion 1).
+ALLOWED_BACKENDS = ("hdbscan", "dense_cdh", "ward", "kmeans_spherical")
+#: Backends sin -1 por contrato: n_noise == 0 y n_assigned == len(uce_ids).
+DENSE_BACKENDS = ("dense_cdh", "ward", "kmeans_spherical")
+
+#: Campos que la regla ya no usa. Si aparecen, el payload es de la epoca de las
+#: intersecciones y hay que avisar (no es un error de esta regla).
+LEGACY_KEYS = (
+    "agreement_sym",
+    "triple_intersection",
+    "n_intersection",
+    "intersection_size",
+    "votes",
+)
+LEGACY_MODES = ("all", "two_of_three", "wc_coref", "wc_emb", "coref_emb", "sim_emb")
+
+INVARIANT_ORDER = (
+    "reproducibility",
+    "discard_reason",
+    "denominator",
+    "backend_coherente",
+    "abstention",
+    "no_intersection",
+    "traceability",
+)
+INVARIANT_SHORT = {
+    "reproducibility": "repro",
+    "discard_reason": "motivo",
+    "denominator": "denom",
+    "backend_coherente": "back",
+    "abstention": "abst",
+    "no_intersection": "inter",
+    "traceability": "traza",
+}
+
+#: Tolerancia de la aritmetica del denominador (redondeo de serializacion).
+DENOM_TOL = 2e-3
+
+_RANK = {"ok": 0, "warn": 1, "fail": 2}
+_TAG = {"ok": "[ok]", "warn": "[!]", "fail": "[X]"}
+_CELL = {"ok": "ok", "warn": "!", "fail": "X"}
+
+_METRIC_FIELDS = tuple(f.name for f in dataclasses.fields(MethodMetrics))
+
+#: Escala del umbral de dominancia segun lo que EXPONE el modulo, no segun una
+#: copia de la formula: con select-v4 (GATE_NOISE.md) existe
+#: MethodMetrics.dominance_corpus y el umbral aplica sobre c (fraccion del
+#: corpus); con select-v3 el umbral persistido aplica sobre a (asignadas).
+_THRESHOLD_SCALE = "corpus" if hasattr(MethodMetrics, "dominance_corpus") else "assigned"
+
+
+def _utf8_stdout() -> None:
+    for stream in (sys.stdout, sys.stderr):
+        if hasattr(stream, "reconfigure"):
+            try:
+                stream.reconfigure(encoding="utf-8", errors="replace")
+            except (ValueError, OSError):
+                pass
+
+
+def _load_snapshot(path: Path) -> Optional[Dict[str, Any]]:
+    try:
+        return json.loads(path.read_text(encoding="utf-8"))
+    except (OSError, ValueError):
+        return None
+
+
+def _as_float(value: Any, default: float = float("nan")) -> float:
+    if value is None or isinstance(value, bool):
+        return default
+    try:
+        return float(value)
+    except (TypeError, ValueError):
+        return default
+
+
+def _as_int(value: Any) -> Optional[int]:
+    if value is None or isinstance(value, bool):
+        return None
+    try:
+        return int(value)
+    except (TypeError, ValueError):
+        return None
+
+
+def _finite(value: float) -> bool:
+    return isinstance(value, float) and math.isfinite(value)
+
+
+def _worst(statuses: Sequence[str]) -> str:
+    out = "ok"
+    for status in statuses:
+        if _RANK.get(status, 0) > _RANK[out]:
+            out = status
+    return out
+
+
+def _threshold_scale() -> str:
+    return _THRESHOLD_SCALE
+
+
+def _default_gate() -> float:
+    """Umbral de dominancia por defecto del modulo (no se cablea aqui)."""
+    blank = Selection(None, float("nan"), float("nan"), 0)
+    return float(blank.max_class_share_max)
+
+
+def _metric_from_row(row: Any) -> Optional[MethodMetrics]:
+    """Reconstruye MethodMetrics desde una fila persistida.
+
+    Filtra por los campos reales del dataclass, asi que es compatible con
+    payloads de select-v3 (sin diagnostic_only) y de select-v4 (con el).
+    """
+    if not isinstance(row, dict) or not row.get("method"):
+        return None
+    data = {k: v for k, v in row.items() if k in _METRIC_FIELDS}
+    data["k"] = _as_int(data.get("k")) or 0
+    data["ari"] = _as_float(data.get("ari"))
+    data["coverage"] = _as_float(data.get("coverage"), 0.0)
+    for key in ("noise_frac", "tiny_frac", "max_class_share", "median_class_size"):
+        data[key] = _as_float(data.get(key), 0.0)
+    if data.get("assigned_frac") is not None:
+        data["assigned_frac"] = _as_float(data.get("assigned_frac"))
+    if "min_class_size" in data:
+        data["min_class_size"] = _as_int(data.get("min_class_size")) or 0
+    if "diagnostic_only" in data:
+        data["diagnostic_only"] = bool(data["diagnostic_only"])
+    data["available"] = bool(data.get("available", True))
+    data["perturbation"] = str(data.get("perturbation") or "")
+    try:
+        return MethodMetrics(**data)
+    except TypeError:
+        return None
+
+
+def _thresholds(sel: Dict[str, Any]) -> Dict[str, Any]:
+    raw = sel.get("k_bounds") or (3, 20)
+    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
+        lo, hi = _as_int(raw[0]), _as_int(raw[1])
+    else:
+        lo, hi = 3, 20
+    return {
+        "margin": _as_float(sel.get("margin"), 0.02),
+        "coverage_floor": _as_float(sel.get("coverage_floor"), 0.15),
+        "k_bounds": (lo if lo is not None else 3, hi if hi is not None else 20),
+        "min_class_share": _as_float(sel.get("min_class_share"), 0.01),
+        "noise_max": _as_float(sel.get("noise_max"), 0.50),
+        "max_class_share_max": _as_float(sel.get("max_class_share_max")),
+    }
+
+
+def _dominance_value(metric: MethodMetrics) -> float:
+    """c = clase mayor / n_total, tal como lo expone el modulo (no se copia)."""
+    try:
+        return _as_float(getattr(metric, "dominance_corpus", None))
+    except (TypeError, ValueError):
+        return float("nan")
+
+
+def _replay(
+    sel: Dict[str, Any],
+    thresholds: Dict[str, Any],
+    same_rule: bool,
+) -> Optional[Selection]:
+    """Recalcula la regla con las metricas persistidas. None si no se puede."""
+    rows = sel.get("all_metrics") or []
+    metrics = [m for m in (_metric_from_row(r) for r in rows) if m is not None]
+    if not metrics:
+        return None
+    kwargs: Dict[str, Any] = {
+        "margin": thresholds["margin"],
+        "coverage_floor": thresholds["coverage_floor"],
+        "k_bounds": thresholds["k_bounds"],
+        "min_class_share": thresholds["min_class_share"],
+        "noise_max": thresholds["noise_max"],
+    }
+    gate = thresholds["max_class_share_max"]
+    # Solo se reutiliza el umbral persistido si el payload es de esta misma
+    # generacion de regla; si no, manda el default del modulo.
+    if same_rule and _finite(gate):
+        kwargs["max_class_share_max"] = gate
+    return select_method(metrics, **kwargs)
+
+
+def _reason_family(text: str) -> str:
+    """Familia del motivo, para comparar texto persistido con texto vigente."""
+    low = str(text or "").lower()
+    if "domin" in low:
+        return "dominance"
+    if "ruido" in low or "noise" in low:
+        return "noise"
+    if "fuera de [" in low or low.startswith("k="):
+        return "k_bounds"
+    if "coverage" in low:
+        return "coverage"
+    if "ari" in low:
+        return "ari"
+    if "particion" in low:
+        return "unavailable"
+    return "other"
+
+
+def _backend_of(
+    ctx: Dict[str, Any], sel: Dict[str, Any], row: Dict[str, Any]
+) -> Tuple[Optional[str], str]:
+    """Backend de Method C en orden de autoridad: fila > selection > context."""
+    candidates = (
+        ("metric.backend", row.get("backend")),
+        ("metric.method_c_backend", row.get("method_c_backend")),
+        ("selection.method_c_backend", sel.get("method_c_backend")),
+        ("context.method_c_backend", ctx.get("method_c_backend")),
+    )
+    for source, value in candidates:
+        if isinstance(value, str) and value.strip():
+            return value.strip().lower(), source
+    return None, ""
+
+
+def _best_params_info(path: Path) -> Dict[str, Any]:
+    try:
+        stat = path.stat()
+    except OSError:
+        return {"path": str(path), "exists": False, "mtime": None, "size": None}
+    return {
+        "path": str(path),
+        "exists": True,
+        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
+        "size": int(stat.st_size),
+    }
+
+
+def _iso(value: Any) -> Optional[datetime]:
+    try:
+        out = datetime.fromisoformat(str(value))
+    except (TypeError, ValueError):
+        return None
+    return out if out.tzinfo else out.replace(tzinfo=timezone.utc)
+
+
+def _check_record(
+    record: Dict[str, Any],
+    best_params: Path,
+    gate_default: float,
+) -> Dict[str, Any]:
+    """Verifica las siete invariantes de UNA corrida persistida."""
+    if not isinstance(record, dict):
+        record = {}
+    ctx = record.get("context")
+    ctx = ctx if isinstance(ctx, dict) else {}
+    sel = record.get("selection")
+    sel = sel if isinstance(sel, dict) else {}
+    rows = [r for r in (sel.get("all_metrics") or []) if isinstance(r, dict)]
+    metrics = [m for m in (_metric_from_row(r) for r in rows) if m is not None]
+    by_method = {m.method: m for m in metrics}
+    row_by_method: Dict[str, Dict[str, Any]] = {}
+    for row in rows:
+        if row.get("method"):
+            row_by_method.setdefault(str(row["method"]), row)
+    c_row = row_by_method.get("C") or {}
+
+    checks: List[Dict[str, str]] = []
+
+    def add(ident: str, status: str, detail: str = "") -> None:
+        checks.append({"id": ident, "status": status, "detail": detail})
+
+    stored_rule = str(sel.get("rule_version") or record.get("rule_version") or "")
+    same_rule = bool(stored_rule) and stored_rule == RULE_VERSION
+    thresholds = _thresholds(sel)
+    persisted_gate = thresholds["max_class_share_max"]
+    gate = persisted_gate if (same_rule and _finite(persisted_gate)) else float(gate_default)
+
+    winner_raw = sel.get("winner")
+    winner = None if winner_raw in (None, "", "null") else str(winner_raw)
+    abstained = bool(sel.get("abstained"))
+    reason = str(sel.get("reason") or "")
+
+    # ── 4. backend coherente ─────────────────────────────────────────────
+    backend, backend_source = _backend_of(ctx, sel, c_row)
+    diag_raw = c_row.get("diagnostic_only")
+    if isinstance(diag_raw, bool):
+        diagnostic: Optional[bool] = diag_raw
+    elif diag_raw is not None:
+        diagnostic = str(diag_raw).strip().lower() in ("1", "true", "yes", "si")
+    else:
+        diagnostic = None
+    effective_diag = diagnostic if diagnostic is not None else (
+        backend == "hdbscan" if backend else None
+    )
+    if backend is None:
+        add(
+            "backend_coherente",
+            "warn",
+            "sin method_c_backend/backend: coherencia no verificable (claves nuevas)",
+        )
+    elif backend not in ALLOWED_BACKENDS:
+        add(
+            "backend_coherente",
+            "fail",
+            f"backend '{backend}' no esta en {list(ALLOWED_BACKENDS)}",
+        )
+    else:
+        detail = (
+            f"backend={backend} ({backend_source}) "
+            f"diagnostic_only={str(bool(effective_diag)).lower()}"
+        )
+        dense_noise = _as_float(c_row.get("noise_frac"), 0.0)
+        if winner == "C" and effective_diag:
+            add(
+                "backend_coherente",
+                "fail",
+                detail
+                + ": ganador C con backend de diagnostico"
+                " (hdbscan no puede ganar; ver BACKEND_C.md)",
+            )
+        elif winner == "C" and backend in DENSE_BACKENDS:
+            add("backend_coherente", "ok", detail + ": ganador C con backend denso")
+        elif backend in DENSE_BACKENDS and dense_noise > 0.0:
+            add(
+                "backend_coherente",
+                "warn",
+                detail
+                + f": backend denso con noise_frac={dense_noise:.3f}"
+                " (contrato: n_noise=0)",
+            )
+        else:
+            add("backend_coherente", "ok", detail)
+
+    # ── 1. reproducibilidad ──────────────────────────────────────────────
+    replay = _replay(sel, thresholds, same_rule)
+    if replay is None:
+        add(
+            "reproducibility",
+            "warn",
+            "sin all_metrics reconstruibles: no se puede recalcular select_method",
+        )
+    elif same_rule:
+        if replay.winner == winner:
+            add(
+                "reproducibility",
+                "ok",
+                f"{winner or 'abstencion'} reproducido con las metricas persistidas",
+            )
+        else:
+            why = "; ".join(
+                f"{d.get('method')}: {d.get('why')}" for d in (replay.discarded or [])
+            )
+            add(
+                "reproducibility",
+                "fail",
+                f"persistido={winner or 'abstencion'}"
+                f" -> recalculado={replay.winner or 'abstencion'}"
+                + (f" (descartes recalculados: {why})" if why else ""),
+            )
+    else:
+        msg = (
+            f"payload {stored_rule or '?'} auditado con {RULE_VERSION}:"
+            f" persistido={winner or 'abstencion'}"
+            f" actual={replay.winner or 'abstencion'}"
+        )
+        if replay.winner != winner:
+            msg += "; la decision NO se sostiene con la regla vigente"
+        add("reproducibility", "warn", msg)
+
+    # ── 2. motivo de descarte consistente ────────────────────────────────
+    persisted_disc: Dict[str, str] = {}
+    for item in sel.get("discarded") or []:
+        if isinstance(item, dict) and item.get("method"):
+            persisted_disc[str(item["method"])] = str(item.get("why") or "")
+    issues: List[Tuple[str, str]] = []
+    if not metrics:
+        add("discard_reason", "warn", "sin all_metrics: motivos no verificables")
+    else:
+        for method in sorted(by_method):
+            why_now = _feasibility(
+                by_method[method],
+                thresholds["coverage_floor"],
+                thresholds["k_bounds"],
+                thresholds["noise_max"],
+                gate,
+            )
+            why_saved = persisted_disc.get(method)
+            if why_now and why_saved:
+                if _reason_family(why_now) != _reason_family(why_saved):
+                    issues.append(
+                        (
+                            "warn",
+                            f"{method}: motivo persistido '{why_saved}'"
+                            f" != vigente '{why_now}'",
+                        )
+                    )
+            elif why_now and not why_saved:
+                issues.append(
+                    (
+                        "warn",
+                        f"{method}: la regla vigente lo descartaria ({why_now});"
+                        " el payload no lo registro",
+                    )
+                )
+            elif why_saved and not why_now:
+                issues.append(
+                    (
+                        "fail",
+                        f"{method}: descartado en el payload ('{why_saved}')"
+                        " pero viable para la regla vigente",
+                    )
+                )
+        for method in sorted(persisted_disc):
+            if _reason_family(persisted_disc[method]) != "dominance":
+                continue
+            metric = by_method.get(method)
+            if metric is None:
+                issues.append(
+                    ("warn", f"{method}: descarte por dominancia sin metricas")
+                )
+                continue
+            scale = _threshold_scale()
+            value = (
+                _dominance_value(metric)
+                if scale == "corpus"
+                else _as_float(metric.max_class_share, 0.0)
+            )
+            label = "c" if scale == "corpus" else "a"
+            if not _finite(value):
+                issues.append(
+                    ("warn", f"{method}: descarte por dominancia sin {label} calculable")
+                )
+            elif _finite(gate) and value <= gate:
+                issues.append(
+                    (
+                        "fail",
+                        f"{method}: descartado por dominancia pero"
+                        f" {label}={value:.3f} <= umbral {gate:.2f}",
+                    )
+                )
+        add(
+            "discard_reason",
+            _worst([s for s, _ in issues]),
+            "; ".join(m for _, m in issues)
+            if issues
+            else "motivos persistidos consistentes con la regla vigente",
+        )
+
+    # ── 3. denominador ───────────────────────────────────────────────────
+    n_total = _as_int(ctx.get("n_uces"))
+    denom: List[Tuple[str, str]] = []
+    verifiable = False
+    for method in sorted(by_method):
+        metric = by_method[method]
+        row = row_by_method.get(method) or {}
+        a = _as_float(metric.max_class_share, 0.0)
+        nu = _as_float(metric.noise_frac, 0.0)
+        assigned_frac = _as_float(metric.assigned_frac)
+        if _finite(assigned_frac) and abs(assigned_frac - (1.0 - nu)) > DENOM_TOL:
+            denom.append(
+                (
+                    "fail",
+                    f"{method}: assigned_frac={assigned_frac:.3f}"
+                    f" != 1-noise_frac={1.0 - nu:.3f}",
+                )
+            )
+        n_assigned = _as_int(row.get("n_assigned"))
+        n_noise = _as_int(row.get("n_noise"))
+        s_max = _as_int(row.get("max_class_size"))
+        if s_max is None:
+            s_max = _as_int(row.get("s_max"))
+        if n_assigned is None and s_max is None:
+            continue
+        verifiable = True
+        if n_assigned is not None:
+            if n_assigned <= 0:
+                denom.append(("fail", f"{method}: n_assigned={n_assigned} <= 0"))
+            if n_total:
+                if n_noise is not None and n_assigned + n_noise != n_total:
+                    if n_assigned + n_noise == n_total - 1:
+                        denom.append(
+                            (
+                                "fail",
+                                f"{method}: n_assigned+n_noise={n_assigned + n_noise}"
+                                f" != n_total={n_total}"
+                                " (denominador resta un -1 de mas)",
+                            )
+                        )
+                    else:
+                        denom.append(
+                            (
+                                "fail",
+                                f"{method}: n_assigned+n_noise={n_assigned + n_noise}"
+                                f" != n_total={n_total}",
+                            )
+                        )
+                expected = int(round(n_total * (1.0 - nu)))
+                if abs(n_assigned - expected) > 1:
+                    denom.append(
+                        (
+                            "fail",
+                            f"{method}: n_assigned={n_assigned}"
+                            f" != round(n_total*(1-noise))={expected}",
+                        )
+                    )
+        if s_max is not None and n_assigned:
+            a_assigned = s_max / float(n_assigned)
+            if abs(a - a_assigned) > DENOM_TOL:
+                if n_total and abs(a - s_max / float(n_total)) <= DENOM_TOL:
+                    denom.append(
+                        (
+                            "fail",
+                            f"{method}: a={a:.4f}=s_max/n_total; el gate divide"
+                            f" por asignadas ({s_max}/{n_assigned}={a_assigned:.4f})",
+                        )
+                    )
+                else:
+                    denom.append(
+                        (
+                            "fail",
+                            f"{method}: a={a:.4f} != s_max/n_assigned={a_assigned:.4f}",
+                        )
+                    )
+            if n_total:
+                c_identity = a * (1.0 - nu)
+                c_true = s_max / float(n_total)
+                if abs(c_identity - c_true) > DENOM_TOL:
+                    denom.append(
+                        (
+                            "fail",
+                            f"{method}: c=a*(1-nu)={c_identity:.4f}"
+                            f" != s_max/n_total={c_true:.4f}"
+                            " (identidad rota: particiones distintas)",
+                        )
+                    )
+    if not verifiable:
+        add(
+            "denominator",
+            "warn",
+            "sin n_assigned/n_noise/max_class_size: aritmetica del denominador"
+            " no verificable (claves nuevas)",
+        )
+    else:
+        add(
+            "denominator",
+            _worst([s for s, _ in denom]),
+            "; ".join(m for _, m in denom)
+            if denom
+            else "a sobre UCEs asignadas; n_assigned+n_noise=n_total",
+        )
+
+    # ── 5. abstencion ────────────────────────────────────────────────────
+    if winner is None and abstained:
+        if reason:
+            add("abstention", "ok", f"abstencion legitima: {reason}")
+        else:
+            add("abstention", "warn", "abstencion sin motivo declarado: no auditable")
+    elif winner is None:
+        add(
+            "abstention",
+            "warn",
+            "winner=null sin abstained=true (payload legado o inconsistente)",
+        )
+    elif abstained:
+        add("abstention", "fail", f"abstained=true con winner={winner}: contradiccion")
+    else:
+        add("abstention", "ok", f"winner={winner}, abstained=false")
+
+    # ── 6. no interseccion ───────────────────────────────────────────────
+    legacy = {k for k in LEGACY_KEYS if k in sel or k in ctx or k in record}
+    for blob in [sel] + rows:
+        legacy |= {
+            str(k)
+            for k in blob
+            if str(k) in LEGACY_KEYS or str(k).startswith("stable_cross")
+        }
+    mode = str(ctx.get("mode") or record.get("mode") or "")
+    if legacy:
+        add(
+            "no_intersection",
+            "warn",
+            f"campos legados {sorted(legacy)}: el pipeline ya no exige"
+            " estabilidad conjunta (se elige UN metodo)",
+        )
+    elif mode in LEGACY_MODES:
+        add(
+            "no_intersection",
+            "warn",
+            f"mode='{mode}' es un modo de interseccion legado",
+        )
+    else:
+        add("no_intersection", "ok", "sin campos de interseccion; un solo metodo")
+
+    # ── 7. trazabilidad ──────────────────────────────────────────────────
+    fingerprint = ctx.get("corpus_fingerprint") or ctx.get("spec") or record.get("spec")
+    spec_hash = (
+        ctx.get("spec_hash")
+        or ctx.get("params_hash")
+        or ctx.get("best_params_sha256")
+    )
+    written_at = ctx.get("written_at")
+    trace: List[Tuple[str, str]] = []
+    if not fingerprint:
+        trace.append(
+            ("fail", "sin corpus_fingerprint ni spec: corrida no atribuible")
+        )
+    if not n_total:
+        trace.append(("warn", "sin n_uces en el context"))
+    if not written_at:
+        trace.append(("warn", "sin written_at en el context"))
+    if not spec_hash:
+        trace.append(("warn", "sin spec_hash: la corrida no se identifica por spec"))
+    bp_info = _best_params_info(best_params)
+    if not bp_info.get("exists"):
+        trace.append(("warn", f"{best_params.name} no existe: origen no trazable"))
+    else:
+        stamp = _iso(bp_info.get("mtime"))
+        when = _iso(written_at)
+        if stamp and when and stamp > when:
+            trace.append(
+                (
+                    "warn",
+                    f"{best_params.name} es MAS NUEVO que la seleccion"
+                    f" ({bp_info.get('mtime')} > {written_at}): decision obsoleta",
+                )
+            )
+    if trace:
+        add("traceability", _worst([s for s, _ in trace]), "; ".join(m for _, m in trace))
+    else:
+        add(
+            "traceability",
+            "ok",
+            f"fingerprint={fingerprint} n_uces={n_total}"
+            f" best_params={bp_info.get('mtime')}",
+        )
+
+    statuses = {c["id"]: c["status"] for c in checks}
+    if "fail" in statuses.values():
+        verdict = "violation"
+    elif winner is None and abstained:
+        verdict = "abstain"
+    else:
+        verdict = "select"
+
+    return {
+        "corpus": str(fingerprint or "?"),
+        "target": ctx.get("uce_target_size"),
+        "mode": ctx.get("mode"),
+        "written_at": written_at,
+        "rule_version": stored_rule or RULE_VERSION,
+        "winner": winner,
+        "backend": backend,
+        "diagnostic_only": effective_diag,
+        "threshold": {"value": gate, "scale": _threshold_scale()},
+        "dominance_corpus": _dominance_value(by_method["C"]) if "C" in by_method else None,
+        "checks": checks,
+        "verdict": verdict,
+    }
+
+
+def _snapshot_report(
+    path: Path, best_params: Path, gate_default: float
+) -> Dict[str, Any]:
+    snap = _load_snapshot(path)
+    info: Dict[str, Any] = {
+        "path": str(path),
+        "exists": snap is not None,
+        "context": {},
+        "selection": {},
+        "checks": [],
+        "verdict": None,
+        "backend": None,
+        "diagnostic_only": None,
+        "threshold": {"value": gate_default, "scale": _threshold_scale()},
+        "best_params": _best_params_info(best_params),
+    }
+    if not snap:
+        return info
+    ctx = snap.get("context") or {}
+    sel = snap.get("selection") or {}
+    info["context"] = dict(ctx) if isinstance(ctx, dict) else {}
+    if isinstance(sel, dict):
+        info["selection"] = {
+            k: sel.get(k)
+            for k in (
+                "winner",
+                "winner_k",
+                "winner_ari",
+                "winner_coverage",
+                "winner_usable_coverage",
+                "abstained",
+                "reason",
+                "rule_version",
+                "max_class_share_max",
+            )
+        }
+    checked = _check_record(snap, best_params, gate_default)
+    info["checks"] = checked["checks"]
+    info["verdict"] = checked["verdict"]
+    info["backend"] = checked["backend"]
+    info["diagnostic_only"] = checked["diagnostic_only"]
+    info["threshold"] = checked["threshold"]
+    return info
+
+
+def _build_report(
+    records: Sequence[Dict[str, Any]],
+    snapshot_path: Path,
+    best_params: Path,
+    gate_default: float,
+) -> Dict[str, Any]:
+    report = audit_history(records)
+    checked = [_check_record(rec, best_params, gate_default) for rec in records]
+    snap = _snapshot_report(snapshot_path, best_params, gate_default)
+    n_fail = sum(
+        1 for r in checked if any(c["status"] == "fail" for c in r["checks"])
+    )
+    n_warn = sum(
+        1 for r in checked if any(c["status"] == "warn" for c in r["checks"])
+    )
+    snap_fail = sum(1 for c in snap["checks"] if c.get("status") == "fail")
+    if n_fail or snap_fail:
+        verdict = "violation"
+    elif checked and all(r["verdict"] == "abstain" for r in checked):
+        verdict = "abstain"
+    else:
+        verdict = "select"
+    report.update(
+        {
+            "audit_contract": AUDIT_CONTRACT,
+            "rule_version": RULE_VERSION,
+            "threshold_scale": _threshold_scale(),
+            "effective_threshold": float(gate_default),
+            "records": checked,
+            "snapshot": snap,
+            "summary": {
+                "n_records": len(checked),
+                "n_fail": n_fail,
+                "n_warn": n_warn,
+                "snapshot_fail": snap_fail,
+                "verdict": verdict,
+            },
+        }
+    )
+    return report
+
+
+def _sanitize(value: Any) -> Any:
+    """NaN/Inf no son JSON valido: se emiten como null (igual que Selection)."""
+    if isinstance(value, float):
+        return value if math.isfinite(value) else None
+    if isinstance(value, dict):
+        return {str(k): _sanitize(v) for k, v in value.items()}
+    if isinstance(value, (list, tuple)):
+        return [_sanitize(v) for v in value]
+    return value
+
+
+def _render_invariants(report: Dict[str, Any], limit: int) -> List[str]:
+    gate = report["effective_threshold"]
+    records = report["records"]
+    lines = [
+        "",
+        f"INVARIANTES ({AUDIT_CONTRACT} | regla vigente {report['rule_version']}"
+        f" | umbral efectivo {gate:.2f} sobre '{report['threshold_scale']}')",
+        "  {:>3}  {:16}  {:>4}  {:3}  {:14}  {:5} {:6} {:5} {:4} {:4} {:5} {:5}  {}".format(
+            "#",
+            "corpus",
+            "tgt",
+            "met",
+            "backend",
+            INVARIANT_SHORT["reproducibility"],
+            INVARIANT_SHORT["discard_reason"],
+            INVARIANT_SHORT["denominator"],
+            INVARIANT_SHORT["backend_coherente"],
+            INVARIANT_SHORT["abstention"],
+            INVARIANT_SHORT["no_intersection"],
+            INVARIANT_SHORT["traceability"],
+            "veredicto",
+        ),
+    ]
+    for idx, rec in enumerate(records[:limit], 1):
+        status = {c["id"]: c["status"] for c in rec["checks"]}
+        lines.append(
+            "  {:>3}  {:16}  {:>4}  {:3}  {:14}  {:5} {:6} {:5} {:4} {:4} {:5} {:5}  {}".format(
+                idx,
+                str(rec["corpus"])[:16],
+                str(rec["target"] if rec["target"] is not None else "?"),
+                str(rec["winner"] or "-"),
+                str(rec["backend"] or "-")[:14],
+                _CELL.get(status.get("reproducibility", "ok"), "?"),
+                _CELL.get(status.get("discard_reason", "ok"), "?"),
+                _CELL.get(status.get("denominator", "ok"), "?"),
+                _CELL.get(status.get("backend_coherente", "ok"), "?"),
+                _CELL.get(status.get("abstention", "ok"), "?"),
+                _CELL.get(status.get("no_intersection", "ok"), "?"),
+                _CELL.get(status.get("traceability", "ok"), "?"),
+                rec["verdict"],
+            )
+        )
+    details: List[str] = []
+    for idx, rec in enumerate(records[:limit], 1):
+        for check in rec["checks"]:
+            if check["status"] != "ok":
+                details.append(
+                    f"  {_TAG[check['status']]} #{idx} {check['id']}: {check['detail']}"
+                )
+    if len(records) > limit:
+        details.append(
+            f"  [!] {len(records) - limit} corrida(s) no mostradas (--limit {limit})"
+        )
+    lines.extend(details)
+    summary = report["summary"]
+    tag = "[X]" if summary["n_fail"] else ("[ok]" if not summary["n_warn"] else "[!]")
+    lines.append(
+        f"  {tag} {summary['n_fail']} [X] / {summary['n_warn']} [!]"
+        f" en {summary['n_records']} corrida(s); veredicto={summary['verdict']}"
+    )
+    return lines
+
+
+def _print_snapshot(info: Dict[str, Any]) -> None:
+    if not info.get("exists"):
+        return
+    ctx = info.get("context") or {}
+    sel = info.get("selection") or {}
+    threshold = info.get("threshold") or {}
+    gate = _as_float(threshold.get("value"))
+    gate_text = f"{gate:.2f}" if _finite(gate) else "?"
+    bp = info.get("best_params") or {}
+    print("")
+    print(f"ULTIMA CORRIDA ({ctx.get('written_at') or '?'})")
+    print(
+        f"  corpus={ctx.get('corpus_fingerprint')}  "
+        f"target={ctx.get('uce_target_size')}  n_uces={ctx.get('n_uces')}"
+    )
+    print(
+        f"  ganador={sel.get('winner')}  K={sel.get('winner_k')}  "
+        f"ARI={sel.get('winner_ari')}  coverage={sel.get('winner_coverage')}"
+    )
+    print(
+        f"  regla={sel.get('rule_version')}  backend={info.get('backend') or '?'}  "
+        f"diagnostico={str(bool(info.get('diagnostic_only'))).lower()}  "
+        f"umbral={gate_text} sobre '{threshold.get('scale')}'"
+    )
+    print(
+        "  best_params.json: "
+        + (f"mtime={bp.get('mtime')}" if bp.get("exists") else "no existe")
+        + f"  spec_hash={ctx.get('spec_hash') or '?'}"
+        + f"  invariantes={info.get('verdict') or '?'}"
+    )
+
+
+def main(argv: Optional[Sequence[str]] = None) -> int:
+    _utf8_stdout()
+    ap = argparse.ArgumentParser(
+        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
+    )
+    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
+    ap.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
+    ap.add_argument("--best-params", default=str(DEFAULT_BEST_PARAMS))
+    ap.add_argument(
+        "--json",
+        nargs="?",
+        const="-",
+        default=None,
+        help="sin valor (o '-') -> JSON a stdout; con ruta -> escribe el informe",
+    )
+    ap.add_argument(
+        "--limit", type=int, default=20, help="maximo de corridas impresas"
+    )
+    ap.add_argument(
+        "--strict",
+        action="store_true",
+        help="exit 1 si hay [X] o el ganador cambia dentro de un mismo corpus",
+    )
+    args = ap.parse_args(list(argv) if argv is not None else None)
+
+    history = Path(args.history)
+    records = read_history(str(history))
+    if not records:
+        print(f"[X] sin historial en {history}")
+        print(
+            "    se llena al correr con --mode select "
+            "(una linea por corrida, append-only)"
+        )
+        return 2
+
+    report = _build_report(
+        records,
+        Path(args.snapshot),
+        Path(args.best_params),
+        _default_gate(),
+    )
+    machine = args.json == "-"
+    if not machine:
+        print(audit_text(records))
+        _print_snapshot(report["snapshot"])
+        print("\n".join(_render_invariants(report, max(1, int(args.limit)))))
+
+    if args.json:
+        text = json.dumps(
+            _sanitize(report), indent=2, ensure_ascii=True, sort_keys=True
+        )
+        if machine:
+            print(text)
+        else:
+            out = Path(args.json)
+            out.parent.mkdir(parents=True, exist_ok=True)
+            out.write_text(text, encoding="utf-8")
+            print(f"\nResultados: {out}")
+
+    unstable = bool(report["instability"])
+    summary = report["summary"]
+    broken = int(summary["n_fail"]) + int(summary["snapshot_fail"])
+    if not machine:
+        print("")
+        if unstable:
+            print("VEREDICTO: la eleccion NO es estable -> no se persiste como decision.")
+        else:
+            print("VEREDICTO: eleccion estable en todos los targets medidos.")
+        if broken:
+            print(
+                f"[X] {summary['n_fail']} corrida(s) con invariantes rotas"
+                f" ({summary['snapshot_fail']} en el snapshot)."
+            )
+    if args.strict and (unstable or broken):
+        return 1
+    return 0
+
+
+if __name__ == "__main__":
+    sys.exit(main())
```

### 4.1 Cambios de comportamiento, uno a uno

| aspecto              | antes                                              | después                                                                                                 |
| -------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `--json`             | obligatorio un PATH                                | `--json` / `--json -` = stdout; `--json PATH` = fichero (compatible)                                    |
| `--strict`           | `exit 1` si el ganador cambia                      | `exit 1` si el ganador cambia **o** hay `[X]`                                                           |
| tabla de estabilidad | `audit_text(records)`                              | igual, sin tocar (tests de `method_selection` intactos)                                                 |
| tabla nueva          | —                                                  | matriz de 7 invariantes + detalle de los no-`ok`                                                        |
| snapshot             | ganador/K/ARI/coverage                             | + `backend`, `diagnostico`, `umbral` y su escala, `spec_hash`, `best_params.mtime`                      |
| JSON                 | `groups/instability/ambiguous/abstained/n_records` | los mismos + `records/snapshot/summary/audit_contract/rule_version/threshold_scale/effective_threshold` |
| exit sin historial   | 2                                                  | 2 (igual)                                                                                               |

---

## 5. Playground

- `scripts/playground/audit_smoke.py` — corre los fixtures contra el auditor.
  Mientras el diff no esté aplicado, extrae el fichero del bloque
  `<!-- audit-v4-source -->` de este documento y ejecuta **ese** código: el
  smoke valida la propuesta, no una copia. Cuando el diff esté aplicado detecta
  `AUDIT_CONTRACT` en `src/9_selection_audit.py` y usa el fichero real.
- Inyecta `read_history`, `_load_snapshot` y `_best_params_info` en memoria:
  **no escribe ficheros** (ni temporales), no toca `data/`, no importa
  `reii.main_workflow`, corre en <1 s.
- `exit != 0` si un veredicto, un `exit_strict` o una invariante declarada en el
  fixture no coincide con lo observado.

| fixture                          | caso                                                      | expectativa                                                                                                                                                                                          |
| -------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_a_winner_ok.json`            | ganador A normal, `a` bajo, un solo método viable por ARI | `select`, 7×`ok`, `exit 0`                                                                                                                                                                           |
| `02_c_dense_dominance_0837.json` | C denso `a=0.8371`, `ν=0` (`{442,45,30,11}` sobre 528)    | bajo `select-v4`: C descartado por `c=0.8371 > 0.70`, replay da A -> `reproducibility [X]`, `exit 1`. Bajo `select-v3`: C gana (0.8371 < 0.90) -> `reproducibility [!]` por drift de regla, `exit 0` |
| `03_c_hdbscan_winner.json`       | ganador C con `backend=hdbscan`, `diagnostic_only=true`   | `backend_coherente [X]`, `exit 1` (independiente de la generación de regla)                                                                                                                          |
| `04_abstention_legit.json`       | `winner=null`, `abstained=true`, 3 métodos infactibles    | `abstain`, sin `[X]`, `exit 0`                                                                                                                                                                       |
| `05_payload_incomplete.json`     | registro sin `context` ni `all_metrics`                   | `traceability [X]`, `exit 1`, sin excepción                                                                                                                                                          |
| `06_legacy_intersection.json`    | `mode="all"` + `agreement_sym` en `selection`             | `no_intersection [!]`, veredicto `select`, `exit 0`                                                                                                                                                  |

### 5.1 El fixture (ii) y la regla nueva

`02_c_dense_dominance_0837.json` reproduce el caso del probe v4 **tal cual lo
formula `GATE_NOISE.md` §2**: `n_total=528`, partición cruda `{442,45,30,11}`,
`ν=0` -> `a = 442/528 = 0.837121`, `c = a·(1−ν) = 0.837121`.

- Con la regla nueva (`c > 0.70`) **debe descartarse**. El fixture lo refleja:
  declara `selection.rule_version="select-v4"` con `winner="C"` y
  `discarded=[]`, es decir un payload **imposible** bajo v4. La auditoría debe
  detectarlo como `reproducibility [X]` ("persistido=C -> recalculado=A
  (descartes recalculados: C: clase dominante 0.84 ... del corpus > max 0.70)")
  y `discard_reason [!]` ("la regla vigente lo descartaria").
- Mientras `src/reii/method_selection.py` siga en `select-v3`, el mismo fixture
  sólo puede producir drift de regla (`[!]`), no `[X]`: el módulo acepta
  `0.8371 < 0.90`. El smoke lo declara como `expect_stale_rule` y **no** falla
  por ello. En cuanto se aplique `GATE_NOISE.md` §12.1, el mismo fixture pasa a
  `[X]` sin editar el fixture.

---

## 6. Resultados reales del smoke

Comando:

```
python scripts/playground/audit_smoke.py
```

Salida (ejecución real, `python 3.13.13`, `src/reii/method_selection.py` en
`select-v3`, diff de este documento **no** aplicado; por eso el motor es el
propuesto, extraído del bloque de la §4):

<!-- smoke-results -->

```
SMOKE auditor audit-v4 | motor: propuesto (AUDIT_V4.md) | regla vigente: select-v3
  fixture                         esperado    observado   exit  estado  rama
  --------------------------------------------------------------------------
  01_a_winner_ok.json             select      select         0  [ok]    select-v3
  02_c_dense_dominance_0837.json  select      select         0  [ok]    select-v3 (fixture escrito para select-v4)
  03_c_hdbscan_winner.json        violation   violation      1  [ok]    select-v3
  04_abstention_legit.json        abstain     abstain        0  [ok]    select-v3
  05_payload_incomplete.json      violation   violation      1  [ok]    select-v3
  06_legacy_intersection.json     select      select         0  [ok]    select-v3

[ok] 6 fixtures: veredictos, exit y invariantes coinciden.
```

Lectura de la tabla (importante: **`exit 0` en el fixture 02 es correcto, no un
fallo del smoke**):

- 01, 04, 06 pasan con las 7 invariantes en el estado declarado. 05 demuestra que
  un payload mutilado no lanza excepción: `traceability [X]` -> `exit 1`.
- 03 falla (`[X]`) por la invariante 4 **hoy**: `winner=C` con
  `diagnostic_only=true`, independiente de la generación de regla. Ese `[X]` es
  real, no está pendiente de nada.
- 02 toma la rama `expect_stale_rule` porque el módulo está en `select-v3`: con
  `a=0.8371 < 0.90` C sigue ganando, así que lo único que puede reportarse es
  `[!]` de drift de regla (`payload select-v4 auditado con select-v3: persistido=C
actual=C`). **El smoke no falla porque el fixture declara las dos
  expectativas**; en cuanto se aplique `GATE_NOISE.md` §12.1 el mismo fixture
  entra por `expect` y exige `reproducibility [X]` + `exit 1` sin editarlo.

Comprobación adicional (modo humano, fixture 02, motor propuesto):

```
INVARIANTES (audit-v4 | regla vigente select-v3 | umbral efectivo 0.90 sobre 'assigned')
    #  corpus             tgt  met  backend         repro motivo denom back abst inter traza  veredicto
    1  fp-smoke-0002       30  C    dense_cdh       !     ok     ok    ok   ok   ok    ok     select
  [!] #1 reproducibility: payload select-v4 auditado con select-v3: persistido=C actual=C
  [!] 0 [X] / 1 [!] en 1 corrida(s); veredicto=select
VEREDICTO: eleccion estable en todos los targets medidos.
```

La columna `umbral efectivo ... sobre 'assigned'` es la prueba de que la escala
se deduce del módulo: con `GATE_NOISE.md` §12.1 aplicado pasa a
`0.70 sobre 'corpus'` sin tocar el auditor.

No se ha ejecutado la ruta `--json RUTA` (escribe fichero) ni el modo real sobre
`data/method_selection_log.jsonl`: ambas quedan prohibidas por el encargo
(corrida en curso). Sí se ha verificado `--json -` (máquina) y el modo humano.
<!-- /smoke-results -->

---

## 7. Riesgos de aplicar el diff con la corrida en curso

1. **`src/` está en uso.** El pipeline importa `reii.*`, no `src/9_selection_audit.py`
   (script de batch, no módulo). Editar el script **no** afecta a los procesos ya
   arrancados: Python ya cargó el bytecode. Riesgo residual: un `python -u
src/9_*.py` lanzado _durante_ tu edición (guardado no atómico). Mitigación:
   escribe el fichero de una vez (o `copy` + `move`), no lo dejes a medias.
2. **`__pycache__`**: no aplica, es un script `__main__` (no genera `.pyc` de sí
   mismo).
3. **No toca `data/`.** El script sólo lee. El `--json data/diag/...` de la
   docstring es del operador, no del auditor.
4. **El veredicto cambia de semántica.** Hoy `--strict` sólo mira estabilidad;
   después también `[X]`. Un pipeline que llame `9_selection_audit.py --strict`
   en CI puede empezar a fallar por invariantes (eso es lo pedido), incluso con
   el payload v3 actual: `[X]` en `traceability` no ocurrirá (hay fingerprint y
   `n_uces`), pero `denominator`/`backend` saldrán `[!]`, no `[X]`.
5. **`--json PATH` intacto, pero el contenido crece.** Consumidores que hagan
   `json.load(...)["instability"]` siguen funcionando (claves preservadas). Uno
   que cuente claves de primer nivel verá 7 nuevas.
6. **Dependencia de un símbolo privado**: `_feasibility`. Si otro agente lo
   renombra en el diff de `GATE_NOISE.md`, el import falla al arrancar (fallo
   ruidoso, no silencioso). Alternativa si molesta: exponer
   `feasibility = _feasibility` en `method_selection.py`.
7. **Orden de aplicación.** Si aplicas este diff **antes** que `GATE_NOISE.md`
   §12.1 y `BACKEND_C.md`, el auditor funciona pero reporta `assigned` +
   `[!]` en backend/denominador. Si lo aplicas **después**, la misma corrida
   pasa a `corpus` automáticamente (`_THRESHOLD_SCALE`). No hay que editar nada.

---

## 8. Invariantes imposibles con el payload actual

| invariante           | ¿posible hoy?                    | por qué                                                                                                                                                                                                                                | mínimo cambio                                                              |
| -------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1 reproducibilidad   | **sí**                           | `all_metrics` + umbrales persistidos están completos                                                                                                                                                                                   | —                                                                          |
| 2 motivo de descarte | **sí** (parcial)                 | el texto y las métricas están; la parte de dominancia sobre `c` sólo se evalúa si el módulo expone `dominance_corpus`                                                                                                                  | `GATE_NOISE.md` §12.1                                                      |
| 3 denominador        | **no del todo**                  | `a` y `ν` están, pero falta el **numerador** (`s_max`): un `a = s_max/n_total` mal calculado es indistinguible de `s_max/n_assigned`. `n_assigned` sólo se puede inferir como `n_total·(1−ν)`, lo que hace la comprobación tautológica | persistir `max_class_size`, `n_assigned`, `n_noise`, `n_total` por fila    |
| 4 backend            | **no**                           | las claves no existen en ningún sitio de `src/**` (grep vacío)                                                                                                                                                                         | `context.method_c_backend` + `all_metrics[C].backend` + `.diagnostic_only` |
| 5 abstención         | **sí**, leyendo el payload crudo | `audit_history` la normaliza (`method_selection.py` L571), por eso el chequeo no puede apoyarse en `audit_text`                                                                                                                        | —                                                                          |
| 6 no intersección    | **sí**                           | basta escanear claves                                                                                                                                                                                                                  | —                                                                          |
| 7 trazabilidad       | **parcial**                      | hay `fingerprint`, `n_uces`, `written_at` y `best_params.json` en disco; falta un hash del spec                                                                                                                                        | `context.spec_hash`                                                        |

### 8.1 Diff mínimo en `main_workflow.py` para desbloquear 3, 4 y 7

Requiere cuatro campos nuevos en `MethodMetrics` (aditivos, con default) — no
vienen en el diff de `GATE_NOISE.md`:

```diff
--- a/src/reii/method_selection.py
+++ b/src/reii/method_selection.py
@@ class MethodMetrics
     max_class_share: float = 0.0
+    #: Tamanos de la particion cruda (sin -1) y su denominador explicito.
+    #: Invariante 3 de la auditoria: a = max_class_size / n_assigned.
+    n_total: int = 0
+    n_assigned: int = 0
+    n_noise: int = 0
+    max_class_size: int = 0
+    #: Backend del paso denso de Method C y su rol (BACKEND_C.md).
+    backend: str = ""
+    diagnostic_only: bool = False
```

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -5665,10 +5665,16 @@
             def _max_share(_m: str) -> float:
                 sizes = [int(s) for s in (_raw_sizes.get(_m) or []) if int(s) > 0]
                 tot = sum(sizes)
                 return (max(sizes) / tot) if tot > 0 else 0.0
+
+            def _raw_stats(_m: str) -> Dict[str, int]:
+                sizes = [int(s) for s in (_raw_sizes.get(_m) or []) if int(s) > 0]
+                assigned = sum(sizes)
+                return {
+                    "n_total": int(n_total),
+                    "n_assigned": assigned,
+                    "n_noise": int(n_total) - assigned,
+                    "max_class_size": max(sizes) if sizes else 0,
+                }
@@ -5687,6 +5693,14 @@
                     median_class_size=float(_stats[_m][1]),
                     max_class_share=_max_share(_m),
+                    **_raw_stats(_m),
+                    backend=(
+                        str(getattr(self.config, "method_c_backend", "hdbscan"))
+                        if _m == "C"
+                        else ""
+                    ),
+                    diagnostic_only=bool(
+                        _m == "C"
+                        and getattr(self.config, "method_c_backend", "hdbscan")
+                        == "hdbscan"
+                        and not getattr(
+                            self.config, "method_c_hdbscan_selectable", False
+                        )
+                    ),
                 )
```

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -5964,6 +5972,14 @@
                 context={
                     "corpus_fingerprint": _corpus_fingerprint(),
                     "uce_target_size": target,
                     "mode": "select",
                     "n_uces": int(n_total),
                     "seed": getattr(self.config, "random_state", None),
+                    "method_c_backend": str(
+                        getattr(self.config, "method_c_backend", "hdbscan")
+                    ),
+                    "spec_hash": hashlib.sha256(
+                        json.dumps(
+                            dataclasses.asdict(self.config), sort_keys=True,
+                            default=str,
+                        ).encode("utf-8")
+                    ).hexdigest()[:16],
                 },
             )
```

`diagnostic_only` se deriva de `method_c_backend == "hdbscan"` **y** del
interruptor de escape `method_c_hdbscan_selectable`; si `BACKEND_C.md` acaba
persistiendo el flag en el backend, se sustituye por `res_emb1["diagnostic_only"]`
sin tocar la auditoría.
