# CONTRADICTIONS_OPTIONS — paper ↔ código: opciones de la más simple a la más visceral

Revisión de **solo lectura**: no se modificó `src/**`, `tests/**`, `docs/paper.md` ni ningún patch; el único fichero creado es este.
**Leyenda.** `min` ≤15 min sin docker · `h` 1–4 h (normalmente 1 corrida) · `d` ≥1 día o trabajo humano. **Cómputo** = docker + horas de Stanza.
**Cifras permitidas** (medidas; nada se suaviza con un número nuevo): `pos_only` 10.72 lemas/UCE / 0.395 · `tool_adverb_filtered` 12.24 / 0.451 · `pos_plus_propn` 13.16 / 0.485 (excluida: cuota 0.375 > 0.30) · ALCESTE 13.95 / 0.537 · `C_in` 0.8472 (nulo 0.8211, +0.0261, z 16.36, p 0.000) · `d` 0.9704 (nulo 0.8590) · `RMSD` 0.0300 (nulo 0.0364, **p 0.910**; aleatoria mismo perfil 0.0327).
**Otras:** 528 UCEs (154 con embedding 384 d) · piso ARI 0.72 (K=3) / 0.903 (K=8) · 442/528 = 0.837 · K ∈ {3,7,11,16} · suite 203 passed (+22 en los 2 ficheros tocados).
**Nota.** `paper.md` ya está parcialmente enmendado y las líneas de `PAPER_AMENDMENTS.md` han derivado; se citan las líneas **actuales** (verificadas con `grep -n`).

### 1. Húngaro "con corrección de K"/muchos-a-uno vs 1-a-1 real
**Paper** `paper.md:184-187` "Mapeo Húngaro Correjo … si K es diferente, usar mapping K1 ↔ K2 … no penalizar UCEs"; `:217` "Húngaro con correjo de K"; `:225` "Mapeo Húngaro corrigido" · **Código** `main_workflow.py:5408-5410` `os.environ.get("REII_STABILITY_MAPPING","hungarian")`; `:5440-5443` `linear_sum_assignment(-overlap)` ⇒ 1-1 sobre UCEs comunes. El muchos-a-uno **ya existe** (`quality/mapping.py:50-101`, `max_merge=2`, `min_share=0.5`) y está probado (`tests/test_quality_core.py:20-26`), pero solo por env var.
**Brecha** El default es 1-1 estricto; "corrección de K" no existe en ninguna ruta (el muchos-a-uno absorbe clases partidas, no corrige K), y el paper vende como mejora algo que el pipeline no ejecuta. Sobre "UCE estable": con 1-1 una clase de A partida en B solo dona estabilidad a su subclase emparejada; con muchos-a-uno el refinamiento se absorbe — son **dos definiciones** y hay que elegir una.

- **S1** Borrar "correjo": "Hungarian 1-1 sobre UCEs comunes; clases sin pareja se reportan" · `min` · riesgo: ninguno · cómputo: 0 · cierra: «no reproducible» · artefacto: ninguno
- **S2** S1 + declarar la ruta opt-in `REII_STABILITY_MAPPING=many_to_one` y sus tests · `min` · riesgo: bajo (flag que no toca las cifras) · cómputo: 0 · cierra: «el paper oculta una capacidad» · artefacto: ninguno
- **S3** Subir `many_to_one` a `Config` (default `hungarian`) y reportar ambos `n_stable` · `h` · riesgo: medio (2 cifras nuevas) · cómputo: 1 corrida `select` · cierra: «el efecto no está cuantificado» · artefacto: `method_selection` con ambos conteos
- **S4** Default `many_to_one` y re-correr todo · `d` · riesgo: **alto** (cambia cobertura, ganador y todos los artefactos) · cómputo: corrida completa + Stanza · cierra: «el default debería ser lo descrito» · artefacto: todo regenerado

**Rec.** S2 — hace el paper verdadero sin invalidar un artefacto; S4 solo si la afirmación se conserva como contribución.

### 2. FMI etiquetado como "Normalized Mutual Information"
**Paper** `paper.md:98` "| **FMI** (Normalized Mutual Information) | Dependencia mutua…"; `:248`/`:261` lo usan como "núcleo vs extensión" · **Código** `quality/metrics.py:579-582` `fmi = fowlkes_mallows_score`, `nmi = normalized_mutual_info_score`, `ari` aparte (`:569-574`).
**Brecha** La etiqueta nombra otra métrica y la comparación "núcleo vs extensión" ya no existe como nivel.

- **S1** "FMI (Fowlkes–Mallows); NMI se reporta aparte" · `min` · riesgo: ninguno · cómputo: 0 · cierra: «definición de métrica incorrecta» · artefacto: ninguno
- **S2** S1 + tabla FMI/NMI/ARI de las particiones reales · `h` · riesgo: bajo · cómputo: 1 `compare_modes` · cierra: «¿y el NMI?» · artefacto: tabla nueva
- **S3** Renombrar la sección a "índices de concordancia" y mantener FMI · `min` · riesgo: ninguno · cómputo: 0 · cierra: «cosmético» · artefacto: ninguno

**Rec.** S1 — una línea cierra la objeción; S2 solo si FMI va en una tabla de resultados.

### 3. Umbrales `silhouette>0.5`, `FMI>0.7`, `C_ext` 0.80/0.85, Kappa sin gold
**Paper** `paper.md:99` ">0.5: buen clustering"; `:102` ">0.8: acuerdo casi perfecto"; `:260` ">0.5"; `:261` ">0.7"; `:262` "0.80 | 0.85"; `:263` "C_in >0.7" · **Código** Ninguno existe en `src/`. Gates reales: `method_selection.py:265` `max_class_share_max=0.70`, `:266-268` `noise_max=0.50`, `k_bounds=(3,20)`; `quality/calibration.py:3-7` usa 0.80/0.70 para _tiers_ de acuerdo, no para `C_ext`. Silhouette medido: 0.042 / 0.077 / 0.239 (`BACKEND_C.md` §11).
**Brecha** Los cuatro son aspiraciones sin constante, sin nulo y sin artefacto, y silhouette > 0.5 está contradicho por lo medido.

- **S1** Declararlos heurísticas de ingeniería, sin pretensión de validez · `min` · riesgo: ninguno · cómputo: 0 · cierra: «¿de dónde sale 0.5?» · artefacto: ninguno
- **S2** Borrarlos y reportar valor medido + n + espacio + nulo · `min` · riesgo: ninguno · cómputo: 0 · cierra: «el umbral no lo cumple nadie» · artefacto: ninguno
- **S3** Sustituirlos por umbrales con nulo/permutación · `h` · riesgo: medio (nulos a fijar antes) · cómputo: 1–2 h CPU · cierra: «el umbral es arbitrario» · artefacto: tabla de nulos (extender `gate_calibration.py`)
- **S4** Gold humano + ≥2 codificadores + kappa · `d` · riesgo: alto (coste humano, acuerdo incierto) · cómputo: anotación · cierra: «no hay validez externa» · artefacto: diseño de anotación + kappa

**Rec.** S2 — cualquier umbral sin artefacto es indefendible; `C_in` conserva su nulo medido y no necesita umbral.

### 4. Benchmark BERTopic y tasas 65–85 % / 70–85 % sin artefacto
**Paper** `paper.md:71` "4.4. Benchmark … BERTopic vs. Modelo Propuesto" (solo índice, **sin cuerpo**); `:245` "65%–85%"; `:258` "65% | 70-85%" · **Código** `main_workflow.py:2244-2258` `BERTopicAnalyzer` es placeholder ("Devuelve la estructura vacía"); `:11027` "no implementado aún". `bertopic` no está en `requirements.txt` (sí `sentence-transformers` y `umap-learn`).
**Brecha** Se anuncia una comparación no ejecutada y dos rangos de retención sin artefacto; lo medido es 0.537 (ALCESTE) y 0.451 (REII).

- **S1** Borrar el 4.4 y sustituir 65–85/70–85 por 0.537 / 0.451 · `min` · riesgo: ninguno · cómputo: 0 · cierra: «afirmación sin dato» · artefacto: ninguno
- **S2** Declarar "no ejecutado / trabajo futuro" · `min` · riesgo: bajo (promesa abierta) · cómputo: 0 · cierra: «¿por qué se anuncia?» · artefacto: ninguno
- **S3** Ejecutar BERTopic sobre el corpus · `h` · riesgo: alto (dependencia nueva, hiperparámetros sin pre-registrar) · cómputo: 2–4 h + docker · cierra: «compara contra el estándar» · artefacto: tabla BERTopic
- **S4** Comparar contra los backends ya implementados (`c_backends_arena.py:607`: `dense_cdh`, `ward`, `kmeans_spherical`, `hdbscan`) · `h` · riesgo: bajo · cómputo: 1–2 h CPU, 0 Stanza · cierra: «no hay comparación de alternativas» · artefacto: arena existente

**Rec.** S4 + S1 — comparación real con lo que existe y borrado de lo que no; S3 solo si el revisor exige BERTopic por nombre.

### 5. "Cobertura por tokens ponderada" vs types/tokens sin ponderar (Efecto 5)
**Paper** `paper.md:96` "Cobertura por Tokens — % de términos del vocabulario … Términos En UCEs Estables / Términos Totales" (fórmula de **types**); `:212` "Sobre tokens, ponderada por frecuencia"; `:226` "pondera por la frecuencia"; `:316` "REII cuenta tokens ponderados" · **Código** `quality/metrics.py:46-78` `cov_types` (formas analizadas / distintas) y `cov_tokens` (ocurrencias / ocurrencias), **ambas sin peso por frecuencia**; `:118-153` `corpus_coverage` → `cov_types_total` / `cov_types_active` (solo types, con `tsj`).
**Brecha** La cobertura ponderada por frecuencia no existe: lo reportado a nivel de corpus es de types y `cov_tokens` es un recuento de ocurrencias.

- **S1** Reetiquetar la unidad y borrar "ponderada" · `min` · riesgo: ninguno · cómputo: 0 · cierra: «definición falsa» · artefacto: ninguno
- **S2** Recalcular con la definición declarada · `h` · riesgo: medio (métrica nueva sin pre-registro) · cómputo: ~1 h offline, 0 Stanza · cierra: «la métrica que describes no está» · artefacto: `coverage_units.json`
- **S3** Reportar types y tokens por separado y eliminar la ponderación · `min` · riesgo: ninguno · cómputo: 0 · cierra: «unidad de conteo no declarada» · artefacto: ninguno (ambos cortes ya existen)
- **S4** Borrar Efecto 5 y la fila de `:96` · `min` · riesgo: bajo (pierde un argumento) · cómputo: 0 · cierra: «razonas sobre una métrica inexistente» · artefacto: ninguno

**Rec.** S3 — las dos definiciones ya existen y están rotuladas; basta dejar de llamarlas "ponderadas".

### 6. Rango de UCE 15–100 vs calibración 26–30
**Paper** `paper.md:168-169` "UCE: 15–100 ocurrencias … flexible: 15–100, con suavizado"; `:211` "Flexible: 15–100 ocurrencias" · **Código** `main_workflow.py:835` `uce_target_size=26`; `:836` `min_forms_uc=[13,17]`; `data/uce_calibration.json` `target 30`, `desired 26.0`, `measured 26.857`, `cache_key "…:14-30"`.
**Brecha** El paper declara un rango 4–7× la media real que el código no usa; el rango calibrado es 14–30.

- **S1** "UCE objetivo 26–30 palabras (calibración 14–30); UC mínimo 13/17 formas" · `min` · riesgo: ninguno · cómputo: 0 · cierra: «parámetro no reproducible» · artefacto: ninguno
- **S2** Marcar 15–100 como `TODO(verificar)` · `min` · riesgo: medio (deja la cifra falsa a la vista) · cómputo: 0 · cierra: ninguna · artefacto: ninguno
- **S3** Recalibrar de verdad a 15–100 · `d` · riesgo: **alto** (invalida segmentación y todos los artefactos) · cómputo: corrida completa + Stanza · cierra: «el rango prometido no se cumple» · artefacto: todo regenerado

**Rec.** S1 — coincide con el artefacto de calibración y con `Config`.

### 7. `ε=0.1` en el paper vs `pseudocount=0.0` en código
**Paper** `paper.md:143` (tabla ALCESTE) y `:215` (tabla REII): "Pseudocuento ε | 0.1" · **Código** `main_workflow.py:870` `pseudocount=0.00`; forzado en `:9148-9149` y `:10336-10337`; classic lo declara `epsilon_status="not_implemented"` (`:11216`) y lo imprime (`:11228-11232`).
**Brecha** ε=0.1 se declara dos veces pero no tiene punto de consumo: el 0.0 se impone como invariante.

- **S1** "ε = 0.0; la especificación dice 0.1 y no está implementado" · `min` · riesgo: ninguno · cómputo: 0 · cierra: «parámetro que no existe» · artefacto: ninguno (classic ya lo avisa)
- **S2** Implementar ε en `_corte_optimo` · `d` · riesgo: **alto** (cambia particiones y obliga a revalidar) · cómputo: corrida completa + Stanza · cierra: «no eres fiel a la especificación» · artefacto: particiones nuevas
- **S3** Borrar la fila ε · `min` · riesgo: bajo (pierde una comparación) · cómputo: 0 · cierra: «declaras algo que no usas» · artefacto: ninguno

**Rec.** S1 — el modo classic ya emite el aviso; el paper debe decir lo mismo.

### 8. CTTEST como criterio de selección vs `use_ctest=False` + umbral 0.45
**Paper** `paper.md:142` (ALCESTE) y `:214` (REII): "Umbral de similitud | CTTEST … Mide qué tan asociados son los términos" · **Código** `main_workflow.py:862` `use_ctest=False`; `:863` `ctest_threshold=0.3`; el umbral de partición es `similarity_threshold=0.45` (`:1221`, consumido en `:4502-4506`); CTTEST solo filtra términos si se activa (`:6866-6869`).
**Brecha** CTTEST está apagado y no es el criterio de partición; 0.45 es un proxy de cohesión.

- **S1** "0,45 (proxy de cohesión); CTTEST no activo y no es el criterio de partición" · `min` · riesgo: ninguno · cómputo: 0 · cierra: «el método descrito no es el ejecutado» · artefacto: ninguno
- **S2** Activar `use_ctest=True` · `h` · riesgo: medio (cambia el conjunto de términos) · cómputo: 1 corrida + Stanza · cierra: «no usas CTTEST» · artefacto: términos regenerados
- **S3** Borrar la fila · `min` · riesgo: bajo · cómputo: 0 · cierra: «declaras un umbral que no usas» · artefacto: ninguno

**Rec.** S1 — activar CTTEST no lo convierte en criterio de partición, así que no cierra la objeción de fondo.

### 9. Cifras del "Efecto 2" (97 % / 30 % / ~90 %) sin artefacto
**Paper** `paper.md:295` (título "explica el 30% vs 97%"), `:299-300` (97 % y 30 %), `:304` ya tiene `TODO(verificar)`; `:299` atribuye ~90 % de UCEs clasificadas a ALCESTE · **Artefactos** Ninguno produce 97/30/90. Disponible: `cov_types_total`/`cov_types_active` sobre `data/workflow_data.json` (528 UCEs) y classic CR = 0.725 sobre 5.809 UCEs (`CLASSIC_MODE.md` §5).
**Brecha** Tres porcentajes sin denominador ni unidad; el 90 % de ALCESTE no es medible en este repo.

- **S1** Borrar los números y conservar el mecanismo Zipf/hapax · `min` · riesgo: bajo (pierde concreción) · cómputo: 0 · cierra: «cifra sin fuente» · artefacto: ninguno
- **S2** Mantenerlos como `TODO(verificar)` · `min` · riesgo: alto (es el estado actual, no publicable) · cómputo: 0 · cierra: ninguna · artefacto: ninguno
- **S3** Re-derivar **solo el lado REII** (types y tokens sobre 528 UCEs) y borrar 97/90 · `h` · riesgo: bajo · cómputo: ~1 h offline, 0 Stanza · cierra: «el 30 % no tiene artefacto» · artefacto: `coverage_units.json`
- **S4** Re-ejecutar la referencia ALCESTE sobre el mismo corpus · `d` · riesgo: alto (no hay script de baseline) · cómputo: corrida completa + Stanza · cierra: «compara contra un número heredado» · artefacto: baseline propio

**Rec.** S3 para el lado REII + S1 para 97 %/90 %; sin S4 el 90 % de ALCESTE no se sostiene.

### 10. "6 condiciones" de ALCESTE y el `p⁶≈0.53`
**Paper** `paper.md:144` "6 condiciones (o más)"; `:151` "Regla muy estricta (6 condiciones)"; pero `:289` dice "sobrevive a **2 corridas**". El `p⁶` **ya fue eliminado** (`:293` "no se calcula como una potencia"; `:332`) · **Código** `select-v4` elige **un** método (`method_selection.py`); la estabilidad es intra-método con dos pases tight/loose. No hay conjunción de seis condiciones en ninguna ruta.
**Brecha** El `p⁶` desapareció, pero "6 condiciones" sigue vivo y contradice a `:289` dentro del mismo documento, sin fuente.

- **S1** Unificar a "2 corridas" y borrar "6 condiciones" en `:144` y `:151` · `min` · riesgo: ninguno · cómputo: 0 · cierra: «contradicción interna» · artefacto: ninguno
- **S2** Dejar 6 con `TODO(verificar)` · `min` · riesgo: medio (mantiene la contradicción) · cómputo: 0 · cierra: ninguna · artefacto: ninguno
- **S3** Verificar la especificación de ALCESTE 2012 en la fuente original · `h` · riesgo: bajo · cómputo: 0 (lectura) · cierra: «citas mal al método de referencia» · artefacto: cita verificada

**Rec.** S1 — el propio documento ya dice "2 corridas"; mantener ambas versiones es indefendible.

### 11. RMSD presentado como validez (p = 0.910)
**Paper** `paper.md:69` ya corregido ("No distingue acuerdo de asignación … nunca como validez externa"); `:203` "RMSD: distancia de perfiles (no mide acuerdo)" · **Código** `quality/metrics.py:960-963` docstring: "un p-valor alto significa 'no hay evidencia de acuerdo' … reportar el RMSD como DESCRIPTOR".
**Brecha** El índice está corregido; el riesgo residual es que el valor viaje sin su nulo o que alguna tabla lo presente como acuerdo (0.0300 < nulo 0.0364 y < aleatoria 0.0327).

- **S1** "RMSD 0.0300 (nulo 0.0364, p = 0.910; aleatoria 0.0327) — sin evidencia de acuerdo" · `min` · riesgo: ninguno · cómputo: 0 · cierra: «presentas ruido como validez» · artefacto: ninguno
- **S2** Añadir ARI/NMI junto a cada RMSD · `h` · riesgo: bajo (`cross_partition_agreement` ya los da) · cómputo: 1 `compare_modes` · cierra: «¿hay acuerdo real?» · artefacto: tabla con ARI/NMI
- **S3** Eliminar RMSD del paper · `min` · riesgo: bajo (pierde un descriptor) · cómputo: 0 · cierra: «métrica sin poder discriminante» · artefacto: ninguno

**Rec.** S1 + S2 — convierte una debilidad en resultado nulo declarado y deja el acuerdo en ARI/NMI.

### 12. Gate de dominancia 0.70 sobre corpus: identificabilidad y negativa a 0.75
**Paper** `paper.md:234` "Gate de dominancia sobre el corpus (≤ 0,70) | Un corpus legítimamente dominado (p₁ = 0,72) se rechaza; el umbral no se sube a 0,75"; `:291` repite · **Código** `method_selection.py:265` `max_class_share_max=0.70`; `:267` `dominance_scale="corpus"`; `:494-500` `_feasibility` sobre `dominance_corpus_effective` (`:195-213`). Medido 442/528 = 0.837. Simulación (`GATE_NOISE.md` §0, reps=120): falso-válida 0.614 → **0.008** con `c > 0.70`.
**Brecha** El umbral es defendible, pero el paper no cita la tabla de error que responde a la objeción de identificabilidad bajo ruido.

- **S1** Mantener 0.70 y citar `GATE_NOISE.md` §0/§9 (tasas por régimen) · `min` · riesgo: ninguno · cómputo: 0 · cierra: «el umbral es arbitrario» · artefacto: ninguno (tabla ya existe)
- **S2** Curva de sensibilidad C0 ∈ {0.70, 0.75, 0.80} (`GATE_NOISE.md` §6.4; `scripts/playground/select_v4_sim.py`) · `h` · riesgo: bajo · cómputo: ~1 h CPU · cierra: «¿y si me equivoco en 0.05?» · artefacto: tabla de sensibilidad
- **S3** Subir a 0.75 · `min` · riesgo: **alto** (rompe la regla pre-registrada, sin análisis que lo sostenga) · cómputo: 0 · cierra: «rechazas corpus legítimos» · artefacto: ninguno

**Rec.** S1 (+S2 si piden robustez) — la respuesta a "no subas a 0.75" es cuantitativa y ya está medida.

### 13. Inestabilidad de K {3,7,11,16} con segmentación idéntica
**Paper** `paper.md:235` "K no es estable (K ∈ {3, 7, 11, 16} con segmentación idéntica)"; `:312` "segmentación idéntica (target 30, `lem_total = 5658`) … K ∈ {3, 7, 11, 16}" · **Artefactos** `data/method_selection_log.jsonl` tiene **una** fila (B k=11, A k=17, C k=2, n=276); los cuatro K no están todos persistidos (`PAPER_GAPS.md` §5).
**Brecha** Se afirma un hecho cuyo artefacto solo tiene una fila: no es auditable.

- **S1** Conservar la frase y citar el procedimiento de barrido · `min` · riesgo: alto (sigue sin artefacto) · cómputo: 0 · cierra: ninguna · artefacto: ninguno
- **S2** Regenerar el barrido de `uce_target_size` (18/22/26/30) y auditar con `src/9_selection_audit.py` · `h` · riesgo: bajo · cómputo: 1 `select` por target + Stanza · cierra: «no reproducible» · artefacto: log con ≥4 filas
- **S3** Rebajar a lo persistido: "K varía por método (A k=17, B k=11, C k=2)" · `min` · riesgo: bajo (pierde el argumento fuerte) · cómputo: 0 · cierra: «cifra sin artefacto» · artefacto: ninguno

**Rec.** S3 ya; S2 antes de enviar si se conserva la frase de los cuatro K.

### 14. Default de retención: `tool_adverb_filtered` (12.24 / 0.451) vs `pos_only` "opt-in"
**Paper** `paper.md:237` "(`pos_only`, **default**) … 10,72 … 0,395 … −3,23 lemas/UCE, −23,2 %"; `:277` "El default sigue siendo `pos_only`"; `:332` "12,24 y 0,451 … opt-in … 0,142 … −26,4 %"; encabezado `:279` "45,1 % … opt-in" · **Código** `retention_policy.py:38` `DEFAULT_RETENTION_POLICY = "tool_adverb_filtered"` (y `:16-23`); tres puntos de definición en `RETENTION_POLICY.md` §10.2. Brecha declarada: −1.71 lemas/UCE (−12.3 %) y −0.086 (−16.0 %).
**Brecha** El código ya resuelve la política conservadora sin argumentos; las tres frases afirman lo contrario.

- **S1** Aplicar `PAPER_AMENDMENTS.md` §«Pendiente por patch 14» **tal cual** (3 bloques redactados; cifras de `RETENTION_POLICY.md` §10.4) · `min` · riesgo: bajo · cómputo: 0 · cierra: «el paper describe otro default» · artefacto: ninguno
- **S2** Reformular como "configuración de las corridas" · `min` · riesgo: **alto** (el código resuelve el default sin argumentos ⇒ la frase sería falsa, lo advierte el apéndice) · cómputo: 0 · cierra: ninguna · artefacto: ninguno
- **S3** Tabla con ambas políticas (`pos_only` 10.72/0.395 · `tool_adverb_filtered` 12.24/0.451), default = conservadora · `min` · riesgo: bajo · cómputo: 0 · cierra: «ocultas la baseline» · artefacto: ninguno
- **S4** Revertir el default a `pos_only` · `min` · riesgo: **alto** (contradice el veredicto pre-registrado, `RETENTION_POLICY.md` §4.1, y el patch 14) · cómputo: 0 · cierra: «vuelves al default peor» · artefacto: ninguno

**Rec.** S3 (= S1 + una fila `pos_only`) — cierra la contradicción y evita que el revisor recalcule 10,72 y concluya que el default miente; S2 no es viable sin tocar código.

### 15. Bug abierto: `hdbscan_uce_min_cluster_frac(_loose)` fuera de la allowlist
**Paper** No lo menciona: no hay afirmación en el paper; es un bug del patchset.
**Código** `main_workflow.py:1075-1076` define ambas en `Config`; se consumen en `:5175-5183`; `best_params` las aplica en `:10362-10385`; el optimizador las mueve (`:9023-9033`). Pero `backend/config_store.py:62-66` lista `hdbscan_uce_min_cluster_size/_loose/epsilon/min_samples/metric` y **no** `_frac` ni `_frac_loose` (tampoco `_min_cluster_floor`, `:1077`, ni `retention_policy`/`content_pos`).
**Brecha** Asimetría: `set_config_many` (`:252`) descarta en silencio esas claves al guardar desde el dashboard; `seed_from_config` (`:201`) nunca las siembra; pero la aplicación (`:11297-11298`) filtra por `__dataclass_fields__`, así que una fila presente en la BD **sí** se aplicaría. El knob se tunea, se consume y no se puede escribir por la vía sancionada.

- **S1** Añadir los dos nombres a `"clasificacion"` + test (2 líneas) · `min` (~30 min) · riesgo: bajo (ver abajo) · cómputo: 0 · cierra: «configuración inconsistente» · artefacto: test nuevo
- **S2** S1 + test de ida y vuelta de **todas** las claves de `Config` contra `ALL_CONFIG_KEYS` · `h` · riesgo: bajo · cómputo: 0 · cierra: «hay más claves huérfanas» · artefacto: test de allowlist
- **S3** S1 + que `set_config_many` rechace claves desconocidas en vez de filtrarlas · `min` · riesgo: medio (puede romper llamadas del dashboard que envían extras) · cómputo: 0 · cierra: «el dashboard pierde valores sin avisar» · artefacto: ninguno

**Rec.** S1 (2 líneas + test) y S2 antes de publicar si el dashboard se usa para las corridas finales. **Riesgo de S1:** al sembrarse, la BD pasa a ser autoritativa para esas claves vía `get_all()` + `dataclasses.replace` (`:11301`), y una BD vieja archivada podría pisar el default en silencio; mitigación: los defaults sembrados (0.08 / 0.05) coinciden con los actuales, así que en BD nueva no hay cambio de comportamiento.

## Tabla de decisión

`¿Bloquea publicar?` = estado **de hoy**, sin aplicar la opción.

| #   | Problema                                          | Opción recomendada                    | Esfuerzo | ¿Bloquea publicar? |
| --- | ------------------------------------------------- | ------------------------------------- | :------: | :----------------: |
| 1   | Húngaro "correjo de K" vs 1-1                     | S2 (declarar 1-1 + ruta opt-in)       |   min    |       **sí**       |
| 2   | FMI etiquetado como NMI                           | S1 (corregir etiqueta)                |   min    |         no         |
| 3   | Umbrales 0.5/0.7/0.80/0.85 sin nulo ni gold       | S2 (medido + nulo, sin umbral)        |   min    |       **sí**       |
| 4   | BERTopic y tasas 65–85/70–85                      | S4 + S1 (arena real, borrar el resto) |    h     |       **sí**       |
| 5   | Cobertura "ponderada por frecuencia"              | S3 (types y tokens por separado)      |   min    |         no         |
| 6   | UCE 15–100 vs 26–30                               | S1 (26–30; calibración 14–30)         |   min    |         no         |
| 7   | ε = 0.1 vs 0.0                                    | S1 (declarar 0.0, no implementado)    |   min    |         no         |
| 8   | CTTEST vs `use_ctest=False` + 0.45                | S1 (0,45 proxy; CTTEST off)           |   min    |         no         |
| 9   | Efecto 2 (97/30/~90)                              | S3 + S1                               |    h     |       **sí**       |
| 10  | "6 condiciones" / p⁶                              | S1 (unificar a 2 corridas)            |   min    |       **sí**       |
| 11  | RMSD como validez (p 0.910)                       | S1 + S2 (valor + nulo + ARI/NMI)      |   min    |         no         |
| 12  | Gate 0.70 y negativa a 0.75                       | S1 (citar tasas de error)             |   min    |         no         |
| 13  | K ∈ {3,7,11,16} sin artefacto completo            | S3 (lo persistido); S2 para conservar |   min    |         no         |
| 14  | Default de retención vs paper                     | S3 (tabla con ambas políticas)        |   min    |       **sí**       |
| 15  | `hdbscan_uce_min_cluster_frac` fuera de allowlist | S1 (2 líneas + test); S2 después      |   min    |         no         |
