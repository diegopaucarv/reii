# Reporte REII — optimizador y segmentador
Generado: 2026-10-29T20:41:08
Python 3.13.13 · numpy 2.5.4 · optuna 3.6.2


## 3. Benchmark (simulación mock, TPE real de Optuna)

Paisajes sintéticos por escenario: **8**. Δ = utilidad relativa al óptimo conocido menos la de la estrategia de referencia (S1 = solo fix de lookup). IC95 aprox. = 1.96·sd/√n.

### Escenario P=30 (coste pase coref por ctx), rho=0.7 (correlación etapa 1↔2)

**A. Mismo nº de trials (90)**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.849 | +0.000 ± 0.000 | 90 | 4052 | 42.6 |
| S1 fix lookup | 0.849 | +0.000 ± 0.000 | 90 | 362 | 42.6 |
| S2 +factible exacto (pares) | 0.853 | +0.004 ± 0.132 | 90 | 416 | 53.8 |
| S2r +rechazo (FAIL) | 0.878 | +0.029 ± 0.091 | 90 | 416 | 55.1 |
| S3 +presupuesto/warm/gate | 0.885 | +0.035 ± 0.117 | 90 | 377 | 42.0 |

**B. Mismo tiempo que el 90-trials actual (con bug)**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.869 | -0.130 ± 0.076 | 215 | 4086 | 29.1 |
| S1 fix lookup | 0.998 | +0.000 ± 0.000 | 300 | 685 | 91.4 |
| S2 +factible exacto (pares) | 0.981 | -0.017 ± 0.066 | 300 | 860 | 148.0 |
| S2r +rechazo (FAIL) | 0.984 | -0.014 ± 0.098 | 300 | 860 | 145.6 |
| S3 +presupuesto/warm/gate | 1.028 | +0.029 ± 0.091 | 300 | 841 | 161.8 |

**C. Mismo tiempo que el 90-trials con el fix**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.691 | -0.118 ± 0.125 | 146 | 399 | 2.1 |
| S1 fix lookup | 0.808 | +0.000 ± 0.000 | 168 | 375 | 10.6 |
| S2 +factible exacto (pares) | 0.842 | +0.034 ± 0.045 | 129 | 369 | 10.5 |
| S2r +rechazo (FAIL) | 0.840 | +0.031 ± 0.062 | 130 | 364 | 10.1 |
| S3 +presupuesto/warm/gate | 0.824 | +0.016 ± 0.095 | 135 | 364 | 14.1 |

### Escenario P=100 (coste pase coref por ctx), rho=0.7 (correlación etapa 1↔2)

**A. Mismo nº de trials (90)**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.849 | +0.000 ± 0.000 | 90 | 13129 | 42.6 |
| S1 fix lookup | 0.849 | +0.000 ± 0.000 | 90 | 829 | 42.6 |
| S2 +factible exacto (pares) | 0.853 | +0.004 ± 0.132 | 90 | 883 | 53.8 |
| S2r +rechazo (FAIL) | 0.878 | +0.029 ± 0.091 | 90 | 883 | 55.1 |
| S3 +presupuesto/warm/gate | 0.885 | +0.035 ± 0.117 | 90 | 844 | 42.0 |

**B. Mismo tiempo que el 90-trials actual (con bug)**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.874 | -0.125 ± 0.071 | 216 | 13215 | 30.0 |
| S1 fix lookup | 0.998 | +0.000 ± 0.000 | 300 | 1152 | 91.4 |
| S2 +factible exacto (pares) | 0.981 | -0.017 ± 0.066 | 300 | 1327 | 148.0 |
| S2r +rechazo (FAIL) | 0.984 | -0.014 ± 0.098 | 300 | 1327 | 145.6 |
| S3 +presupuesto/warm/gate | 1.028 | +0.029 ± 0.091 | 300 | 1308 | 161.8 |

**C. Mismo tiempo que el 90-trials con el fix**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.638 | -0.161 ± 0.118 | 159 | 938 | 1.6 |
| S1 fix lookup | 0.799 | +0.000 ± 0.000 | 179 | 854 | 10.0 |
| S2 +factible exacto (pares) | 0.861 | +0.061 ± 0.089 | 137 | 837 | 7.8 |
| S2r +rechazo (FAIL) | 0.846 | +0.046 ± 0.070 | 138 | 840 | 8.8 |
| S3 +presupuesto/warm/gate | 0.824 | +0.025 ± 0.120 | 135 | 831 | 14.1 |

### Escenario P=30 (coste pase coref por ctx), rho=0.3 (correlación etapa 1↔2)

**A. Mismo nº de trials (90)**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.869 | +0.000 ± 0.000 | 90 | 3861 | 43.5 |
| S1 fix lookup | 0.869 | +0.000 ± 0.000 | 90 | 362 | 43.5 |
| S2 +factible exacto (pares) | 0.844 | -0.025 ± 0.143 | 90 | 416 | 52.0 |
| S2r +rechazo (FAIL) | 0.896 | +0.027 ± 0.112 | 90 | 416 | 56.2 |
| S3 +presupuesto/warm/gate | 0.824 | -0.045 ± 0.102 | 90 | 376 | 41.4 |

**B. Mismo tiempo que el 90-trials actual (con bug)**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.694 | -0.256 ± 0.093 | 203 | 3878 | 27.9 |
| S1 fix lookup | 0.950 | +0.000 ± 0.000 | 300 | 714 | 100.1 |
| S2 +factible exacto (pares) | 0.984 | +0.034 ± 0.097 | 300 | 860 | 149.9 |
| S2r +rechazo (FAIL) | 0.972 | +0.022 ± 0.084 | 300 | 860 | 147.2 |
| S3 +presupuesto/warm/gate | 0.987 | +0.037 ± 0.069 | 300 | 842 | 162.9 |

**C. Mismo tiempo que el 90-trials con el fix**

| Estrategia | utilidad/óptimo | Δ vs S1 (IC95) | trials | coste | válidos etapa 2 |
|---|---|---|---|---|---|
| B0 actual (bug cache) | 0.609 | -0.060 ± 0.057 | 144 | 389 | 2.5 |
| S1 fix lookup | 0.669 | +0.000 ± 0.000 | 160 | 365 | 9.0 |
| S2 +factible exacto (pares) | 0.728 | +0.059 ± 0.072 | 127 | 363 | 9.9 |
| S2r +rechazo (FAIL) | 0.685 | +0.016 ± 0.107 | 132 | 370 | 12.9 |
| S3 +presupuesto/warm/gate | 0.808 | +0.139 ± 0.136 | 136 | 363 | 15.1 |


## 4. Resumen

- Tests: omitidos (--skip-tests)
- Benchmark: Δ vs S1 con su IC95; si el IC cruza 0 no hay evidencia de diferencia.
