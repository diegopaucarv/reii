# Gate de no dominancia dependiente del ruido (`select-v4`)

Estado: propuesta cerrada, pendiente de aplicar el diff de la seccion 12.
Autor: analisis estadistico sobre el codigo de `select-v3` y simulacion
autocontenida (`scripts/playground/gate_calibration.py`).
Alcance: `src/reii/method_selection.py`, `src/reii/main_workflow.py`.

---

## 0. Resumen

El gate vigente (`max_class_share > 0.90`) no mide degeneracion: mide
degeneracion **diluida por el ruido**, porque divide por las UCEs asignadas.
Su umbral implicito sobre el corpus es `0.90 * (1 - nu)`, que va de 0.900
(sin ruido) a 0.315 (65 % de ruido). De ahi la asimetria del probe v4: dos
particiones con identico `c = 0.837` reciben veredictos opuestos.

La regla propuesta fija el umbral en la escala donde el problema esta
formulado:

```
degenerada(m)  <=>  c_m = max_class_share_m * (1 - noise_frac_m) > C0
```

con **un solo parametro libre, `C0 = 0.70`**. Leida sobre la variable que hoy
imprime el codigo equivale a `max_class_share > 0.70 / (1 - nu)` (creciente en
`nu`). No anade dependencia del ruido: la **elimina**, porque `c` ya la
contiene.

Resultado de la simulacion (reps=120 por celda, 528 UCEs):

| forma                    | falso-valida (global) | falso-degenerada (global) | peor celda          |
| ------------------------ | --------------------- | ------------------------- | ------------------- |
| actual `a > 0.90`        | 0.614                 | 0.001                     | 0.88 (nu=0)         |
| **propuesta `c > 0.70`** | **0.008**             | **0.000**                 | **0.051 (nu=0.20)** |

---

## 1. Notacion e identidad exacta

Para un metodo `m` en una corrida con `n` UCEs totales:

| simbolo | definicion                                         | de donde sale                                      |
| ------- | -------------------------------------------------- | -------------------------------------------------- |
| `s_max` | tamano de la clase mayor de la particion **cruda** | `_partition_sizes_from_labels` excluye `label < 0` |
| `m`     | UCEs asignadas (`label >= 0`)                      | suma de esos tamanos                               |
| `nu`    | `noise_frac = (n - m) / n`                         | `_noise_frac` divide por `n_total`                 |
| `a`     | `max_class_share = s_max / m`                      | `_max_share` en `main_workflow.py`                 |
| `c`     | `s_max / n` (fraccion del corpus)                  | **no existe hoy en el codigo**                     |

Entonces, para `m > 0`:

```
c = s_max / n = (s_max / m) * (m / n) = a * (1 - nu)
```

Es una **identidad**, no una aproximacion, y depende de dos decisiones que ya
estan tomadas en el codigo:

- `_partition_sizes_from_labels` (L542) cuenta los tamanos **excluyendo**
  `label < 0`;
- `_noise_frac` (L6190) divide las UCEs sin asignar por `n_total`.

Si alguna de las dos cambiara (p.ej. si `sizes` se calculara sobre otra
particion que la del `noise_frac`), la identidad pasaria a ser aproximada.
`assigned_frac = 1 - noise_frac` (L5686) ya se construye asi, de modo que el
codigo es autoconsistente por construccion.

Consecuencia operativa: **`c` es calculable hoy sin tocar `main_workflow.py`**,
basta con multiplicar los dos campos que ya viajan en `MethodMetrics`.

---

## 2. Que mide realmente el gate vigente

Sustituyendo la identidad en la regla actual `a > 0.90`:

```
a > 0.90   <=>   c / (1 - nu) > 0.90   <=>   c > 0.90 * (1 - nu)      (nu < 1)
```

Es decir, el gate de `select-v3` **ya era dependiente del ruido**, pero en la
direccion equivocada y sin declararlo:

| `nu`  | umbral implicito sobre `c` |
| ----- | -------------------------- |
| 0.000 | 0.900                      |
| 0.050 | 0.855                      |
| 0.100 | 0.810                      |
| 0.119 | 0.793                      |
| 0.200 | 0.720                      |
| 0.350 | 0.585                      |
| 0.500 | 0.450                      |
| 0.650 | 0.315                      |

El umbral efectivo se mueve un factor 3 solo por el ruido. Eso explica
exactamente el hallazgo del probe v4 y no requiere ninguna hipotesis nueva:

| caso                             | `a`   | `nu`  | `c`   | umbral implicito | veredicto `select-v3` |
| -------------------------------- | ----- | ----- | ----- | ---------------- | --------------------- |
| denso sin ruido `{442,45,30,11}` | 0.837 | 0.000 | 0.837 | 0.900            | **acepta** (mal)      |
| HDBSCAN `{442,23}` + 63 ruido    | 0.951 | 0.119 | 0.837 | 0.793            | descarta (bien)       |

Misma `c`, distinto veredicto: el gate no esta midiendo la particion, esta
midiendo la particion **y** el ruido, y el ruido entra con signo arbitrario.

> Nota sobre las cifras del reporte original. `442/528 = 0.837` y
> `442/465 = 0.951` no son simultaneamente compatibles con "5 % de ruido":
> `465/528` implica `nu = 0.119`. La lectura consistente del probe es
> `{442, 23}` asignadas + 63 en `-1`, que es la que usa la tabla anterior y
> la seccion [5] del playground. El caso `4.9 %` (`{476,26}` + 26) es el que
> si corresponde a `nu = 0.049`, con `c = 0.902` (mas degenerado que los otros
> dos, aunque `select-v3` lo descartara solo por el denominador).

---

## 3. Por que el gate no puede ser un umbral de ARI

Antes de elegir la forma del gate hay que descartar la tentacion obvia:
"descarta si el ARI es demasiado alto". El ARI es alto por construccion cuando
hay una clase dominante. La simulacion [8] lo cuantifica con un predictor
**trivial** que solo acierta la clase mayor y asigna el resto al azar:

| `c`   | ARI (azar puro) | ARI (trivial, K=3) | ARI (trivial, K=8) |
| ----- | --------------- | ------------------ | ------------------ |
| 0.500 | 0.000           | 0.419              | 0.721              |
| 0.650 | -0.000          | 0.575              | 0.828              |
| 0.700 | -0.000          | 0.628              | 0.852              |
| 0.750 | -0.000          | 0.656              | 0.874              |
| 0.800 | -0.001          | 0.691              | 0.891              |
| 0.837 | -0.000          | 0.724              | 0.903              |
| 0.900 | -0.000          | 0.755              | 0.915              |
| 0.950 | 0.000           | 0.775              | 0.922              |

El azar puro da 0 (el ARI esta corregido por azar); el piso viene de la clase
mayor. Con `c = 0.837` el ARI regalado es ~0.72 (K=3), que es el rango en el
que se mueven los ganadores reales del proyecto (0.35-0.79). Cualquier umbral
de ARI que bloqueara ese piso bloquearia tambien las particiones legitimas.
Conclusion: **el gate tiene que actuar sobre `c`, no sobre el ARI**, y el ARI
debe seguir leyendose como criterio relativo entre metodos, no como
certificado de validez.

Corolario para el paper: un ARI alto con `c` alto no es evidencia.

---

## 4. Rutas evaluadas (y por que se descartan)

Se compararon seis formas, todas con el mismo nivel `T0 = 0.70` cuando el nivel
es comparable. `fv` = falso-valida (se acepta una degenerada), `fd` =
falso-degenerada (se descarta una valida). "[gate]" aisla la dominancia
(solo particiones con `K` en `[3,20]`); "pipeline" incluye el gate de `K`.

| [1] forma                   | fv pipeline | fd pipeline | fv gate   | fd gate   |
| --------------------------- | ----------- | ----------- | --------- | --------- |
| `actual  a>0.90`            | 0.274       | 0.070       | **0.614** | 0.001     |
| `asignada a>0.70`           | 0.000       | 0.093       | 0.000     | 0.027     |
| `mixta   a>0.70/(1-nu)^0.5` | 0.000       | 0.069       | 0.000     | 0.000     |
| **`corpus  c>0.70`**        | **0.003**   | **0.069**   | **0.008** | **0.000** |
| `tail    1-nu-c<0.30`       | 0.000       | 0.262       | 0.000     | **0.208** |
| `keff    Hill<2`            | 0.124       | 0.082       | **0.279** | 0.015     |

### 4.1 Ruta 1 del pedido: entropia efectiva (Hill, `exp(H)`)

Implementada como `keff: K_eff < 2 => degenerada`. Falla por dos motivos:

- `K_eff` es funcion del **perfil completo** de tamanos, no de la masa
  dominante. `{0.70, 0.10, 0.10, 0.10}` tiene `K_eff ~ 3.2` y pasaria, siendo
  justo el caso a detectar.
- No usa el ruido en absoluto, asi que reproduce la asimetria del probe.

Medido: `fv(gate) = 0.279`. Descartada.

### 4.2 Ruta 2 del pedido: umbral derivado de un modelo nulo

Un nulo honesto para `max_class_share` es una composicion aleatoria
(`Dirichlet(1)`) con el `K` observado. Sus cuantiles, sin embargo, dependen
brutalmente de `K`:

| [9] `K` | q50   | q95   | q99   | q95 vs `c=0.837`           |
| ------- | ----- | ----- | ----- | -------------------------- |
| 3       | 0.592 | 0.873 | 0.946 | acepta (no detecta el bug) |
| 5       | 0.437 | 0.687 | 0.791 | descarta                   |
| 8       | 0.324 | 0.519 | 0.626 | descarta                   |
| 11      | 0.262 | 0.418 | 0.502 | descarta                   |
| 14      | 0.221 | 0.355 | 0.430 | descarta                   |
| 18      | 0.185 | 0.294 | 0.357 | descarta                   |
| 20      | 0.172 | 0.272 | 0.330 | descarta                   |

El umbral variaria entre 0.87 y 0.27 segun `K`. Eso viola dos requisitos:

1. **No detecta el caso que motivo el cambio**: a `K = 3` el cuantil 95 es
   0.873 y `0.837 < 0.873` pasaria.
2. **Duplica el gate de `K`**: `K_eff` y el cuantil nulo son funciones
   monotonas de `K`, de modo que la regla dejaria de ser compatible con
   `k_bounds` (dos condiciones que miden lo mismo con nombres distintos).

Ademas el nulo responde otra pregunta: "es la clase mayor mas grande de lo que
daria el azar?", cuya respuesta en un corpus real es siempre si, y es
precisamente el fenomeno de interes (un corpus puede estar legitimamente
dominado por un tema). No distingue "el corpus esta dominado" de "el
clustering colapso". Descartada.

### 4.3 Ruta 3 del pedido: correccion por ruido con umbral decreciente

Formalizada como `tail: 1 - nu - c < 0.30`, equivalente a `c < 0.70 - nu`
(la masa no dominante del corpus debe superar 0.30). A `nu = 0` coincide con
la regla propuesta; a partir de ahi resta `nu` **por segunda vez**, porque `c`
ya contiene el factor `(1 - nu)`.

Error medido: `fd(gate) = 0.208` global, y por regimen **0.17 / 0.42 / 1.00**
en `nu = 0.35 / 0.50 / 0.65`. Es decir: a ruido alto rechaza el 100 % de las
particiones legitimas. Descartada: es la correccion con el signo invertido.

El mismo argumento en una linea: bajo ruido informativo la clase dominante
casi no pierde puntos (`q_1 ~ 0`), asi que `c ~ p_1` y `a = c/(1-nu) ~ p_1/(1-nu)`.
Exigir que el umbral **decrezca** con `nu` equivale a exigir
`p_1 <= (0.70 - nu)/(1 - q_1)`, que para `nu = 0.35` y `q_1 ~ 0` da
`p_1 <= 0.35`: imposible para una particion legitima con `p_1 = 0.5`.

### 4.4 Forma mixta (`gamma = 0.5`)

`a > 0.70 / (1-nu)^0.5` no produce errores en la simulacion
(`fv = fd = 0.000`), pero es una interpolacion entre dos reglas con semantica
clara: `gamma = 0` es "umbral constante sobre lo asignado" y `gamma = 1` es
"umbral constante sobre el corpus". El exponente `0.5` no tiene estimando: no
se puede escribir que cantidad mide. Se reporta como sensibilidad, no como
regla.

### 4.5 Forma "asignada" (`a > 0.70`)

Cambia el nivel pero no el denominador. `fv = 0`, pero `fd = 0.027` global y
hasta `0.08` a `nu = 0.65`: al no descontar el ruido, `a` se infla y descarta
particiones legitimas con ruido alto. Es el espejo exacto del bug actual.

---

## 5. Regla propuesta

### 5.1 Definicion

Para cada metodo `m` con particion cruda:

```
c_m = max_class_share_m * (1 - noise_frac_m)        (fraccion del corpus)

m es no viable por dominancia  <=>  c_m > C0,   C0 = 0.70
```

Implementada en `_feasibility` junto al resto de condiciones necesarias. El
motivo del descarte reporta `a`, `c`, `nu` y el umbral, para que el log sea
auditable.

### 5.2 Propiedades

1. **Monotonia.** Para `a` fijo, `c` decrece estrictamente en `nu`: mas ruido
   nunca convierte una particion valida en degenerada. Para `nu` fijo, `c`
   crece estrictamente en `a`. La region de aceptacion
   `{(a,nu) : a(1-nu) <= C0}` es convexa y mononota en ambas coordenadas.
2. **Continuidad.** Sin escalones ni cambios de regimen: una sola desigualdad
   lineal en `(a, nu)`. No hay "si `nu > 0.2` entonces otra regla".
3. **Un solo parametro libre.** `C0`. Se eliminan `gamma`, `tau` y el umbral
   de `K_eff`. `k_bounds`, `noise_max` y `coverage_floor` no se tocan.
4. **Compatible con `k_bounds`.** La desigualdad no contiene `K`: no duplica
   el gate de `K`. Ambos son condiciones necesarias y el orden de evaluacion
   es irrelevante.
5. **Lectura dual.** `c > C0` (corpus) y `a > C0/(1-nu)` (asignadas) son la
   misma regla. La segunda es la que se imprime hoy, y su umbral **se relaja**
   con el ruido, que es la direccion correcta porque `a` ya esta normalizada
   por lo asignado.

### 5.3 Por que sobre el corpus y no sobre lo asignado

`c` es la cantidad que entra en el argumento del ARI (seccion 3): la masa de
pares que un predictor trivial puede acertar. `a` es `c` dividido por una
cantidad que depende de un hiperparametro del metodo (`min_cluster_size` de
HDBSCAN, ventana de coref), no del corpus. Un gate sobre `a` hace que el
veredicto dependa de ese hiperparametro; uno sobre `c` no.

---

## 6. Calibracion de `C0`

### 6.1 Verdad declarada (zonas de indiferencia)

La simulacion necesita una verdad generativa. Se declara, **antes** de mirar
los resultados:

- **valida**: `p_1 <= 0.65` (clase mayor del corpus);
- **degenerada**: `p_1 >= 0.75`;
- **ambigua**: `0.65 < p_1 < 0.75`, no se puntua.

El hueco `(0.65, 0.75)` es una **decision de politica**, no una estimacion.
Su funcion es impedir que `C0` se ajuste contra los mismos casos que debe
clasificar; su costo es que las particiones en esa banda no se evaluan. Un
revisor puede objetar (con razon) que el hueco no lo identifican los datos.
Ver seccion 14.

### 6.2 Presupuesto de error declarado

| error            | que significa          | presupuesto                          |
| ---------------- | ---------------------- | ------------------------------------ |
| falso-valida     | aceptar una degenerada | **<= 1 % global**, <= 10 % por celda |
| falso-degenerada | descartar una valida   | <= 5 % global                        |

Se controla el **falso-valida** y no el simetrico porque el costo es
asimmetrico: una degenerada aceptada gana por ARI y se publica como hallazgo
(invisible en el resultado); una valida descartada produce abstención, que es
visible, trazable y recuperable.

### 6.3 Protocolo de seleccion del nivel

1. Fijar la verdad (6.1) y el presupuesto (6.2).
2. Barrer `T0` en una rejilla (`0.58 ... 0.80`, paso 0.02) y medir `fv`, `fd`
   y el peor regimen.
3. Admitir el conjunto `fv <= 1 %`: de la tabla [4] resulta
   `T0 in [0.58, 0.70]`.
4. Dentro de ese conjunto, exigir `fd <= 0.5 %`: queda `T0 in [0.64, 0.70]`.
5. Elegir el **extremo mas permisivo** del conjunto admisible (`0.70`), porque
   dentro de el la unica diferencia es cuantas particiones legitimas y
   desbalanceadas se descartan de mas.

`0.70` es ademas el punto medio del hueco declarado (`0.65`, `0.75`). Dos
criterios independientes coinciden; ninguno de los dos se uso para "afinar"
el numero.

### 6.4 Curva de sensibilidad (tabla [4])

```
    T0       fv       fd   peor celda   veredicto
  0.58    0.000    0.022        0.068   [ok]
  0.60    0.000    0.013        0.045   [ok]
  0.62    0.000    0.006        0.021   [ok]
  0.64    0.000    0.003        0.010   [ok]
  0.66    0.000    0.001        0.004   [ok]
  0.68    0.002    0.000        0.018   [ok]
  0.70    0.008    0.000        0.051   [ok]  <- elegido
  0.72    0.024    0.000        0.127   [X]
  0.74    0.065    0.000        0.287   [X]
  0.76    0.169    0.000        0.600   [X]
  0.78    0.278    0.000        0.811   [X]
  0.80    0.402    0.000        0.942   [X]
```

Lectura: por debajo de `0.64` el gate empieza a cobrar `fd` (descarta
particiones legitimas desbalanceadas); por encima de `0.70` `fv` se dispara.
La meseta es un intervalo, no un punto: **la simulacion no identifica `C0`
dentro de `[0.64, 0.70]`**, solo descarta los extremos.

---

## 7. Tabla `noise_frac -> umbral`

Ambas columnas son la misma regla. `umbral a = 1` o mayor significa que el
gate de dominancia no restringe (cualquier particion con `K` valido y ruido
`<= noise_max` pasa).

| `nu` | umbral sobre `c` | umbral sobre `a` | efecto          |
| ---- | ---------------- | ---------------- | --------------- |
| 0.00 | 0.700            | 0.700            | activo          |
| 0.05 | 0.700            | 0.737            | activo          |
| 0.10 | 0.700            | 0.778            | activo          |
| 0.15 | 0.700            | 0.824            | activo          |
| 0.20 | 0.700            | 0.875            | activo          |
| 0.25 | 0.700            | 0.933            | activo          |
| 0.30 | 0.700            | 1.000            | frontera        |
| 0.35 | 0.700            | 1.077            | sin restriccion |
| 0.50 | 0.700            | 1.400            | sin restriccion |
| 0.65 | 0.700            | 2.000            | sin restriccion |

Comparacion con el umbral implicito de `select-v3` (`0.90(1-nu)`):

| `nu` | implicito v3 | propuesto v4 | diferencia                       |
| ---- | ------------ | ------------ | -------------------------------- |
| 0.00 | 0.900        | 0.700        | v4 mas estricto (corrige el bug) |
| 0.12 | 0.792        | 0.700        | v4 algo mas estricto             |
| 0.20 | 0.720        | 0.700        | casi igual                       |
| 0.35 | 0.585        | 0.700        | v4 mas permisivo                 |
| 0.50 | 0.450        | 0.700        | v4 mas permisivo                 |

---

## 8. Regimen donde el gate queda vacuo (`nu >= 0.30`)

A partir de `nu = 0.30` el umbral sobre `a` alcanza 1 y el gate deja de
restringir. Esto **no** es un hueco de la regla: es lo que corresponde, porque
`noise_max = 0.50` ya acota el ruido aceptable, y ademas las particiones
degeneradas dejan de llegar al gate antes:

| [3] llegada al gate (K en [3,20]) | nu=0 | 0.05 | 0.10 | 0.20 | 0.35     | 0.50     | 0.65     |
| --------------------------------- | ---- | ---- | ---- | ---- | -------- | -------- | -------- |
| validas                           | 1.00 | 1.00 | 0.99 | 0.97 | 0.93     | 0.87     | 0.77     |
| degeneradas                       | 0.99 | 0.97 | 0.83 | 0.33 | **0.00** | **0.00** | **0.00** |

Bajo ruido informativo las clases satelite desaparecen, `K` cae por debajo de
3 y **`k_bounds` descarta antes**. El resto de la tabla [2] lo confirma: a
partir de `nu = 0.35` la columna del gate esta vacia (`--`), porque no hay
degeneradas que puntuar.

Consecuencia que hay que declarar: **a ruido alto este gate no hace el
trabajo; lo hace `k_bounds`**. Si se quisiera cubrir ese regimen habria que
bajar `noise_max`, no bajar `C0`.

---

## 9. Tasas de error por regimen (numeros reales)

Solo gate de dominancia (particiones con `K` en `[3,20]`), `fv/fd`. Es la
salida literal de la seccion [2] del script:

```
nu nominal ->      0.00      0.05      0.10      0.20      0.35      0.50      0.65
nu real    ->     0.000     0.050     0.100     0.199     0.350     0.500     0.650
actual  a>0.90               0.88/0.00 0.64/0.00 0.46/0.00 0.10/0.00  -- /0.00  -- /0.00  -- /0.00
asignada a>0.70              0.00/0.00 0.00/0.00 0.00/0.00 0.00/0.02  -- /0.04  -- /0.06  -- /0.08
mixta   a>0.70/(1-nu)^0.5    0.00/0.00 0.00/0.00 0.00/0.00 0.00/0.00  -- /0.00  -- /0.00  -- /0.00
corpus  c>0.70               0.00/0.00 0.00/0.00 0.00/0.00 0.05/0.00  -- /0.00  -- /0.00  -- /0.00
tail    1-nu-c<0.30          0.00/0.00 0.00/0.00 0.00/0.01 0.00/0.06  -- /0.17  -- /0.42  -- /1.00
keff    Hill<2               0.51/0.00 0.29/0.00 0.09/0.00 0.00/0.00  -- /0.03  -- /0.04  -- /0.04
```

`--` = no hay degeneradas en esa celda que lleguen al gate (las descarta
`k_bounds` antes); ver seccion 8.

Error por `K` con la regla propuesta (tabla [7]):

| `K` | `fv`  | `fd`  | n(gate) |
| --- | ----- | ----- | ------- |
| 3   | 0.027 | 0.000 | 886     |
| 5   | 0.013 | 0.000 | 1100    |
| 8   | 0.013 | 0.000 | 1193    |
| 11  | 0.003 | 0.000 | 1224    |
| 14  | 0.000 | 0.000 | 1231    |
| 18  | 0.000 | 0.000 | 1237    |
| 20  | 0.003 | 0.000 | 1228    |

El error no crece con `K`; es maximo a `K = 3`, que es donde `c = 0.70` esta
mas cerca de una particion legitima muy concentrada.

---

## 10. Caso del probe v4

Tabla [5], con la misma estructura y distinto ruido:

```
caso                       a     nu     c       0.90  corpus  veredicto
denso sin ruido        0.837  0.000 0.837     acepta descarta  DISCREPAN
HDBSCAN 11.9% ruido    0.951  0.119 0.837   descarta descarta  ambas descartan
HDBSCAN  4.9% ruido    0.948  0.049 0.902   descarta descarta  ambas descartan
```

Los tres son degenerados (`c >= 0.837`). `select-v3` acertaba en dos y fallaba
en el tercero por el denominador; `select-v4` acierta en los tres. La
discrepancia desaparece porque se elimina la variable que la causaba (`nu`),
no porque se haya movido el nivel.

---

## 11. Politica de error y su tasa

|                                         | global    | peor celda          | como se mide    |
| --------------------------------------- | --------- | ------------------- | --------------- |
| falso-valida (gate aislado)             | **0.008** | 0.051 (`nu = 0.20`) | tabla [4]       |
| falso-degenerada (gate aislado)         | **0.000** | 0.000               | tabla [4]       |
| falso-valida (pipeline, con `k_bounds`) | 0.003     | no medida por celda | tabla [1]       |
| falso-degenerada (pipeline)             | 0.069     | 0.23 (`nu = 0.65`)  | tablas [1], [3] |

El `fd` del pipeline (0.069) **no** viene de este gate. Se reconstruye con la
tabla [3]: la fraccion de particiones validas que no llega al gate es
0.00 / 0.00 / 0.01 / 0.03 / 0.07 / 0.13 / 0.23 en
`nu = 0 / .05 / .10 / .20 / .35 / .50 / .65`, y su promedio sobre las siete
celdas es 0.067. Ese descarte lo produce `k_bounds`: cuando el ruido elimina
clases satelite, `K` cae por debajo de 3. Es comportamiento preexistente y el
diff no lo modifica; a `nu = 0` el descarte por `K` es practicamente nulo
(llegada 1.00).

Declaracion para el paper: _el gate controla el falso-valida a <= 1 % global
(<= 10 % por regimen) bajo el modelo de ruido informativo de la seccion 12.1;
su falso-degenerada es 0 % en las celdas simuladas, con la salvedad de que
`k_bounds` aporta el 6.9 % restante del pipeline_.

---

## 12. Diffs exactos

Los tres archivos siguientes **no** se han modificado: este documento es la
especificacion. El diff es ASCII en las cadenas impresas (la consola del host
es cp1252); los comentarios siguen el estilo ya presente en cada archivo.

### 12.1 `src/reii/method_selection.py`

```diff
--- a/src/reii/method_selection.py
+++ b/src/reii/method_selection.py
@@ -17,1 +17,1 @@
-Regla (lexicográfica, versión ``select-v3``)
+Regla (lexicográfica, versión ``select-v4``)
@@ -35,6 +35,10 @@
-6. **No dominancia** (select-v3): si la clase mayor se come más del
-   ``max_class_share_max`` (90 %) de las UCEs **asignadas**, la partición es
-   degenerada (una clase + satélites) y el método se descarta. El ARI de una
-   partición así es alto por construcción: replicar una clase gigante es
-   fácil. Se mide sobre lo asignado, no sobre el corpus, para que el ruido de
-   HDBSCAN no maquille la dominancia.
+6. **No dominancia** (select-v4): la dominancia se mide sobre el **corpus**,
+   no sobre lo asignado. ``c = max_class_share * (1 - noise_frac)`` es la
+   fracción del corpus ocupada por la clase mayor; si ``c > C0`` (0.70 por
+   defecto) la partición es degenerada (una clase + satélites) y el método se
+   descarta. El ARI de una partición así es alto por construcción: replicar
+   sólo la clase gigante ya cobra ARI~0.72 cuando c=0.84. Medir sobre lo
+   asignado hacía depender el veredicto del ruido: {442,45,30,11} sin ruido
+   daba 0.837 (pasaba) y {442,23}+63 de ruido daba 0.951 (se descartaba), con
+   idéntico c=0.837. Derivación y calibración en
+   ``docs/design_v5/GATE_NOISE.md``.
@@ -86,1 +86,1 @@
-RULE_VERSION = "select-v3"
+RULE_VERSION = "select-v4"
@@ -129,9 +129,23 @@
     @property
     def usable_coverage(self) -> float:
         """UCEs estables y en una clase con masa suficiente / ``n_total``."""
         return max(0.0, float(self.coverage) - float(self.tiny_frac))
-
+    @property
+    def dominance_corpus(self) -> float:
+        """Clase mayor / ``n_total`` = ``max_class_share * (1 - noise_frac)``.
+
+        ``max_class_share`` divide por las UCEs asignadas; el ruido
+        (``label < 0``) la infla sin que la partición cambie, así que el gate
+        necesita el denominador común del corpus para no medir "dominancia
+        diluida". La conversión es exacta porque
+        ``_partition_sizes_from_labels`` ya excluye ``label < 0``.
+        """
+        share = float(self.max_class_share or 0.0)
+        noise = min(1.0, max(0.0, float(self.noise_frac or 0.0)))
+        return share * (1.0 - noise)
+
     def as_row(self) -> Dict[str, Any]:
         row = asdict(self)
         row["usable"] = self.usable_coverage
+        row["dominance_corpus"] = self.dominance_corpus
         return row
@@ -165,4 +165,7 @@
     min_class_share: float = 0.01
     noise_max: float = 0.50
-    max_class_share_max: float = 0.90
+    #: Gate de dominancia sobre el CORPUS: c = max_class_share*(1-noise_frac).
+    max_class_share_max: float = 0.70
     k_bounds: Tuple[int, int] = (3, 20)
+    #: c del ganador (clase mayor / n_total). Diagnóstico y auditoría.
+    winner_dominance_corpus: float = float("nan")
@@ -191,10 +191,11 @@
         for key in (
             "winner_ari",
             "winner_coverage",
             "winner_usable_coverage",
             "runner_up_ari",
             "runner_up_coverage",
             "runner_up_usable_coverage",
             "delta_ari",
             "delta_coverage",
+            "winner_dominance_corpus",
         ):
@@ -210,3 +210,3 @@
             f"  {'met':4}{'K':>4}{'ARI':>8}{'cov':>7}{'usable':>8}{'noise':>7}"
-            f"{'tiny':>7}{'dom':>7}  {'perturbacion':14}  veredicto",
+            f"{'tiny':>7}{'dom':>7}{'domC':>7}  {'perturbacion':14}  veredicto",
         ]
@@ -226,4 +226,5 @@
                 f"{float(row.get('noise_frac') or 0.0):>7.2f}"
-                f"{float(row.get('tiny_frac') or 0.0):>7.2f}  "
-                f"{float(row.get('max_class_share') or 0.0):>7.2f}  "
+                f"{float(row.get('tiny_frac') or 0.0):>7.2f}"
+                f"{float(row.get('max_class_share') or 0.0):>7.2f}"
+                f"{float(row.get('dominance_corpus') or 0.0):>7.2f}  "
                 f"{str(row.get('perturbation') or '-'):14}  {tag}"
@@ -247,6 +247,8 @@
         _win_row = next((r for r in self.all_metrics if r["method"] == self.winner), {})
+        _win_dom = float(_win_row.get("dominance_corpus") or 0.0)
         out.append(
             f"     clase mayor={float(_win_row.get('max_class_share') or 0.0):.2f} "
-            f"de las asignadas (gate: >{self.max_class_share_max:.2f} = "
+            f"de las asignadas = {_win_dom:.2f} del corpus "
+            f"(gate: >{self.max_class_share_max:.2f} del corpus = "
             "particion degenerada)"
         )
@@ -279,1 +279,1 @@
-    max_class_share_max: float = 0.90,
+    max_class_share_max: float = 0.70,
@@ -295,7 +295,9 @@
     dom = float(getattr(metric, "max_class_share", 0.0) or 0.0)
-    if dom > float(max_class_share_max):
+    noise = min(1.0, max(0.0, float(getattr(metric, "noise_frac", 0.0) or 0.0)))
+    dom_corpus = dom * (1.0 - noise)
+    if dom_corpus > float(max_class_share_max):
         return (
-            f"clase dominante {dom:.2f} de las asignadas > max "
-            f"{float(max_class_share_max):.2f} (particion degenerada: una clase"
-            " y satelites)"
+            f"clase dominante {dom:.2f} de las asignadas = {dom_corpus:.2f} del "
+            f"corpus > max {float(max_class_share_max):.2f} (particion "
+            f"degenerada: una clase y satelites; ruido {noise:.2f} descontado)"
         )
@@ -319,4 +319,4 @@
     noise_max: float = 0.50,
-    max_class_share_max: float = 0.90,
+    max_class_share_max: float = 0.70,
 ) -> Selection:
-    """Aplica la regla ``select-v3`` y devuelve la ``Selection``.
+    """Aplica la regla ``select-v4`` y devuelve la ``Selection``.
@@ -401,2 +401,3 @@
         winner_usable_coverage=float(win.usable_coverage),
+        winner_dominance_corpus=float(win.dominance_corpus),
         runner_up=runner.method if runner else None,
```

Notas de aplicacion:

- `_feasibility` recibe `noise_frac` desde `MethodMetrics`; no hace falta
  pasar nada nuevo por parametro.
- Los dos `Selection(...)` de abstención (L328 y L355) no necesitan cambios:
  usan el parametro `max_class_share_max` y `winner_dominance_corpus` queda
  en `NaN` (que `to_dict` serializa como `null` y `from_dict` restaura).
- La cabecera de la tabla gana la columna `domC`; el ancho total cambia, pero
  los tests solo comprueban subcadenas, no el ancho.

### 12.2 `src/reii/main_workflow.py`

```diff
--- a/src/reii/main_workflow.py
+++ b/src/reii/main_workflow.py
@@ -1003,4 +1003,6 @@
-    # Gate de no dominancia: si la clase mayor se come mas de esta fraccion de
-    # las UCEs ASIGNADAS (particion cruda), la particion es degenerada
-    # (una clase + satelites) y el metodo se descarta.
-    method_selection_max_class_share: float = 0.90
+    # Gate de no dominancia: si la clase mayor ocupa mas de esta fraccion del
+    # CORPUS (c = max_class_share * (1 - noise_frac), particion cruda), la
+    # particion es degenerada (una clase + satelites) y el metodo se descarta.
+    # Medir sobre el corpus y no sobre lo asignado evita que el ruido de
+    # HDBSCAN decida el veredicto (ver docs/design_v5/GATE_NOISE.md).
+    method_selection_max_class_share: float = 0.70
@@ -5706,3 +5706,3 @@
                 max_class_share_max=float(
-                    getattr(self.config, "method_selection_max_class_share", 0.90)
+                    getattr(self.config, "method_selection_max_class_share", 0.70)
                 ),
@@ -8670,4 +8670,8 @@
             trial.set_user_attr(
                 "max_class_share_winner",
                 float(_win_row.get("max_class_share") or 0.0),
             )
+            trial.set_user_attr(
+                "dominance_corpus_winner",
+                float(_win_row.get("dominance_corpus") or 0.0),
+            )
```

`_partition_sizes_from_labels` y `_max_share` **no** cambian: `a` sigue
siendo la cantidad que se mide y se registra; el umbral es lo que se aplica
sobre `c`.

### 12.3 Aviso de compatibilidad de la corrida en curso

La corrida actual usa el codigo de `select-v3`. Al aplicar el diff, el
siguiente `--mode select` puede elegir otro metodo (ese es el objetivo), pero
**`data/method_selection.json` puede reutilizar un snapshot `select-v3`**:
`load_selection` valida `corpus_fingerprint` y `uce_target_size`, no
`rule_version`. Antes de re-correr:

```
del data\method_selection.json
```

(o moverlo a `data/method_selection_v3.json` como registro). El historial
`data/method_selection_log.jsonl` es append-only y puede quedarse: cada linea
lleva su `rule_version`.

---

## 13. Tests existentes que cambian

`tests/test_method_selection.py`:

| test                                      | estado            | motivo                                                   |
| ----------------------------------------- | ----------------- | -------------------------------------------------------- |
| `test_dominancia_092_descarta`            | **sigue pasando** | `0.92 > 0.70`, y `"0.92"` sigue apareciendo en el motivo |
| `test_dominancia_089_no_descarta`         | **rompe**         | `0.89` sin ruido ahora es degenerada                     |
| `test_dominancia_gate_aparece_en_explain` | **rompe**         | busca `"gate: >0.90"`                                    |
| `test_explain_is_ascii_only`              | sigue pasando     | el texto nuevo es ASCII                                  |
| `test_roundtrip_preserves_decision`       | sigue pasando     | el campo nuevo esta en la lista de `NaN`                 |

Reemplazos exactos (no se ha escrito el archivo):

```python
    def test_dominancia_089_con_ruido_no_descarta(self):
        """0.89 de las asignadas con 30% de ruido = 0.62 del corpus: valida."""
        c = MethodMetrics(
            "C", 3, 0.785, 0.371, noise_frac=0.30, max_class_share=0.89
        )
        a = _m("A", 4, 0.250, 0.574)
        sel = select_method([a, c])
        assert sel.winner == "C"
        assert not any("dominante" in d["why"] for d in sel.discarded)
        assert sel.winner_dominance_corpus == pytest.approx(0.623)

    def test_dominancia_gate_aparece_en_explain(self):
        """El umbral se imprime en unidades del corpus (0.70)."""
        c = MethodMetrics("C", 3, 0.785, 0.371, max_class_share=0.60)
        sel = select_method([c])
        text = sel.explain_text()
        assert "clase mayor=0.60" in text
        assert "= 0.60 del corpus" in text
        assert "gate: >0.70 del corpus" in text

    def test_dominancia_ignora_el_ruido_en_el_denominador(self):
        """Misma particion, misma c: el ruido mueve a pero no el veredicto."""
        limpia = MethodMetrics("C", 3, 0.785, 0.371, max_class_share=442 / 528)
        ruidosa = MethodMetrics(
            "C",
            3,
            0.785,
            0.371,
            noise_frac=63 / 528,
            max_class_share=442 / 465,
        )
        a = _m("A", 4, 0.250, 0.574)
        for m in (limpia, ruidosa):
            sel = select_method([a, m])
            assert sel.winner == "A", m.max_class_share
            assert any("dominante" in d["why"] for d in sel.discarded)
```

El tercero es el test de regresion del bug: los dos casos tienen `c = 0.837`
y deben compartir veredicto.

`tests/test_select_mode_objective.py::TestDominanceGateWiring` sigue pasando:
`test_partition_sizes_exclude_noise` comprueba el helper, que no cambia, y
`test_max_class_share_winner_is_recorded` usa `max_class_share=0.52`, que
pasa el gate (`c = 0.52`). Su docstring queda conceptualmente desactualizado
("diluir con -1 no debe valer"): ahora diluir es exactamente lo que se hace,
pero contra un umbral mas bajo. Conviene reescribir el comentario, no el
assert.

---

## 14. Limitaciones y sesgos admitidos

1. **No identificabilidad con ruido informativo.** Con `q_1 = 0` se cumple
   `c = p_1`; con `q_1 > 0` se cumple `c = p_1 (1 - q_1) < p_1`. Solo se
   observa `c`, nunca `p_1`, de modo que `p_1 in [c, c/(1-nu)]` (extremos:
   ruido solo de satelites vs. ruido proporcional). El gate sobre `c` es la
   **cota liberal** (puede aceptar una degenerada si el ruido afecto tambien a
   la clase mayor) y el gate sobre `a` es la **cota conservadora** (descarta
   validas). Se eligio la liberal para controlar falso-valida a costa de
   falso-degenerada; es una decision explicita, no un olvido.
2. **`C0` no lo identifican los datos.** La simulacion solo acota
   `[0.64, 0.70]`; el valor sale del hueco declarado `(0.65, 0.75)`. Si el
   hueco estuviera mal puesto, `C0` estaria mal puesto. Mitigacion: la curva
   de sensibilidad publicada permite recalcular el veredicto con otro nivel.
3. **El modelo de ruido es una eleccion.** Se usa dropout por umbral
   (`q_j = min(1, t/p_j)`), que reproduce HDBSCAN (clases pequenas
   desaparecen). Con dropout uniforme la conclusion se invierte: `a` no se
   inflaria y el gate actual no tendria el bug. La evidencia empirica del
   probe (`{442,23}` + 63) es consistente con el modelo informativo, no con
   el uniforme.
4. **La simulacion no modela** `coverage_floor`, `tiny_frac`,
   `min_class_share` ni la geometria real del embedding. Aisla el gate de
   dominancia mas `k_bounds`. Las tasas del pipeline de la seccion 11 son las
   que incluyen `k_bounds`, y no incluyen el piso de cobertura.
5. **La familia "valida" es demasiado limpia.** `p_min >= 0.005` no admite
   clases de 1-2 UCEs, que en ALCESTE real existen y que el pipeline ya trata
   por otra via (`tiny_frac`). El `fd = 0` de la seccion 9 no cubre ese caso.
6. **A `nu >= 0.30` el gate es vacuo**, y `noise_max = 0.50` permite llegar
   ahi. En ese regimen la proteccion real es `k_bounds`. Si se quiere una
   garantia en `nu in [0.30, 0.50]`, hay que bajar `noise_max`, no `C0`.

---

## 15. Reproducibilidad

```
python scripts/playground/gate_calibration.py
python scripts/playground/gate_calibration.py --reps 200
python scripts/playground/gate_calibration.py --metrics-selftest
```

- Autocontenido: no importa `reii.main_workflow` (Stanza), no usa red, Docker
  ni la BD, no escribe en `data/`, `output/` ni `models/`.
- Semillas fijas (`build_sample(1000)` calibracion, `build_sample(2000)`
  validacion). Los numeros de este documento son reproducibles bit a bit.
- Tiempo medido en la maquina del proyecto: ~14 s el estudio del gate,
  ~40 s con `--metrics-selftest` (la mayor parte, lectura del JSON de 29 MB y
  los 300 remuestreos de `C_in`). El pipeline en Docker corriendo en paralelo
  puede duplicar esos tiempos.
- Salida ASCII puro (`->`, `[ok]`, `[!]`, `[X]`), sin box-drawing ni emojis.
