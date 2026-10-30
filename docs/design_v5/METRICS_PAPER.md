# Metricas del paper: `C_in`, `d` y `RMSD`

Estado: especificacion cerrada, pendiente de aplicar el diff de la seccion 5.
Destino: `src/reii/quality/metrics.py` (funciones puras, mismo estilo del
modulo).
Prototipo ejecutable: `scripts/playground/gate_calibration.py`
(`--metrics-selftest`), con la misma logica que el diff.

---

## 0. Que promete el paper y que existe hoy

| metrica                     | promesa                                     | estado                              |
| --------------------------- | ------------------------------------------- | ----------------------------------- |
| `C_ext`                     | acuerdo con el estandar humano              | existe: `external_coherence`        |
| acuerdo entre particiones   | FMI/NMI/ARI                                 | existe: `cross_partition_agreement` |
| coherencia interna `C_in`   | homogeneidad intra-clase                    | **no existe**                       |
| diversidad de particion `d` | equilibrio de tamanos de clase              | **no existe**                       |
| `RMSD`                      | distancia de proporciones entre particiones | **no existe**                       |

`silhouette_labels` existe y es de la misma familia que `C_in`, pero no sirve
como sustituto: (i) es un cociente entre distancia intra y extra, de modo que
castiga particiones con clases vecinas legitimas; (ii) con `metric="cosine"`
sobre 384 dimensiones es O(n^2) y se submuestrea; (iii) no tiene nulo, y sin
nulo un `C_in` alto no distingue "clases coherentes" de "corpus homogeneo".
Se conserva como diagnostico, no como la metrica del paper.

Las tres funciones nuevas devuelven `dict` con valor, IC bootstrap, nulo,
`n` y la **etiqueta del espacio/referencia**. Ninguna devuelve un numero
suelto.

---

## 1. `C_in`: coherencia interna

### 1.1 Definicion

Sea `X` una matriz `n x d` en un espacio vectorial **declarado** por
`space_id` (p. ej. `"uce_embedding_384d"`, `"chi2_profiles"`), `y` las
etiquetas de clase (`y_i >= 0`; se descartan `y_i < 0`), `C_k` la clase `k`
con `n_k` elementos, `m = sum_k n_k` y `K` el numero de clases.

Normalizacion y centroide:

```
x~_i = x_i / ||x_i||                      (si ||x_i|| = 0, se deja en 0)
mu_k = (1/n_k) * sum_{i in C_k} x~_i
```

Score por UCE y por clase (truncado a `[0, 1]`):

```
s_i = max(0, <x~_i, mu_k> / ||mu_k||)     = max(0, cos(x_i, mu_k))
S_k = (1/n_k) * sum_{i in C_k} s_i
```

Las tres versiones que se reportan:

```
C_in        = (1/K) * sum_k S_k                    (macro; es la del paper)
C_in^w      = sum_k (n_k/m) * S_k                  (ponderada por tamano)
C_in^pair   = (1/K) * sum_k mean_{i<j in C_k} max(0, cos(x_i, x_j))
```

y el efecto sobre el nulo:

```
excess = C_in - E_nulo[C_in]
```

### 1.2 Por que macro y no ponderada

La version ponderada la domina la clase mayor: si `n_1/m = 0.4`, el 40 % del
numero depende de una sola clase, y una clase pequena mal separada desaparece
del promedio. La macro da a cada clase el mismo peso, que es lo que se quiere
cuando la pregunta es "son las clases homogeneas" y no "esta la mayoria del
corpus bien agrupada". Se reportan las dos porque la discrepancia es
informativa: `C_in^w << C_in` indica que las clases pequenas son las malas.

`C_in^pair` (media de similitudes pairwise intra-clase) se reporta como
control del sesgo del centroide: el coseno a un centroide esta sesgado hacia
arriba cuando `n_k` es pequeno (el centroide se ajusta a sus propios puntos).
Si `C_in` y `C_in^pair` discrepan mucho, el valor macro es en parte artefacto
de tamanos. En el bench, `C_in = 0.8165` y `C_in^pair = 0.6550`: el
centroide regala ~0.16 y hay que decirlo, no esconderlo.

### 1.3 Sesgo por dimension y truncado

En `d` alta, el coseno entre dos vectores no relacionados se concentra cerca
de 1, de modo que **el valor absoluto de `C_in` no es interpretable**: en el
bench, con 384 dimensiones, el nulo por etiquetas permutadas ya vale 0.7950.
Por eso:

- se reporta `excess` como tama�o de efecto, no `C_in` solo;
- se reporta `frac_negative` (proporcion de cosenos crudos < 0, truncados a
  0). Si esa fraccion no es despreciable, el truncado esta haciendo trabajo y
  hay que decirlo. En el bench vale 0.0000.

### 1.4 Nulo: permutacion de etiquetas

Hipotesis nula: _las etiquetas no llevan informacion sobre el espacio_, es
decir, la asignacion de UCEs a clases es intercambiable. Se remuestrean las
etiquetas `y` (permutacion uniforme, que preserva exactamente los tamanos de
clase) y se recalcula `C_in`:

```
C_in*_b = C_in(X, pi_b(y)),   b = 1..B
null_mean = mean_b C_in*_b
null_std  = sd_b C_in*_b
z         = (C_in - null_mean) / null_std
p         = #{b : C_in*_b >= C_in} / B      (una cola)
```

Es el nulo correcto porque fija el espacio y los tamanos y solo rompe la
asociacion; compara contra "el mismo corpus, agrupado al azar con el mismo
perfil de tamanos". Es la familia estandar de validacion por modelo nulo
(permutacion de etiquetas), no una invencion.

`p >= 0.05` significa: estas clases no son mas coherentes que un corpus
homogeneo partido al azar. Es la frase que va en el paper, y es la que evita
la objecion "su `C_in` de 0.82 solo dice que sus embeddings son densos".

---

## 2. `d`: diversidad de la particion

### 2.1 Definicion

Con `p_k = n_k / m` (sobre UCEs asignadas):

```
H          = -sum_k p_k ln p_k                (entropia de Shannon, nats)
d          = H / ln K                          (entropia normalizada, [0,1])
Simpson    = 1 - sum_k p_k^2                   (Gini-Simpson)
K_eff      = exp(H)                            (numero efectivo de Hill)
max_share  = max_k p_k
```

`K_eff` es la escala interpretable: "la particion se comporta como si tuviera
`K_eff` clases iguales". Para `K = 7` con tamanos casi iguales, `K_eff ~ 6.6`;
para una clase que se come el 80 %, `K_eff ~ 1.6`.

### 2.2 Que NO es

**No es `diversidad_semantica`**, la metrica por UCE que ya existe en el
pipeline y vale ~0.82 por UCE (mide dispersion lexica dentro de una unidad de
contexto). `d` es una propiedad de la **particion**: solo depende del vector
de tamanos de clase, no del contenido ni de que UCE este en que clase.

Tampoco es una medida de calidad: una particion perfectamente balanceada de
un corpus sin estructura tiene `d = 1` y es basura. `d` se reporta como
descripcion del equilibrio, junto al gate de dominancia, no como validacion.

### 2.3 Nulo: por que la permutacion de etiquetas es vacua

`d` es funcion solo de `(n_1, ..., n_K)`. Permutar etiquetas no cambia ningun
`n_k`, asi que `d*` es identicamente igual a `d`: el nulo por permutacion
tiene varianza cero y un p-valor sin sentido. Esto se declara en la salida con
`label_permutation_invariant = True`.

El nulo que si informa es de **composicion**: si los tamanos fueran una
composicion aleatoria sin estructura,

```
(p_1, ..., p_K) ~ Dirichlet(1, ..., 1)     (uniforme sobre el simplex)
d*_b = H(p_b) / ln K
```

y se comparan los cuantiles. Es un nulo deliberadamente debil ("no hay
preferencia por ninguna clase") y su papel es detectar el extremo opuesto al
gate: particiones **demasiado** balanceadas para el corpus que las produce.

### 2.4 Bootstrap

Se remuestrean UCEs con reemplazo y se recalcula `H/ln K` **manteniendo `K`
fijo en el denominador**. Si se dejara que `K` cambie en cada remuestra, la
entropia normalizada subiria por la desaparicion de clases raras y el IC
estaria sesgado al alza. Con `K` fijo el IC refleja solo la incertidumbre de
las proporciones.

Advertencia de comparabilidad: `d` esta normalizada por `ln K`, de modo que
dos particiones con distinto `K` no son directamente comparables. Para
comparar se usa `K_eff` (que esta en unidades de clases) y se reporta `K`.

---

## 3. `RMSD`: distancia de proporciones entre particiones

### 3.1 Definicion

Dadas dos particiones de las **mismas** UCEs, `ref` (referencia declarada) y
`obs` (observada), se restringe a `U = {u : ref_u >= 0 y obs_u >= 0}`, con
`n = |U|`. Sea `O` la matriz de confusion `k_ref x k_obs` con
`O_ij = #{u in U : ref_u = i, obs_u = j}`, `r_i` las sumas de fila y `c_j` las
de columna.

Emparejamiento Hungarian que maximiza el solape:

```
pi* = argmax_pi sum_i O_{i, pi(i)}
```

Pares de proporciones (las clases sin pareja entran con 0):

```
emparejadas:   (r_i/n, c_{pi*(i)}/n)
ref sin pareja: (r_i/n, 0)
obs sin pareja: (0, c_j/n)
```

y

```
RMSD = sqrt( (1/L) * sum_{l=1..L} (p_l - q_l)^2 )    en [0, 1]
```

con `L = max(k_ref, k_obs)` pares. `RMSD = 0` si y solo si las proporciones de
las clases emparejadas coinciden exactamente y `k_ref = k_obs`.

Se reportan `n`, `k_ref`, `k_obs`, `k_union` (pares usados) y `n_unmatched`
(clases sin pareja, que es lo que penaliza la diferencia de `K`).

### 3.2 Que mide y que no

Mide **concordancia de perfil de tamanos** tras alinear clases. **No mide
acuerdo de asignacion**: un RMSD bajo solo dice que las dos particiones
reparten el corpus en bloques de tamanos parecidos. El emparejamiento
Hungarian esta elegido justamente para maximizar el solape, lo que hace que el
RMSD sea un estadistico **conservador**: tiende a dar valores bajos aunque las
particiones no tengan nada que ver.

Esto no es una hipotesis, esta medido. En el bench, contra una referencia
declarada (`kmeans_k7` sobre el mismo embedding):

```
RMSD = 0.0478   IC95 [0.0327, 0.0712]
nulo (permutacion de etiquetas) = 0.0489   sd = 0.0126   q95 = 0.0683   p = 0.545
control: particion ALEATORIA con el mismo vector de tamanos = 0.0393
```

El valor observado es **indistinguible del nulo** (`p = 0.545`) y, sobre
todo, **mayor que el de una particion aleatoria con el mismo perfil**
(0.0478 > 0.0393). Si el estadistico midiera acuerdo, el azar no podria
puntuar mejor que la particion real. Conclusion operativa: **el RMSD mide
desajuste de perfiles de tamano, no acuerdo**, y no puede reportarse como
evidencia de validez por si solo. Va siempre acompanado de
`cross_partition_agreement` (FMI/NMI/ARI), que si mide acuerdo a nivel de
UCE. El RMSD aporta lo que el ARI no ve: cuanto se parecen (o no) los
repartos de masa entre las dos particiones.

### 3.3 Nulo e IC

- **Nulo**: permutar las etiquetas observadas entre las UCEs comunes. Es
  valido en el sentido debil de que el estadistico **si** cambia (el
  emparejamiento Hungarian depende de la matriz de confusion): en el bench el
  nulo recorre `[0.019, 0.078]` con `sd = 0.0126`. Lo que **no** es es un test
  de asociacion con potencia: como el estadistico esta dominado por los
  marginales, un azar con el mismo perfil puede puntuar mejor que la
  particion real (seccion 3.2).
- **IC bootstrap**: remuestrear UCEs comunes y **recalcular el emparejamiento**
  en cada remuestra (no reutilizar `pi*`), porque el matching es parte del
  estimador.
- **Lectura correcta del p-valor**: `p >= 0.05` significa "no hay evidencia
  de que el acuerdo de perfiles supere el azar". NO significa "hay
  desacuerdo". Y `p < 0.05` no convierte al RMSD en medida de validez: solo
  dice que los perfiles se alinean mejor de lo esperable por azar, lo que
  puede ocurrir por construccion cuando `k_ref = k_obs`.

Diagnostico recomendado (implementado en el self-test): calcular tambien
`RMSD(ref, aleatoria_con_el_mismo_perfil)`. Si ese valor es menor o igual
que el observado, el estadistico no distingue acuerdo de perfil y hay que
decirlo en el informe.

### 3.4 El problema de la referencia

`RMSD` necesita una segunda particion. `reference_id` es **obligatorio** en la
firma para que sea imposible reportar el numero sin decir contra que se
compara:

- referencia = estandar humano o codificacion manual -> el RMSD es parte de la
  **validez externa** (junto a `C_ext`);
- referencia = otra salida del pipeline (otro metodo, otro `K`, otra semilla)
  -> es **concordancia entre metodos**, no validez;
- referencia = particion derivada del mismo dato por otro algoritmo (como en
  el self-test) -> es una **ilustracion del calculo**, no evidencia.

El paper promete RMSD como validez externa: solo puede escribirse asi cuando
la referencia es externa. En los demas casos la frase correcta es
"concordancia de proporciones entre X e Y".

---

## 4. Familias bibliograficas (sin citas concretas)

| metrica | familia de metodo                                                                                                          | indices/herramientas estandar                                                                                                                             |
| ------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C_in`  | validacion interna de clustering por cohesion: similitud media de cada objeto a su centroide (o a sus companeros de clase) | coseno al centroide, coherencia intra-cluster tipo silhouette sobre coseno, validacion por modelo nulo / permutacion de etiquetas                         |
| `d`     | numeros efectivos de diversidad y entropias de particion                                                                   | entropia de Shannon normalizada, indice de Simpson (Gini-Simpson), numeros de Hill `exp(H)`, nulo Dirichlet sobre el simplex                              |
| `RMSD`  | distancias entre vectores de composicion, con alineamiento de etiquetas por asignacion optima                              | RMSD de proporciones, emparejamiento biparticion de peso maximo (Hungarian / Kuhn-Munkres), kappa/accuracy alineada de la familia de `external_coherence` |

Ninguna de las tres es nueva como metodo; lo que se aporta es (i) el espacio
declarado, (ii) el nulo explicito y (iii) el reporte conjunto con IC. Eso es
lo que hay que defender en la revision, no la novedad del indice.

---

## 5. Diff exacto para `src/reii/quality/metrics.py`

### 5.1 Cambio de imports

```diff
--- a/src/reii/quality/metrics.py
+++ b/src/reii/quality/metrics.py
@@ -6,1 +6,1 @@
-from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
+from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
```

`numpy`, `scipy.sparse` y `linear_sum_assignment` ya estan importados; no se
anade ninguna dependencia.

### 5.2 Bloque a anadir al final del archivo

Se inserta **despues** de `external_coherence` (final del modulo). No se
modifica ninguna funcion existente.

```python
# ── Coherencia interna, diversidad de particion y RMSD de proporciones ──────


def _row_normalize(X):
    """Normaliza filas a norma unitaria (densa o dispersa), respetando ceros."""
    if sparse.issparse(X):
        X = X.tocsr().astype(float)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        return sparse.diags(1.0 / np.where(norms > 0, norms, 1.0)) @ X
    X = np.asarray(X, float)
    norms = np.linalg.norm(X, axis=1)
    return X / np.where(norms > 0, norms, 1.0)[:, None]


def _cos_to_centroid(Xn, y, classes):
    """Coseno al centroide de cada clase: por clase y por UCE, y n negativos."""
    per_class: List[Tuple[int, float]] = []
    per_item: List[float] = []
    n_neg = 0
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0:
            continue
        mu = np.asarray(Xn[idx].mean(axis=0)).ravel()
        norm = float(np.linalg.norm(mu))
        cos = (
            np.zeros(len(idx))
            if norm <= 0
            else np.asarray(Xn[idx] @ (mu / norm)).ravel()
        )
        n_neg += int((cos < 0.0).sum())
        cos = np.clip(cos, 0.0, 1.0)
        per_class.append((len(idx), float(cos.mean())))
        per_item.extend(cos.tolist())
    return per_class, np.asarray(per_item, float), n_neg


def internal_coherence(
    matrix,
    labels: Sequence[int],
    space_id: str,
    metric: str = "cosine",
    max_pairwise: int = 1500,
    n_boot: int = 500,
    n_perm: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """C_in: homogeneidad intra-clase medida en un espacio DECLARADO.

    ``space_id`` es obligatorio: un C_in no significa nada sin decir en que
    espacio se midio (embedding de 384d, perfiles chi2, tf-idf...). La funcion
    no adivina el espacio: lo recibe y lo devuelve en el resultado.

    Para cada clase ``C_k`` con ``n_k`` elementos y centroide normalizado
    ``mu_k``, el score de la UCE ``i`` es ``max(0, cos(x_i, mu_k))`` (truncado
    a [0,1]: un coseno negativo es anti-similitud y se reporta aparte en
    ``frac_negative``). Se devuelven tres versiones:

    ``cin``           media MACRO de las medias por clase (la del paper).
    ``cin_weighted``  ponderada por tamano de clase.
    ``cin_pairwise``  media de similitudes pairwise intra-clase (control del
                      sesgo del centroide hacia arriba para clases pequenas).

    En d alta el valor absoluto no es interpretable: se anade ``excess =
    cin - null_mean``, donde el nulo permuta las etiquetas (preserva tamanos
    de clase y espacio, rompe la asociacion UCE-clase). Reportar ``cin`` sin
    ``excess`` no distingue "clases coherentes" de "corpus homogeneo".

    ``ci_low``/``ci_high`` es el IC bootstrap de la media macro (sobre las K
    medias de clase); ``ci_low_weighted``/``ci_high_weighted`` lo es de la
    ponderada (sobre las n UCEs). Con K chico el IC macro es conservador por
    construccion: se remuestrean K valores.

    Ignora etiquetas < 0. Devuelve NaN (con ``n`` y ``k`` poblados) si hay
    menos de 3 UCEs o menos de 2 clases.
    """
    from scipy import sparse  # noqa: F401  (import local: ver _row_normalize)

    y = np.asarray(list(labels)).astype(int)
    X_full = matrix if hasattr(matrix, "shape") else np.asarray(matrix, float)
    if X_full.shape[0] != len(y):
        raise ValueError("matrix y labels deben tener el mismo numero de filas")
    keep = y >= 0
    X = X_full[keep] if sparse.issparse(X_full) else np.asarray(X_full, float)[keep]
    y = y[keep]
    n = int(len(y))
    classes = np.unique(y)
    k = int(len(classes))
    out: Dict[str, Any] = {
        "cin": float("nan"),
        "cin_weighted": float("nan"),
        "cin_pairwise": float("nan"),
        "space_id": space_id,
        "metric": metric,
        "n": n,
        "k": k,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "ci_low_weighted": float("nan"),
        "ci_high_weighted": float("nan"),
        "frac_negative": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
        "excess": float("nan"),
        "z_score": float("nan"),
        "p_value": float("nan"),
    }
    if n < 3 or k < 2:
        return out
    Xn = _row_normalize(X)
    per_class, per_item, n_neg = _cos_to_centroid(Xn, y, classes)
    if not per_class:
        return out
    coin = np.array([m for _, m in per_class])
    counts = np.array([c for c, _ in per_class], float)
    cin = float(coin.mean())
    out["cin"] = cin
    out["cin_weighted"] = float((counts * coin).sum() / counts.sum())
    out["frac_negative"] = n_neg / float(n)
    out["ci_low"], out["ci_high"] = bootstrap_ci(
        coin, np.mean, n_boot=n_boot, seed=seed
    )[1:]
    out["ci_low_weighted"], out["ci_high_weighted"] = bootstrap_ci(
        per_item, np.mean, n_boot=n_boot, seed=seed
    )[1:]
    # pairwise: O(n_c^2) por clase, con tope por clase
    pair = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        if len(idx) < 2:
            continue
        if len(idx) > max_pairwise:
            idx = np.random.default_rng(seed).choice(idx, max_pairwise, replace=False)
        gram = Xn[idx] @ Xn[idx].T
        sim = np.asarray(gram.todense()) if sparse.issparse(gram) else np.asarray(gram)
        iu = np.triu_indices(len(idx), 1)
        pair.append(float(np.clip(sim[iu], 0.0, 1.0).mean()))
    if pair:
        out["cin_pairwise"] = float(np.mean(pair))
    # nulo: permutacion de etiquetas (rompe la asociacion UCE-clase)
    rng = np.random.default_rng(seed)
    nulls = np.empty(n_perm)
    for i in range(n_perm):
        pc, _, _ = _cos_to_centroid(Xn, rng.permutation(y), classes)
        nulls[i] = float(np.mean([m for _, m in pc]))
    std = float(nulls.std(ddof=1))
    out["null_mean"] = float(nulls.mean())
    out["null_std"] = std
    out["excess"] = cin - out["null_mean"]
    out["z_score"] = (cin - out["null_mean"]) / std if std > 0 else float("nan")
    out["p_value"] = float((nulls >= cin).mean())
    return out


def _entropy_stats(y, k_ref: Optional[int] = None) -> Dict[str, float]:
    """Entropia y tamanos de una particion; ``k_ref`` fija K en el denominador."""
    counts = np.bincount(y)
    counts = counts[counts > 0]
    n = int(counts.sum())
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    k = int(len(p))
    k_norm = int(k_ref) if k_ref else k
    return {
        "n": n,
        "k": k,
        "entropy": h,
        "entropy_norm": h / np.log(k_norm) if k_norm > 1 else float("nan"),
        "simpson": float(1.0 - (p * p).sum()),
        "effective_k": float(np.exp(h)),
        "max_share": float(p.max()),
    }


def partition_diversity(
    labels: Sequence[int], n_boot: int = 500, n_null: int = 500, seed: int = 42
) -> Dict[str, Any]:
    """d: diversidad de la particion (entropia normalizada, Simpson, Hill).

    Propiedad de la PARTICION, no del contenido: solo depende del vector de
    tamanos de clase. NO es ``diversidad_semantica`` (la metrica por UCE del
    pipeline, ~0.82): aquella mide dispersion lexica dentro de una UCE, esta
    mide equilibrio entre clases.

    ``entropy``       H en nats.
    ``entropy_norm``  H / ln K en [0,1] (la ``d`` que pide el paper).
    ``simpson``       1 - sum p_k^2 (Gini-Simpson, en [0,1)).
    ``effective_k``   exp(H): "numero efectivo de clases" (Hill).
    ``max_share``     max p_k (coincide con el gate de dominancia sobre lo
                      asignado cuando no hay ruido).

    Nulo: la permutacion de etiquetas es VACUA aqui (invariante: permutar no
    cambia ningun p_k), y se declara con ``label_permutation_invariant``. El
    nulo que se reporta es de composicion, Dirichlet(1) con el mismo K: mide
    si la particion esta mas (o menos) balanceada de lo que daria un reparto
    aleatorio uniforme. El p-valor es bilateral.

    ``ci_low``/``ci_high`` es el IC bootstrap de ``entropy_norm`` resampleando
    UCEs con K FIJO en el denominador (si K variara en cada remuestra, el IC
    estaria sesgado al alza por desaparicion de clases raras).

    ``entropy_norm`` no es comparable entre particiones con distinto K
    (normaliza por ln K); para comparar se usa ``effective_k``.
    """
    y = np.asarray(list(labels)).astype(int)
    y = y[y >= 0]
    base = _entropy_stats(y) if len(y) else None
    out: Dict[str, Any] = (
        dict(base)
        if base
        else {
            "n": 0,
            "k": 0,
            "entropy": float("nan"),
            "entropy_norm": float("nan"),
            "simpson": float("nan"),
            "effective_k": float("nan"),
            "max_share": float("nan"),
        }
    )
    out.update(
        {
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "null_mean": float("nan"),
            "null_q05": float("nan"),
            "null_q95": float("nan"),
            "p_value": float("nan"),
            "label_permutation_invariant": True,
            "space_id": "cluster_id",
        }
    )
    if base is None or base["k"] < 2:
        return out
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = _entropy_stats(
            rng.choice(y, size=len(y), replace=True), k_ref=base["k"]
        )["entropy_norm"]
    out["ci_low"] = float(np.quantile(boots, 0.025))
    out["ci_high"] = float(np.quantile(boots, 0.975))
    comp = rng.dirichlet(np.ones(base["k"]), size=n_null)
    h = -(comp * np.log(comp)).sum(axis=1) / np.log(base["k"])
    out["null_mean"] = float(h.mean())
    out["null_q05"] = float(np.quantile(h, 0.05))
    out["null_q95"] = float(np.quantile(h, 0.95))
    p_lo = float((h <= base["entropy_norm"]).mean())
    p_hi = float((h >= base["entropy_norm"]).mean())
    out["p_value"] = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    return out


def _rmsd_core(ra: np.ndarray, ob: np.ndarray) -> Dict[str, float]:
    """RMSD de proporciones con emparejamiento Hungarian de clases."""
    ru, ri = np.unique(ra, return_inverse=True)
    ou, oi = np.unique(ob, return_inverse=True)
    n = len(ra)
    ov = np.zeros((len(ru), len(ou)))
    np.add.at(ov, (ri, oi), 1.0)
    row, col = ov.sum(axis=1), ov.sum(axis=0)
    r, c = linear_sum_assignment(-ov)
    pairs = [(row[i] / n, col[j] / n) for i, j in zip(r, c)]
    used_r, used_c = set(r.tolist()), set(c.tolist())
    pairs += [(row[i] / n, 0.0) for i in range(len(ru)) if i not in used_r]
    pairs += [(0.0, col[j] / n) for j in range(len(ou)) if j not in used_c]
    pr = np.array([p for p, _ in pairs])
    po = np.array([q for _, q in pairs])
    return {
        "rmsd": float(np.sqrt(np.mean((pr - po) ** 2))) if len(pr) else float("nan"),
        "k_ref": int(len(ru)),
        "k_obs": int(len(ou)),
        "k_union": int(len(pr)),
        "n_unmatched": int(len(pr) - len(r)),
    }


def rmsd_proportions(
    labels_ref: Mapping[str, int],
    labels_obs: Mapping[str, int],
    reference_id: str,
    n_boot: int = 300,
    n_null: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """RMSD entre los vectores de proporcion de clase de dos particiones.

    Se usan las UCEs comunes con etiqueta >= 0 en ambas. Las clases se
    emparejan con Hungarian maximizando el solape de la matriz de confusion, de
    modo que el estadistico es invariante a la numeracion de las clases y
    conservador: el emparejamiento se elige para que las proporciones se
    parezcan lo mas posible.

    ``reference_id`` es OBLIGATORIO y se devuelve en la salida. Sin el, el
    numero no es interpretable:

    * referencia externa (codificacion humana) -> parte de la validez externa;
    * otra salida del pipeline -> concordancia entre metodos, NO validez;
    * particion derivada del mismo dato -> ilustracion del calculo.

    ADVERTENCIA (medida, no teorica): esto mide concordancia de PERFIL de
    tamanos, NO de asignacion. El emparejamiento maximiza el solape, de modo
    que el estadistico queda dominado por los marginales: en el self-test,
    una particion ALEATORIA con el mismo vector de tamanos da RMSD menor que
    la particion real. El nulo por permutacion de etiquetas es valido (el
    emparejamiento depende de la asociacion) pero su dispersion es chica, y
    un p-valor alto significa "no hay evidencia de acuerdo", no "hay
    desacuerdo". Para acuerdo a nivel de UCE usar ``cross_partition_agreement``
    (FMI/NMI/ARI). Recomendacion: reportar el RMSD como DESCRIPTOR de
    perfiles y nunca como evidencia de validez.

    ``k_union`` son los pares usados y ``n_unmatched`` las clases sin pareja
    (penalizan que las dos particiones tengan distinto K). El IC bootstrap
    remuestrea UCEs y RECALCULA el emparejamiento en cada remuestra.
    """
    common = [
        u
        for u in sorted(set(labels_ref) & set(labels_obs))
        if labels_ref[u] is not None
        and labels_obs[u] is not None
        and int(labels_ref[u]) >= 0
        and int(labels_obs[u]) >= 0
    ]
    out: Dict[str, Any] = {
        "rmsd": float("nan"),
        "n": len(common),
        "k_ref": 0,
        "k_obs": 0,
        "k_union": 0,
        "n_unmatched": 0,
        "reference_id": reference_id,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "null_mean": float("nan"),
        "null_std": float("nan"),
        "null_q05": float("nan"),
        "null_q95": float("nan"),
        "p_value": float("nan"),
    }
    if len(common) < 2:
        return out
    ra = np.array([int(labels_ref[u]) for u in common])
    ob = np.array([int(labels_obs[u]) for u in common])
    out.update(_rmsd_core(ra, ob))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(common), len(common))
        boots[i] = _rmsd_core(ra[idx], ob[idx])["rmsd"]
    out["ci_low"] = float(np.quantile(boots, 0.025))
    out["ci_high"] = float(np.quantile(boots, 0.975))
    nulls = np.empty(n_null)
    for i in range(n_null):
        nulls[i] = _rmsd_core(ra, rng.permutation(ob))["rmsd"]
    out["null_mean"] = float(nulls.mean())
    out["null_std"] = float(nulls.std(ddof=1))
    out["null_q05"] = float(np.quantile(nulls, 0.05))
    out["null_q95"] = float(np.quantile(nulls, 0.95))
    out["p_value"] = float((nulls >= out["rmsd"]).mean())
    return out
```

Nota: dentro de `internal_coherence` el `from scipy import sparse` local es
redundante en `metrics.py` (ya esta a nivel de modulo) y puede eliminarse; se
deja porque el prototipo del playground lo necesita.

### 5.3 Que NO se toca

- `silhouette_labels`, `cross_partition_agreement`, `external_coherence`,
  `bootstrap_ci`: sin cambios. Las tres funciones nuevas usan `bootstrap_ci`.
- No se anaden dependencias: `numpy`, `scipy.sparse`,
  `linear_sum_assignment` y `sklearn` ya estan.
- No hay cambios en `main_workflow.py`: son funciones puras. El cableado al
  informe queda como tarea aparte (seccion 7).

---

## 6. Resultados ilustrativos del self-test

`python scripts/playground/gate_calibration.py --metrics-selftest` sobre
`data/workflow_data.json` (**bench `uce=26`**, 600 UCEs de las cuales 235
tienen `embedding`; **no es la corrida en curso**). Referencia del RMSD: una
particion `kmeans` con el mismo `K` sobre el mismo espacio, construida dentro
del script porque ese bench no trae `projected_cluster_id`.

```
C_in  (coseno al centroide de clase, macro, clip a [0,1])
  espacio=uce_embedding_384d  n=235  k=7
  C_in=0.8165   IC95=[0.8113, 0.8218]
  ponderado=0.8152  IC95=[0.8102, 0.8207]
  pairwise=0.6550  cosenos negativos truncados=0.0000
  nulo(permutacion de etiquetas)=0.7950  exceso=+0.0216  z=22.55  p=0.000
  [ok] C_in supera el nulo de etiquetas permutadas

d   (diversidad de la particion; NO es diversidad semantica por UCE)
  n=235  k=7  H/lnK=0.9702  IC95=[0.9381, 0.9830]
  Simpson=0.8411  K_eff(Hill)=6.61  max_share=0.2255
  nulo(composicion Dirichlet(1))=0.8204  q05=0.6638  q95=0.9346  p=0.013

```

RMSD (proporciones; referencia declarada = particion k-means)
referencia=kmeans_k7_sobre_uce_embedding_384d
n=235 k_ref=7 k_obs=7 k_union=7
RMSD=0.0478 IC95=[0.0327, 0.0712]
nulo(permutacion de etiquetas)=0.0489 sd=0.0126 q95=0.0683 p=0.545
[!] RMSD indistinguible del nulo: proporciones parecidas NO son evidencia
de acuerdo entre particiones
control: RMSD contra particion ALEATORIA con el mismo perfil = 0.0393
(observado 0.0478)
[!] el azar con el mismo perfil da RMSD igual o menor: el estadistico no
distingue acuerdo de perfil

```

### 6.1 Como se lee cada numero

- **`C_in = 0.8165` con nulo `0.7950`**: el valor bruto parece alto pero 0.795
  es gratis (embedding denso de 384d). El unico numero que dice algo es
  `excess = +0.0216` con `z = 22.55`. Es un efecto **real pero pequeno**: las
  clases ordenan el espacio mejor que el azar, no mucho mejor. Asi hay que
  escribirlo. Nota metodologica: con `n = 235` y `B = 300` permutaciones, el
  p-valor esta limitado por `1/(B+1)`; `p = 0.000` significa `p < 0.003`.
- **`C_in` vs `C_in^pair` (0.82 vs 0.655)**: el centroide regala ~0.16. Si se
  reporta solo el macro, un revisor puede atribuirlo al sesgo del centroide;
  por eso se reportan ambos.
- **`d = 0.9702` con `K_eff = 6.61` para `K = 7`**: la particion esta casi
  perfectamente balanceada, y `p = 0.013` indica que esta **mas** balanceada
  que una composicion aleatoria. No es una buena noticia automaticamente: un
  `d` alto puede ser un clustering que reparte el corpus en bloques de tamano
  parecido sin respetar la estructura. Se reporta como descripcion.
* **`RMSD = 0.0478` con `p = 0.545` y control 0.0393**: no se distingue del
  nulo y el azar con el mismo perfil puntua mejor. Es el resultado mas
  importante del self-test: confirma que el RMSD **no** puede venderse como
  evidencia de validez. Va como tabla descriptiva, junto a FMI/NMI/ARI, y con
  la frase "concordancia de perfiles de tamano".

### 6.2 Costo

`C_in` con `n_perm = 300` y `n_boot = 300` sobre 235 x 384: ~4 s. `d` ~0.03 s.
`RMSD` con 200/200 remuestreos: ~2.6 s. Total del self-test ~27 s (incluye la
lectura del JSON de 29 MB), dentro del limite de 60 s.

---

## 7. Como se reportan en el paper

Tabla propuesta (una fila por corrida y metrica):

| metrica         | valor  | IC95             | nulo               | exceso  | n   | k     | espacio / referencia  |
| --------------- | ------ | ---------------- | ------------------ | ------- | --- | ----- | --------------------- |
| `C_in` macro    | 0.8165 | [0.8113, 0.8218] | 0.7950             | +0.0216 | 235 | 7     | `uce_embedding_384d`  |
| `C_in` pairwise | 0.6550 | -                | -                  | -       | 235 | 7     | `uce_embedding_384d`  |
| `d` (`H/ln K`)  | 0.9702 | [0.9381, 0.9830] | 0.8204 (Dirichlet) | +0.1498 | 235 | 7     | `cluster_id`          |
| `K_eff`         | 6.61   | -                | -                  | -       | 235 | 7     | `cluster_id`          |
| `RMSD` | 0.0478 | [0.0327, 0.0712] | 0.0489 (sd 0.0126) | -0.0011 | 235 | 7 / 7 | `kmeans_k7` (interna) |

Reglas de redaccion:

1. Ninguna metrica se reporta sin `n`, `K` y espacio/referencia.
2. `C_in` se reporta con `excess` y `p`; sin eso no es interpretable.
3. `d` va con `K_eff`, nunca sola.
4. `RMSD` va con `reference_id` y con el nulo; si `p >= 0.05`, se dice
   explicitamente que no aporta evidencia.
5. `RMSD` solo se llama "validez externa" si la referencia es externa al
   pipeline.

### 7.1 Cableado pendiente (fuera de este diff)

Las tres funciones son puras y no se llaman desde `main_workflow.py`. Para
que aparezcan en el informe hace falta un paso posterior:

- `C_in`: `internal_coherence(embeddings_assigned, labels, "uce_embedding_384d")`
  donde hoy se llama `silhouette_labels` (misma matriz, mismo filtro `>= 0`);
- `d`: `partition_diversity(labels)` en la seccion de tamanos de clase, junto
  al gate de dominancia;
- `RMSD`: `rmsd_proportions(labels_ref, labels_obs, reference_id)` solo cuando
  exista una referencia declarada (codificacion humana o `K` alternativo).

No se incluye el diff de cableado porque depende de decisiones de informe que
no estan tomadas y porque la corrida en curso no debe tocarse.

---

## 8. Limitaciones que un revisor vera

1. **El nulo de `C_in` es condicional al espacio.** Permutar etiquetas fija
   `X`, de modo que si el embedding ya esta "alisado" (vectores con coseno
   alto entre si), el nulo sube y el `excess` baja. No es un defecto: es la
   respuesta correcta a "¿aportan las clases algo sobre este espacio?", pero
   no responde "¿es este el espacio correcto?".
2. **`C_in` premia clases esfericas.** Un coseno alto al centroide no distingue
   "una clase coherente" de "una clase alargada en una direccion". El
   `cin_pairwise` mitiga, no resuelve.
3. **`d` es invariante a la permutacion de etiquetas y a la composicion
   interna.** No dice nada sobre si las clases son correctas; solo sobre su
   equilibrio. No puede sustituir a `C_ext` ni al gate.
4. **`RMSD` no mide acuerdo**: es un descriptor de perfiles, el Hungarian
   maximiza el solape y en el self-test el azar con el mismo perfil puntua
   mejor que la particion real. Reportarlo como "validez" seria un error de
   interpretacion; reportarlo como "concordancia de perfiles" es correcto y
   util cuando se compara el mismo metodo con distinto `K` o distinta semilla.
5. **El IC bootstrap de `C_in` macro remuestrea `K` valores.** Con `K = 7` el
   IC es aproximado y conservador; el IC ponderado (sobre `n` UCEs) es el que
   tiene las propiedades asintoticas habituales. Se reportan los dos.
```
