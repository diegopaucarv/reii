reii: niveles de calidad, pruebas y persistencia
Diseño propuesto a partir de la lectura de main_workflow.py y dashboard.py (diegopaucarv/reii). No se ejecutó el pipeline ni se vieron los datos: todo lo cuantitativo es una propuesta a calibrar.

1. Objetivo y principios
Objetivo: ofrecer 2–3 resultados de clasificación según un umbral de calidad, ajustables en vivo, sin Revisa repositorio GitHub diegopaucarv/reii. Ve al src y analiza el workflow (no el clasico), que es un archivo py extenso. necesito que identifiques las diferencias que impactan en la estimacion de tasa de retención de uces y cobertura de vocabulario.
luego lee el archivo dashboard.py y encuentra los métodos que me permiten calcular estos resultados.
compara estos métodos con Los originales de alceste. luego haz un reporte que me diga por qué estoy obteniendo con los mismos datos tasas más reducidas de cobertura (30%) y retención (50%).
<alceste>
1. Síntesis Conceptual Abductiva (Nueva Formulación)
   La estimación de la cobertura de vocabulario y de la tasa de retención de Unidades de Contexto Elemental (UCE) en el método ALCESTE (Analyse des Lexèmes Cooccurrents dans un Ensemble de Segments de Texte) de Max Reinert no constituye un mero filtrado de superficie, sino un procedimiento estadístico-léxico de validación de estabilidad inductiva [1-3]. El algoritmo evalúa la calidad matemática de los datos textuales mediante la reducción de la hiper-dimensionalidad del vocabulario y la verificación cruzada de la congruencia entre agrupamientos [4-6].
   +-------------------------------------------------------------------------------------------------+
   | FLUSO DE ESTIMACIÓN Y CLASIFICACIÓN EN ALCESTE |
   +-------------------------------------------------------------------------------------------------+
   | CORPUS TEXTUAL BRUTO (UCI) |
   | └── Segmentación Morfosintáctica y Lematización (SpaCy / Dictionaries) [7, 8] |
   | |
   | 1. ESTIMACIÓN DE COBERTURA DE VOCABULARIO |
   | ├── Filtrado Categórico: Formas Activas (N, V, Adj, Adv) vs. Suplementarias [3, 7, 9] |
   | ├── Umbral de Frecuencia $TSJ \ge 3$ o $4$ & Eliminación de Hapax Legomena [3, 10, 11] |
   | └── Métrica TTR y Ajuste a Curva Zipf / Entropía Léxica [3, 12, 13] |
   | |
   | 2. TASA DE RETENCIÓN DE UCEs (DOBLE CLASIFICACIÓN) |
   | ├── Construcción de UCEs y UCs (Ventana Umbral 1 vs. Ventana Umbral 2) [2, 3, 14] |
   | ├── Matriz Dispersa Binarizada $U \times V$ (Presencia/Ausencia) con $\epsilon = 0.1$ [4, 6] |
   | ├── CDH en Paralelo sobre Ambas Particiones (Maximización $\chi^2$ e Inercia) [5, 15, 16] |
   | ├── Matriz de Solapamiento e Intersección Húngara (Algoritmo Munkres) [17, 18] |
   | └── Regla de Estabilidad: Retención de UCEs Congruentes ($CR$) vs. Descarte [2, 18, 19] |
   +-------------------------------------------------------------------------------------------------+
   A. Estimación de la Cobertura y Riqueza del Vocabulario
   La cobertura del vocabulario operacionaliza el grado de diversidad y suficiencia léxica necesario para construir tablas de contingencia estables (\\(U \times V\\)) [4, 7, 20]. Reinert estructuró esta fase a través de tres operaciones de calibración léxica:
1. Partición Categórica y Filtrado Gramatical: ALCESTE distingue entre formas activas (analysables) y formas suplementarias (mots-outils) [3, 7, 21]. Las formas activas —comprendidas por sustantivos, verbos (excluyendo auxiliares y modales), adjetivos, adverbes de modo y palabras no reconocidas de alta frecuencia— son las únicas que integran la matriz binaria de coocurrencia [3, 7, 9]. Las formas suplementarias (preposiciones, pronombres, artículos, números) se excluyen de la construcción de las clases, aunque se conservan para describir el perfil discursivo ulterior [7, 9, 21].
1. Control de Frecuencia Mínima (\\(TSJ\\)) y Truncamiento de Hapax: El sistema establece un umbral de frecuencia mínima (\\(TSJ\\), típicamente \\(TSJ \ge 3\\) o \\(4\\)) para filtrar la masa de términos infrecuentes [3, 10, 16]. Aunque los hapax legomena (palabras de una sola aparición) representan habitualmente más del 50% de las formas distintas del corpus, ALCESTE los margina de la matriz primaria para prevenir distorsiones causadas por ruido informativo aleatorio [3, 10, 11]. No obstante, Reinert demostró experimentalmente que elevar en exceso el parámetro \\(TSJ\\) degrada velozmente la inercia extraída por la clasificación, ya que la acumulación total de vocabulario de baja frecuencia resulta indispensable para la diferenciación léxica [10].
1. Métricas Léxicas Integradas: La riqueza del vocabulario analizado se expresa mediante el Type-Token Ratio (TTR) normalizado —proporción entre formas reducidas únicas y ocurrencias totales— y la pendiente logarítmica de la curva de Zipf, que evalúa la velocidad de incorporación de léxico nuevo [3, 12, 13]. La tasa de riqueza del vocabulario indica el porcentaje final de lemas activos retenidos para el cálculo multivariado tras la aplicación del filtro gramatical y el umbral \\(TSJ\\) [3, 22].
   B. Métrica y Mecanismo de la Tasa de Retención de UCEs
   La tasa de retención o cobertura de UCEs (\\(CR\\)) define el porcentaje de Unidades de Contexto Elemental que el algoritmo logra clasificar con éxito en las clases terminales estables, desestimando aquellas unidades cuya asignación es oscilante [2, 18, 19].
1. Segmentación jerárquica de la unidad estadística: El texto original (Unidades de Contexto Iniciales o UCI) es fragmentado en Segments de Texte Calibrés (STC) según marcas de puntuación y longitud en caracteres [2, 23, 24]. Los STC contiguos se concatenan para estructurar las UCEs (equivalentes aproximados a frases o cláusulas de 15 a 100 ocurrencias) [2, 24, 25]. A su vez, las UCEs se agrupan en Unidades de Contexto (UC) de mayor extensión para efectuar el análisis [2, 3, 14].
1. El protocolo de Doble Clasificación Descendente Jerárquica (DHC/CDH): Para erradicar artefactos metodológicos derivados del tamaño arbitrario del corte de ventana, ALCESTE ejecuta dos clasificaciones jerárquicas independientes sobre las UCEs utilizando dos configuraciones distintas de tamaño de UC (por ejemplo, solicitando un mínimo de 10 palabras analizadas por UC en la primera corrida y 14 en la segunda) [2, 3, 14].
1. Prueba de Estabilidad e Intersección Combinatoria: Sobre las dos particiones obtenidas, se construye una matriz de contingencia o solapamiento que cruza las clases de la primera clasificación con las de la segunda [2, 15, 17]. Mediante la aplicación del algoritmo de asignación Húngaro (Munkres) sobre la matriz de similitud de \\(\chi^2\\) o de conteo de UCEs compartidas, el sistema identifica el mapeo biunívoco que maximiza la asociación inter-clase [17, 18].
1. Regla Estricta de Retención (\\(CR\\)): Una UCE es declarada estable y retribuida a una clase definitiva si y solo si es clasificada en la misma clase lógica (bajo el mapeo húngaro) en ambas corridas independientes [2, 18, 19]. Aquellas UCEs que permutan de clase por efecto de la variación del tamaño de ventana son rechazadas [2, 18, 26]. La tasa de retención se formaliza como:
   \\[\text{Tasa de Retención de UCEs } (CR) = \frac{\text{UCEs Estables Retenidas}}{\text{UCEs Totales del Corpus}} \times 100\%\\]
   En corpus estructurados y homogéneos, la tasa de retención oscila entre el 65% y el 85%, mientras que en discursos fragmentarios o lábiles la tasa puede descender, señalando la inestabilidad de los "mundos lexicales" subyacentes [2, 25, 27].
1. Hiperoptimización Algorítmica Contemporánea: En arquitecturas modernas, el control de la retención de UCEs se refina aplicando optimizadores de búsqueda de patrones (PatternSearch) que ajustan simultáneamente el umbral de similitud entre UCEs, el coeficiente de contingencia \\(CTEST\\), un pseudocuento \\(\epsilon = 0.1\\) para estabilizar divisiones en matrices dispersas y el Índice de Rand Ajustado (ARI) entre umbrales [6, 10, 28, 29].
1. Validación de la Síntesis: Estrés en Ejes Antipodales
   • Fricción Empírica (Micro-realidad vs. Latencia Estructural Macro):
   La estimación del vocabulario y la retención de UCEs sostienen un balance riguroso ante la experiencia discursiva. En la micro-realidad, el algoritmo respeta la brevedad y sintaxis del enunciado al segmentar por puntuación (STC/UCE) y aplicar análisis cualitativos de confirmación sobre verbatims característicos [2, 8, 30]. En la escala macro-estructural, la prueba de doble clasificación filtra la volatilidad del habla espontánea, garantizando que solo la latencia de las estructuras de coocurrencia estables (los "mundos lexicales") sea retenida para la proyección en el Análisis Factorial de Correspondencias (AFC) [2, 25, 31].
   • Tensión Epistémica (Elegancia Parsimoniosa vs. Contingencia del Caos):
   El modelo aporta parsimonia formal al comprimir matrices lógicas masivas e hiper-dispersas (\\(>99\%\\) de ceros) en un número reducido de clases disyuntivas [4, 25, 32]. No obstante, preserva la contingencia del texto real al negarse a forzar la inclusión del 100% de los datos: el descarte explícito de las UCEs inestables (que puede alcanzar entre un 15% y un 40% del corpus) reconoce abiertamente la presencia de ruido, ambigüedad sintáctica o transiciones temáticas no consolidadas sin disolver el rigor de la partición [2, 3, 27].
   • Alcance Práctico (Tracción Instrumental vs. Potencia Crítica):
   En el terreno operativo, las métricas de cobertura y retención ofrecen tracción técnica precisa para la auditoría de software CAQDAS/Lexicométricos (ALCESTE, IRAMUTEQ, R-Rainette), proveyendo indicadores de calidad replicables (\\(\chi^2\\), ARI, TTR, \\(CR\\)) [7, 26, 29, 33]. Simultáneamente, despliegan potencia crítica al desarmar la "ilusión de objetividad" del procesamiento automatizado: recuerdan que la retención matemática mide coherencia estadística de coocurrencia, pero no agota la interpretación semántica e intersubjetiva que el investigador debe ejercer sobre el texto [31, 34, 35].
1. Registro Situado de Autorías y Evidencia
   Arquitectos Primarios del Concepto
   • Max Reinert [1983, 1986, 1990, 2003 | CNRS / Université de Toulouse, Francia]: Creador originario del software ALCESTE, del algoritmo de Clasificación Descendente Jerárquica (CDH), del protocolo de Doble Clasificación sobre UCEs y de la teoría de los "mundos lexicales" estables [1, 2, 36, 37].
   • Jean-Paul Benzécri [1973, 1981 | Université Paris VI, Francia]: Formalizador del Análisis Factorial de Correspondencias (AFC) y de las métricas de distancia \\(\chi^2\\) utilizadas por Reinert como motor de corte en la extracción del primer factor de inercia [5, 16, 35, 38].
   Comentaristas, Teóricos de Enlace y Revisiones Críticas
   • Pierre Ratinaud y Pascal Marchand [2009, 2012 | Université de Toulouse / IRAMUTEQ]: Desarrolladores del software libre IRAMUTEQ; reimplementaron el algoritmo Reinert de doble clasificación en Python y R, difundiendo las métricas de perfiles léxicos (\\(\chi^2\\), \\(\phi\\)) y retención de UCEs [7, 33, 36, 39].
   • Lucile Montalescot, K. Lamore, C. Flahault y A. Untas [2024 | Université Paris Cité, Francia]: Autoras de las guías metodológicas de doble interpretación del método Reinert; denunciaron la "ilusión de objetividad" de las estadísticas léxicas y sistematizaron la lectura de UCEs retenidas frente a descartadas [30, 31, 34].
   • Diego Paucar [2025 | Análisis Estadístico de Texto, Perú]: Desarrollador de arquitecturas modernas de la Clasificación Jerárquica Divisiva (CJD); integró la optimización de cobertura de UCEs mediante PatternSearch, pseudocuentos \\(\epsilon = 0.1\\), alineación por el algoritmo Húngaro (Munkres) y validación del Índice de Rand Ajustado (ARI) [6, 17, 28, 29].
   • Guillermo Rojo [2021 | Universidad de Santiago de Compostela / RAE]: Lexicómetra y lingüista de corpus; analizó las distribuciones de Zipf-Mandelbrot, los ratios TTR y la discrepancia entre listas de frecuencias de diccionarios e inventarios de corpus [11, 13, 40].
   Evidencia Empírica de Contraste y Estudios de Falsación
   • Max Reinert [1990, 2008 | Estudio empírico sobre Aurélia de Gérard de Nerval y el Petit Larousse]: Investigación empírica donde comprobó que en la obra literaria Aurélia la doble clasificación retuvo el 58% de las UCEs, mientras que en la prueba masiva del dictionnaire Larousse (34 193 UCEs) la concordancia topica alcanzó una correlación de \\(\chi^2\\) sumamente elevada con un 72.5% de retención estable [2, 16, 23, 41].
   • C. Ferreira et al. [2020 | Estudio comparativo ALCESTE vs. IRAMUTEQ]: Evaluación empírica en salud que demostró que sobre un corpus de entrevistas a pacientes, ALCESTE e IRAMUTEQ alcanzaron una retención de UCEs idéntica del 68%, confirmando la reproducibilidad estricta del algoritmo divisor de Reinert [27, 42, 43].
   <\alceste> el pipeline y sin perder trazabilidad.
1. Un solo puntaje por UCE, varios cortes. Los niveles son cortes de un puntaje, no pipelines distintos.
1. Las clases se definen una sola vez, sobre el núcleo estricto. Cambiar de nivel solo cambia qué UCEs se muestran asignadas.
1. Umbrales y criterios de éxito fijados antes de ver resultados. Un nivel es el principal; los otros son análisis de sensibilidad.
1. Lo inductivo y lo proyectado se mantienen separados, como ya hace _project_liminal_uces.
1. Resultados inmutables y versionados. Una corrida no se sobrescribe; se agrega otra.
1. Cuellos de botella identificados (según el código)

#

Cuello de botella
Dónde
1
Triple intersección de 6 condiciones (A, B, C y los tres cruces)
_triple_stability_verdicts
2
Hungarian uno a uno: si K difiere entre corridas, las clases sin pareja pierden todas sus UCEs
_hungarian_stability_direct
3
El CDH puede terminar con K distinto en cada corrida (R² ≥ 0.05, permutación, mínimo 10 % de las UC por nodo)
ClasificadorDescendente._partition
4
Bigramas y trigramas en la matriz binaria: más dimensión y dispersión
MatrizBuilder._iter_terms
5
Optuna evalúa sin vectores retrofitted, así que no mide el criterio de la corrida final
_evaluacion_rapida
6
Cobertura de vocabulario: unidades distintas en numerador (stems, bigramas, trigramas) y denominador (lemas); vocabulario recalculado solo sobre UCEs estables
dashboard.py ~1512–1517; main_workflow.py ~7572
7
UC final de cada documento sin piso de formas
construir_ucs 3. Soluciones propuestas
ID
Solución
Pros
Contras
S1
Mapeo muchos a uno restringido (o K fijo) en lugar del Hungarian 1-1
Elimina la pérdida por K distinto; cambio local
Puede fusionar clases distintas; requiere restricción
S2
Puntaje de consenso blando en lugar de intersección dura
Graduado, usa los tres métodos
El umbral es una decisión de diseño
S3
Proyección liminal como nivel aparte
Ninguna UCE se pierde como dato
No es inductivo; la retención pasa a ser por niveles
S4
Quitar o restringir n-gramas; vocabulario sobre todo el corpus
Barato; arregla la cobertura
Se pierde información multi-palabra
S5
Alinear Optuna con la corrida final y optimizar fiabilidad, validez y utilidad
Aprovecha la infraestructura
Más ruido; riesgo de sobreajuste
S6
Piso para la UC final (fusionar con la anterior)
Reduce perfiles ruidosos
Cambia levemente las UC
Orden recomendado: primero lo que no sacrifica precisión (S1, S6, métrica de cobertura), luego S2 y S3 con la calibración contra el estándar humano, y por último S4 y S5 como experimentos. 4. Puntaje de consenso y niveles
4.1 Puntaje por UCE
Para cada UCE se calcula, sobre las comparaciones disponibles (A1↔A2, B1↔B2, C1↔C2, A↔B, B↔C, A↔C):
• s_acuerdo = fracción de comparaciones en que la UCE coincide (0–1).
• s_margen = margen χ² a su clase (distancia a la segunda clase menos la distancia a la primera, normalizada).
• s = combinación (por ejemplo lexicográfica: primero s_acuerdo, desempate por s_margen).
Las comparaciones no disponibles (por ejemplo sin método C) se excluyen del denominador y se registra cuántas fueron.
4.2 Calibración
Sobre el subconjunto codificado a mano se ajusta una regresión isotónica de s a la probabilidad de acierto p_cal. Cada nivel se define por una precisión mínima esperada, no por una retención:
Nivel
Definición (valores propuestos, a fijar de antemano)
Contenido
Estricto
s_acuerdo = 1 (equivale al núcleo actual)
UCEs inductivas del núcleo
Equilibrado
p_cal ≥ 0.80
Núcleo + consenso alto
Amplio
p_cal ≥ 0.70 + proyectadas
Todo lo anterior + liminales (marcadas)
Invariante: los niveles están anidados (Estricto ⊆ Equilibrado ⊆ Amplio). 5. Persistencia de más de un resultado por umbral
Necesidad: poder guardar varios resultados para el mismo umbral (distintas semillas, configuraciones o versiones de código) y compararlos, sin sobrescribir.
5.1 Modelo
• run: una corrida completa. Identificador run_id = hash de (corpus, configuración, versión de código, semilla). Inmutable.
• tier_result: el resultado de aplicar un perfil de umbral a una corrida. Clave (run_id, tier_id). Varios tier_result pueden compartir el mismo tier_id si vienen de corridas distintas.
• uce_assignment: una fila por (run_id, uce_id) con el puntaje y la asignación; el nivel de pertenencia se deriva del puntaje, no se duplica.
• active_pointer: qué run_id y tier_id muestra el dashboard por defecto (y cuál es el nivel principal declarado).
5.2 Esquema (borrador SQL, adaptar al almacenamiento real)
CREATE TABLE run (
run_id TEXT PRIMARY KEY,
created_at TIMESTAMP NOT NULL,
corpus_hash TEXT NOT NULL,
code_version TEXT NOT NULL,
seed INTEGER NOT NULL,
config_json JSONB NOT NULL,
mode TEXT NOT NULL,
label TEXT
);

CREATE TABLE tier_profile (
tier_id TEXT PRIMARY KEY,
name TEXT NOT NULL,
rule_json JSONB NOT NULL,
preregistered BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE uce_assignment (
run_id TEXT REFERENCES run(run_id),
uce_id TEXT NOT NULL,
doc_id TEXT NOT NULL,
cluster_core INTEGER,
s_acuerdo REAL,
s_margen REAL,
p_cal REAL,
n_comparisons INTEGER,
projected_cluster_id INTEGER,
PRIMARY KEY (run_id, uce_id)
);

CREATE TABLE tier_result (
run_id TEXT REFERENCES run(run_id),
tier_id TEXT REFERENCES tier_profile(tier_id),
metrics_json JSONB NOT NULL,
PRIMARY KEY (run_id, tier_id)
);

CREATE TABLE active_pointer (
scope TEXT PRIMARY KEY,
run_id TEXT NOT NULL,
tier_id TEXT NOT NULL
);
metrics_json guarda una instantánea por nivel: retención, precisión esperada, Jaccard medio por clase, punto en la curva riesgo-cobertura, tamaños de clase.
5.3 Puntos de integración en el código
• El puntaje se calcula en _triple_stability_verdicts (que ya tiene todos los conjuntos estables y cruzados).
• _save_all_uces ya serializa uce.to_dict() más los campos de proyección; se añaden s_acuerdo, s_margen, p_cal y run_id.
• El dashboard lee data["uces"] y calcula la retención ahí (dashboard.py ~1507–1516); se reemplaza por una lectura filtrada por nivel.
• La carga de configuración vía ConfigStore sobrescribe valores del dashboard; la configuración efectiva debe guardarse en run.config_json.
5.4 Reglas
• Escritura de solo agregado: ninguna corrida se modifica ni se borra automáticamente.
• Re-ejecutar con el mismo run_id es idempotente.
• El dashboard permite comparar dos (run_id, tier_id) lado a lado.
• Al cambiar de nivel en vivo no se escribe nada: es una consulta. 6. Plan de pruebas
6.1 Pruebas unitarias (pytest, carpeta tests/)
• Mapeo muchos a uno (S1): con K distintos A=4 y B=3, ninguna UCE queda sin pareja salvo las de clases vacías; con K iguales reproduce el resultado del Hungarian.
• Puntaje: s_acuerdo está en [0, 1]; con todas las comparaciones coincidentes vale 1; excluye comparaciones no disponibles.
• Anidamiento: para cualquier umbral, Estricto ⊆ Equilibrado ⊆ Amplio.
• Clases fijas: cambiar de nivel no altera cluster_core de ninguna UCE del núcleo.
• Invariante abductivo: ninguna UCE tiene a la vez cluster_id y projected_cluster_id.
• Determinismo: misma semilla y misma configuración producen el mismo run_id y las mismas asignaciones.
• Cobertura de vocabulario: sobre un corpus de juguete calculado a mano, la métrica por tokens coincide.
• Piso de UC (S6): ninguna UC queda con menos de min_forms formas salvo documentos más cortos que el piso.
6.2 Pruebas de persistencia
• Ida y vuelta: guardar y leer una corrida devuelve asignaciones idénticas.
• Idempotencia: guardar dos veces el mismo run_id no duplica filas.
• Varios resultados por umbral: dos corridas con semillas distintas y el mismo tier_id coexisten y se consultan por separado.
• Concurrencia: dos escrituras simultáneas no se pisan (clave primaria).
• Migración: los datos actuales (data["uces"] sin puntaje) se leen sin error y se marcan como legacy.
6.3 Pruebas basadas en propiedades
• Renombrar etiquetas de clase no cambia el ARI ni el puntaje.
• Permutar el orden de los documentos no cambia las asignaciones de las UCEs.
• Duplicar el corpus no cambia las clases del núcleo de forma sustancial.
6.4 Corpus sintético con verdad conocida
Generar corpus desde un modelo de temas con K y nivel de ruido controlados. Medir cómo cambian el ARI, el F1 y la retención al subir el ruido y al variar el largo de las UCEs. Incluir un corpus de solo ruido (texto barajado): el método no debe devolver clases estables; la "retención" en ese caso es la referencia nula.
6.5 Estándar humano
• Muestra estratificada de ~150–200 UCEs por clase, nivel y documento.
• Dos codificadores independientes; kappa o alfa de Krippendorff entre ellos; resolver desacuerdos.
• Por nivel: precisión, recall y F1 macro con intervalos bootstrap.
• Con ~120 UCEs y precisión cercana a 0.80, el intervalo del 95 % ronda ±7 puntos; más muestra, menos ancho.
6.6 Validación fuera de muestra
Validación cruzada agrupada por documento: se definen las clases en el conjunto de entrenamiento y se proyectan las UCEs del de validación; se mide ahí el acuerdo con el estándar y con la clasificación de referencia.
6.7 Matriz de ablación
Factores: mapeo (1-1, muchos a uno, K fijo), consenso (6 condiciones, blando, 2 de 3), n-gramas (sí/no), tsj, classification_mode (wc_only, wc_coref, all), proyección (sí/no). Diseño factorial fraccionado, cinco semillas por celda. Salidas: embudo de retención A → A∩B → A∩B∩C, ARI, F1 contra el estándar, Jaccard por clase, AURC.
6.8 Sensibilidad al nivel
Para cada clase, comparar los 50 términos característicos entre niveles (correlación de rangos de Kendall). Una clase cuyo perfil cambie mucho al pasar de Estricto a Amplio se marca como frágil.
6.9 Regresión y rendimiento
• Instantánea de métricas de wc_only como línea base; cualquier cambio posterior se compara contra ella.
• Cambiar de nivel en el dashboard debe responder en una consulta (sin recalcular AFC ni CDH); medir latencia y memoria. 7. Indicadores y criterios de aceptación
Indicadores: precisión y recall por nivel contra el estándar, curva riesgo-cobertura (AURC), Jaccard bootstrap por clase, AMI/ARI, PAC, coherencia temática (NPMI o por embeddings), recuperabilidad predictiva (F1 macro con validación agrupada), SMD de retenidas contra descartadas, z-score frente a la referencia nula.
Criterios propuestos (a ajustar antes de ejecutar):

1. La precisión del núcleo no baja más de 2 puntos contra la línea base.
2. Cada nivel cumple su precisión mínima declarada con el límite inferior del IC al 95 %.
3. La retención del nivel principal supera la de la línea base con significancia sobre la referencia nula.
4. Jaccard medio por clase ≥ 0.75 en el núcleo; las clases por debajo de 0.60 se reportan como inestables.
5. SMD de longitud y metadatos entre retenidas y descartadas < 0.10, o se documenta el sesgo.
6. Los niveles cumplen el anidamiento en todas las corridas.
7. Plan por fases
   Fase
   Contenido
   Resultado
   0
   Registrar el embudo y las métricas actuales (línea base)
   Instantánea de referencia
   1
   S1, S6 y métrica de cobertura por tokens
   Mediciones sin sesgo de artefactos
   2
   Puntaje de consenso y tablas de persistencia
   Un puntaje por UCE guardado por corrida
   3
   Estándar humano y calibración
   p_cal y definición de niveles
   4
   Niveles en el dashboard, comparación de corridas
   Control en vivo
   5
   Ablación y Optuna alineado
   Parámetros estructurales elegidos con datos
8. Riesgos
   • Elegir el nivel mirando los resultados: mitigado con el nivel principal declarado de antemano.
   • Sesgo de selección en las UCEs retenidas: se mide con las SMD y se reporta.
   • Calibración con poca muestra: intervalos anchos; se amplía el estándar humano antes de fijar niveles.
   • Sobreajuste del optimizador: varias semillas por configuración y validación agrupada por documento.
   • Cambio de esquema: migración con marca legacy y lectura retrocompatible.
9. Decisiones abiertas
10. ¿Existe ya algún subconjunto codificado a mano, o hay que construirlo?
11. ¿El almacenamiento actual es PostgreSQL (vía ConfigStore) o db.data en archivo? Define cómo se implementa el esquema.
12. ¿Cuál será el nivel principal declarado?
13. ¿Qué precisión mínima esperas por nivel (por ejemplo 90/80/70 %)?
14. Frontend (dashboard)
    Los cambios de interfaz van en un módulo aparte, reii/quality/dashboard_panel.py, que lee de las tablas de corridas y se monta como una pestaña nueva ("F · Calidad y niveles") siguiendo el mismo patrón de carga diferida (tab_f.open) del resto del dashboard. También funciona como página independiente (python -m reii.quality.cli serve). Cambiar de corrida o de nivel es una consulta: no recalcula CDH, AFC ni φ.
    11.1 Selector de corridas y niveles
    • Lista de corridas no dominadas (mejor compromiso entre retención del núcleo, Jaccard medio, ARI, precisión y AURC), marcadas con ★; un interruptor muestra todas. La corrida migrada (legacy) siempre aparece.
    • Botones de nivel: Estricto, Equilibrado, Amplio.
    • "Fijar como corrida por defecto" guarda el puntero activo (quality_active).
    11.2 Calibración configurable
    • Umbrales por nivel: deslizadores. Con calibración son probabilidades de acierto (p_cal); sin calibración se aplican sobre el puntaje de acuerdo.
    • Modo exploratorio: si el perfil difiere del declarado de antemano (default-v1), aparece una insignia roja y el hash del perfil queda registrado con el resultado.
    • Recalibrar: ajuste isotónico contra el estándar humano, habilitado con un mínimo configurable de UCEs codificadas (60 por defecto). Cada calibración queda versionada.
    11.3 Indicadores nuevos
    Indicador
    Qué muestra
    Retención por nivel
    UCEs asignadas / totales, con las proyectadas contadas aparte
    Embudo
    Retención acumulada A → A∩B → A∩B∩C → cruces
    Precisión y recall por nivel
    Contra el estándar humano, con IC bootstrap (cuando hay codificación)
    Curva riesgo-cobertura y AURC
    Precisión a medida que se amplía la cobertura
    Cobertura de vocabulario, dos definiciones
    Por tipos (la anterior) y por tokens (la nueva), rotuladas
    Jaccard por clase
    Estabilidad entre las dos clasificaciones de conteo de palabras; marca "frágil" bajo 0.60
    Sesgo de selección
    SMD de longitud entre retenidas y descartadas; dispersión de retención por documento
    Pendientes que el panel no calcula todavía: z-score frente a la referencia nula (requiere rehacer la corrida con datos barajados), Jaccard por bootstrap y la correlación de rangos de términos entre niveles (la función existe en metrics.py pero no está conectada al panel).
    11.4 Cambios adicionales
15. Detalle por UCE: clase inductiva, clase proyectada, puntaje de acuerdo, número de comparaciones, p_cal, nivel mínimo y origen del puntaje.
16. Insignia de proyectadas y advertencia si la diferencia de longitud entre retenidas y descartadas supera |SMD| 0.10.
17. Estándar humano dentro del dashboard: muestra estratificada por nivel y clase, tabla editable de aciertos, guardado por codificador.
18. Guardar resultado: persiste (corrida, nivel, perfil de umbrales); pueden coexistir varios resultados por umbral.
19. Exportar la vista activa (JSON con corrida, nivel, perfil y métricas) y las asignaciones (CSV).
    11.5 Criterios de aceptación del frontend
    • Cambiar de nivel no ejecuta CDH ni AFC.
    • El panel funciona con la corrida migrada (puntaje binario) y con corridas nuevas (puntaje graduado).
    • Una vista exportada contiene todo lo necesario para reproducirla (corrida, nivel, perfil, hash).
20. Archivos Python y orden de ejecución
    Todo vive en src/reii/quality/, más un script de parches y dos archivos de pruebas. Se entrega como reii_quality_pack.zip (descomprimir en la raíz del repo).
    12.1 Archivos, en orden de dependencia

#

Archivo
Función
1
store.py
Persistencia append-only: corridas, asignaciones por UCE, resultados por nivel y perfil, estándar humano, calibraciones versionadas, puntero activo, auditoría. PostgreSQL (psycopg) o SQLite
2
mapping.py
Estabilidad entre dos particiones: Hungarian 1-1 o muchos a uno restringido (min_share, max_merge); compatible con la firma de _hungarian_stability_direct
3
consensus.py
Puntaje de acuerdo por UCE y margen χ²
4
metrics.py
Cobertura por tokens y tipos, embudo, Jaccard, SMD, riesgo-cobertura, IC bootstrap, Pareto
5
calibration.py
Perfiles de niveles, hash de perfil, modo exploratorio, calibración isotónica, asignación anidada
6
service.py
Evaluación de una corrida, recalibración, muestra estratificada
7
hooks.py
Guarda la corrida de calidad tras _save_all_uces; nunca rompe el pipeline; oculta claves y DSN de la configuración
8
migrate_legacy.py
Migra los resultados ya almacenados (PostgreSQL o JSON) como corrida legacy
9
dashboard_panel.py y app.py
Panel Streamlit y página independiente
10
cli.py
init, migrate-legacy, runs, evaluate, set-active, selftest, serve
11
scripts/apply_quality_patches.py
Tres parches mínimos e idempotentes (hook, mapeo por variable de entorno, pestaña F), con copia .quality.bak, respeto de CRLF y --revert
12.2 Resultados anteriores ya almacenados
La corrida actual del sistema completo (tabla uces y kv_store en PostgreSQL) se migra como una corrida de tipo legacy:
• Se conservan todas las UCEs (estables y no estables), la clase inductiva, la proyección liminal, la longitud y el vocabulario.
• El identificador es determinista (legacy- más un hash del contenido): migrar dos veces no duplica.
• Las tablas nuevas no tienen claves foráneas hacia uces, así que Database._save() (que reconstruye uces en cada corrida) no puede borrar el historial de calidad.
• Limitación: las banderas por comparación (stable_wc, stable_sim, stable_emb, stable_cross_*) son atributos dinámicos que UCE.to_dict() y migrate_uces no guardan. Para la corrida previa solo existe is_stable, así que el puntaje de acuerdo es 1.0 en las estables y 0.0 en el resto (s_source = legacy_is_stable). Los puntajes graduados aparecen en las corridas nuevas guardadas con el hook.
• La corrida migrada sirve como línea base de la Fase 0 del plan.
12.3 Orden de ejecución

1. python -m reii.quality.cli selftest
2. python -m pytest tests/test_quality_core.py tests/test_quality_store_flow.py
3. python scripts/apply_quality_patches.py --dry-run, y luego sin --dry-run
4. python -m reii.quality.cli init
5. python -m reii.quality.cli migrate-legacy --dry-run, y luego sin --dry-run (o --json RUTA)
6. streamlit run src/reii/dashboard.py y abrir la pestaña F
7. En las siguientes corridas del pipeline el hook guarda la corrida de calidad; opcionalmente REII_STABILITY_MAPPING=many_to_one activa el mapeo muchos a uno.
   12.4 Qué se probó y qué no
   Probado (43 pruebas, también desde una extracción limpia del zip): mapeo (K distinto, equivalencia con el Hungarian cuando K es igual, límite de fusiones), consenso, métricas, anidamiento de niveles, calibración, persistencia (idempotencia, aislamiento entre corridas, varios resultados por umbral, versiones), migración desde JSON, hook con objetos simulados, ocultamiento de secretos, parches (idempotentes, CRLF, reversibles, sintaxis válida) y el panel con las pruebas de Streamlit.
   No probado: el camino PostgreSQL real (el mismo código se probó en SQLite), el hook dentro de una corrida completa del pipeline, la lectura de PostgreSQL en la migración (solo la ruta JSON), y el efecto del mapeo muchos a uno sobre tus datos reales.
   12.5 Decisiones que quedan
8. Fijar de antemano el nivel principal y los umbrales de cada nivel (hoy son los valores propuestos de default-v1).
9. Construir el estándar humano: sin él no hay p_cal ni precisión por nivel, y los niveles Equilibrado y Amplio usan umbrales de respaldo sobre el puntaje de acuerdo.
10. Integrar el objetivo de Optuna alineado (S5): no se incluyó en este paquete porque toca la evaluación rápida del workflow.
