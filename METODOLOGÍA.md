# REPORTE COMPLETO DE DECISIONES METODOLÓGICAS Y ARQUITECTURA COMPUTACIONAL

**Proyecto:** Modelamiento Voxel-wise del Tiempo Gramatical Finito en fMRI Naturalista.  
**Conjunto de Datos (Dataset):** ds003020 (Derivados fMRI preprocesados en HDF5 y transcripciones alineadas en TextGrids).

Este documento detalla exhaustivamente cada etapa del flujo de trabajo, las opciones evaluadas, la justificación teórica subyacente a la extracción lingüística y las decisiones de ingeniería de software implementadas para optimizar el rendimiento computacional.

---

## 1. Diseño del Estudio y Arquitectura General del Flujo de Trabajo

El estudio adopta un diseño observacional basado en datos secundarios de acceso abierto, estudiando la comprensión del habla continua en condiciones naturalistas sin necesidad de recolectar nuevos datos neurofuncionales. La arquitectura computacional del proyecto se dividió en módulos independientes diseñados para mantener una separación estricta entre la **Extracción de Características (Representación del Estímulo)**, el **Modelamiento Temporal**, la **Estimación Predictiva (Machine Learning)** y la **Visualización**. 

### 1.1 Decisiones de Infraestructura y Resiliencia
* **Formatos de Almacenamiento (I/O):** Para manejar la alta carga de lectura/escritura de matrices densas, se sustituyeron los archivos CSV por el formato binario `.parquet` mediante el motor `pyarrow`, el cual conserva los tipos de datos nativos (ej. `int8`, `float32`) y reduce drásticamente el uso de almacenamiento y los cuellos de botella de disco.
* **Sistema de Punto de Control (Checkpointing Atómico):** Debido a la alta exigencia computacional, todos los orquestadores implementan un sistema de validación de existencia de archivos. Si ocurre una interrupción (ej. pérdida de energía), el sistema identifica el último hito procesado y reanuda el cálculo exclusivamente con los datos faltantes, protegiendo días de procesamiento continuo.

---

## 2. Auditoría del Corpus Empírico y Configuración de Prueba

Antes del procesamiento de los estímulos, se implementó un motor de auditoría automatizado apoyado en una base de datos relacional (SQLite).
* **Extracción Acústica en $O(1)$:** En lugar de decodificar matrices de audio pesadas a la memoria, la duración de cada estímulo acústico se determinó leyendo directamente los metadatos de las cabeceras `.wav`, logrando extraer la duración del corpus completo en fracciones de segundo.
* **Validación de la Historia de Prueba:** La historia *"wheretheressmoke"* fue designada como conjunto de prueba independiente (Test Set) para evaluar la confiabilidad del modelo y aproximar el techo de ruido (Noise Ceiling). Una consulta a la base de datos validó estadísticamente sus reproducciones múltiples por participante, identificando asimetrías de escucha que guiarán la evaluación cruzada final.

---

## 3. Construcción de los Espacios de Control (Features)

Para separar rigurosamente la representación lingüística, las matrices se construyeron inicialmente a una alta resolución temporal (100 Hz), preservando la fidelidad acústica antes de la transformación hemodinámica.

### 3.1. Arquitectura de Inferencia NLP (Single-Pass Inference)
* **El Problema:** La extracción categorial, sintáctica, semántica y del tiempo gramatical requiere procesar texto usando un modelo de lenguaje masivo (Transformer RoBERTa, `en_core_web_trf` de spaCy). Desplegar este modelo en paralelo empleando múltiples núcleos generaría un colapso de memoria RAM (más de 20 GB en pesos del modelo) e interbloqueos (*deadlocks*) debido al manejo interno de hilos por PyTorch.
* **La Solución:** Se implementó una **Inferencia de Paso Único (Single-Pass)**. El texto se procesa una única vez por el Transformer y el objeto resultante (`Doc`) se inyecta de forma secuencial a todas las clases extractoras (Categorial, Sintáctica, Semántica y Tiempo Gramatical). Esto aceleró el proceso de NLP en un 75% manteniendo el consumo de RAM seguro y eficiente.

### 3.2. Espacio Fonético-Fonológico
* **Acción:** Conversión de las transcripciones de ARPABET a una matriz temporal binaria de 14 dimensiones (tipo `np.int8`).
* **Decisión Metodológica:** Se optó por una representación híbrida de rasgos articulatorios y distintivos (ej. oclusivo, fricativo, sonoro, bilabial) en lugar de un *one-hot encoding* de fonemas aislados. 
* **Justificación:** Controla perfectamente la varianza acústico-motora en la corteza auditiva primaria y superior. Aísla la carga fonológica de los sufijos gramaticales de interés (ej. la fricatividad/sonoridad de /-s/ y /-z/ en el presente, o las oclusivas /-t/ y /-d/ del pasado), previniendo la confusión entre procesamiento acústico y flexión gramatical.

### 3.3. Espacio Léxico-Estadístico
* **Acción:** Extracción de Ocurrencia (1/0), Frecuencia Logarítmica (Zipf), Longitud de palabra en caracteres y Duración en segundos.
* **Decisión Metodológica:** La métrica de frecuencia se calculó estrictamente sobre la **forma superficial** de la palabra pronunciada y no sobre su lema.
* **Justificación:** Su objetivo es controlar el esfuerzo neurocognitivo ascendente (*bottom-up*) de decodificación y acceso léxico. Las formas finitas tienen distribuciones dispares (ej. "is" es significativamente más frecuente que "was"). Calcular la frecuencia sobre la forma superficial absorbe este sesgo estadístico, evitando que la regresión atribuya al tiempo gramatical la varianza propia de la facilidad de acceso léxico.

### 3.4. Espacio Léxico-Categorial (POS Tagging)
* **Acción:** Clasificación gramatical mediante NLP en 8 dimensiones categoriales.
* **Decisión Metodológica 1 (Puntuación Acústica):** Se simuló una sintaxis basada en umbrales empíricos derivados de la duración de pausas y silencios en el corpus. Pausas entre 0.35s y 0.65s se convirtieron en comas; pausas superiores a 0.65s (percentil 75) en puntos. Se ignoraron etiquetas metalingüísticas de alta varianza.
* **Justificación 1:** Provee a la red neuronal un contexto con fronteras clausales biológicamente informadas, evitando la sobre-segmentación por vacilaciones articulatorias y optimizando la inferencia de dependencias sintácticas.
* **Decisión Metodológica 2 (Exclusión de Verbos Finitos):** Se incluyó la categoría de verbos no finitos (`non_finite_verb`: infinitivos, participios, gerundios), pero se excluyeron deliberadamente todos los verbos finitos.
* **Justificación 2:** Previene colinealidad. Se controla el procesamiento de material verbal en general sin absorber la varianza del contraste flexivo finito, el cual es el fenómeno de interés exclusivo del estudio.

### 3.5. Espacio Sintáctico
* **Acción:** Extracción de Distancia de Dependencia, Profundidad Sintáctica y Estado de Raíz.
* **Decisión Metodológica:** Parametrización parsimoniosa basada en el recuento en números enteros (`np.int16`) de los árboles de dependencia de spaCy.
* **Justificación:** Controla la dinámica de integración clausal y la carga en la memoria de trabajo (Teoría de Integración de Gibson). Estos fenómenos de estructuración sintáctica general pueden covariar con la aparición de verbos principales (finitos) y generan fluctuaciones marcadas en la señal BOLD, por lo cual deben ser controlados.

### 3.6. Espacio Léxico-Semántico Reducido (LSA)
* **Acción:** Construcción de un Análisis Semántico Latente (TF-IDF + Truncated SVD) reducido a 10 dimensiones latentes.
* **Decisión Metodológica 1 (Lemas y Stopwords Conversacionales):** Se entrenó el pipeline global usando únicamente lemas. Se inyectó un diccionario de palabras vacías ajustado empíricamente tras una auditoría, filtrando "muletillas" (ej. *uh, um, like*) y "verbos ligeros" (ej. *know, say, get*).
* **Justificación 1:** La lematización garantiza que el modelo capture contenido abstracto y no superficie flexiva. El filtrado manual fue crítico, ya que el algoritmo SVD interpretaba inicialmente las muletillas hiperfrecuentes como el "tópico principal" de las historias. Su remoción permitió revelar los ejes semánticos reales del discurso narrativo.
* **Decisión Metodológica 2 (Mantenimiento del Estado Semántico):** Los vectores LSA resultantes por cláusula (delimitada por pausas >0.65s) se proyectaron sosteniendo sus valores temporalmente.
* **Justificación 2:** Modela cómo el cerebro actualiza y retiene su estado semántico frase a frase dentro de la memoria de trabajo del oyente.

---

## 4. Construcción del Espacio de Interés

### Espacio de Tiempo Gramatical Finito
* **Acción:** Extracción de las realizaciones internas de Pasado (Regular/Irregular) y No Pasado (Marcado/No Marcado).
* **Decisión Metodológica:** Detección sustentada en etiquetas del Penn Treebank y heurística morfológica (sufijo *-ed*). Se implementó la **exclusión absoluta** de modales defectivos (MD) y contracciones ambiguas (ej. *'d*).
* **Justificación:** Valida el contraste temporal protegiendo la "pureza conceptual" del fenómeno evaluado. Al excluir elementos que entrelazan tiempo, aspecto y modo (TAM), se previene que la varianza atribuida a este espacio obedezca a modalizaciones irrelevantes al contraste puramente flexivo. Para el análisis confirmatorio, estas 4 dimensiones se colapsan en dos dominios globales: Pasado y No Pasado.

---

## 5. Modelamiento Temporal y Transformación Hemodinámica

Para relacionar la matriz de características con la resolución de captura cerebral (fMRI), las matrices a 100 Hz se transformaron y redujeron temporalmente a un TR de 2.0 segundos.

* **Decisión Metodológica 1 (Función Double-Gamma):** Se utilizó una HRF Double-Gamma (parámetros canónicos de SPM) en lugar de un modelo Single-Gamma.
* **Justificación 1:** Tras pruebas de rendimiento (Benchmarking), se demostró que el costo computacional era idéntico. La Double-Gamma ofrece superioridad biológica al modelar con precisión la caída metabólica posterior al estímulo (post-stimulus undershoot).
* **Decisión Metodológica 2 (Convolución Optimizada por FFT):** Se reemplazó la convolución estándar por una basada en la Transformada Rápida de Fourier (`scipy.signal.fftconvolve`).
* **Justificación 2:** Convolucionar matrices temporalmente masivas (historias de 15 minutos a 100 Hz contra filtros de 32 segundos) mediante métodos iterativos presenta una complejidad $O(N \times M)$. La FFT reduce esto a $O(N \log N)$, explotando la capacidad multi-núcleo de la CPU y reduciendo el tiempo de cálculo de horas a segundos.
* **Decisión Metodológica 3 (Diezmado Anti-Aliasing):** El downsampling se ejecutó mediante la función `resample_poly` de SciPy.
* **Justificación 3:** La reducción agresiva de 100 Hz a 0.5 Hz (TR=2s) introduce errores de fase graves (aliasing). Este método aplica de manera nativa un filtro polifásico FIR paso-bajo (anti-aliasing) protegiendo la integridad frecuencial de los picos de activación.

---

## 6. Arquitectura Predictiva e Inferencia Estadística (Machine Learning)

La estimación individual (sujeto por sujeto) se desarrolló empleando modelos masivos sobre toda la corteza simultáneamente (hasta ~110,000 vóxeles por sujeto).

### 6.1. Toma de Decisión de Hardware (CPU vs GPU)
* **Justificación:** A pesar de la disponibilidad de aceleración gráfica, se priorizó estrictamente el procesamiento centralizado en CPU (Intel i9, 16 hilos y 64 GB RAM). Una arquitectura de Banded Ridge Regression con múltiples espacios requiere calcular matrices densas simultáneas y múltiples copias en memoria durante la validación cruzada. Operar en una GPU con 4 GB de VRAM dedicados resultaría inexorablemente en el agotamiento de memoria (*Out of Memory*) o en un fenómeno de *Memory Thrashing* (cuello de botella extremo por transferencias constantes en el bus PCIe). La CPU solucionó de raíz el problema mediante cálculo en bloques iterativos y paralelización de bajo nivel sobre NumPy.

### 6.2. Estrategia de Modelamiento
* **Decisión Metodológica 1 (Banded Ridge Regression):** Se implementó usando la librería `himalaya` con hiper-parametrización L2 independiente (`alphas`) para cada espacio funcional. 
* **Justificación 1:** Dado que los espacios tienen dimensiones y niveles de colinealidad radicalmente diferentes, una penalización global (Ridge estándar) habría permitido que el espacio de mayor tamaño (ej. fonológico) dominara matemáticamente la predicción. La variante en bandas con búsqueda aleatoria (*random_search*) calibra la ponderación óptima de cada dominio neurocognitivo por separado.
* **Decisión Metodológica 2 (Estandarización Estricta - Z-score):** Las medias y varianzas de los espacios se aprendieron exclusivamente del conjunto de entrenamiento y se aplicaron a posteriori sobre el conjunto de evaluación. Además, se forzó la imputación nula de los vóxeles extra-corticales (fondo fMRI) marcados como `NaN` para evitar colapsos aritméticos.
* **Justificación 2:** Prevención de *Data Leakage*. Garantiza que el $\Delta R^2$ (Varianza Predictiva Única) represente una genuina capacidad de generalización fuera de muestra, evitando sesgos optimistas.

### 6.3. Validación Estadística Inferencial
* **Acción:** Generación empírica de una distribución nula mediante *Circular Shift* y control del error mediante corrección *FDR*.
* **Justificación:** A diferencia de una permutación aleatoria (shuffle) clásica, el desplazamiento circular destruye la asociación estímulo-corteza preservando la autocorrelación temporal inherente del flujo sanguíneo fMRI. La corrección Benjamini-Hochberg (Tasa de Falso Descubrimiento, FDR) restringe la aparición de falsos positivos derivados de realizar decenas de miles de contrastes simultáneos (uno por vóxel).

---

## 7. Visualización Anatómica Cortical

Para inspeccionar topográficamente la inferencia estadística del modelo a nivel individual, se implementó una interfaz de exploración 3D empleando `pycortex`.

* **Decisión Metodológica (Evadir límites de la API interna):** Se diseñó una integración de nivel de sistema para cargar las transformaciones anatómicas directamente desde el *Filestore* local (`.json`, `.ctm`), evadiendo restricciones dependientes de las versiones de la API de PyCortex y generando servidores web estáticos de libre acceso (WebGL/HTML).
* **Justificación de Renderizado (Volume vs Vertex):** Los datos fMRI de OpenNeuro operan volumétricamente, obligando a proyectar el $\Delta R^2$ aplanado mediante la clase `cortex.Volume` con mapas de calor univariados (ej. `hot`), o utilizar la clase `cortex.VolumeRGB` para mapeo bivariado (adición de canales de luz para Pasado y No Pasado), garantizando que las mallas corticales se hidraten desde las transformaciones fMRIPrep sin depender de archivos de texturas 2D susceptibles a errores de lectura del navegador. Esto confiere al estudio un recurso analítico, gráfico y divulgativo de alta precisión y calidad reproducible en entornos de Ciencia Abierta.