# src/config.py
"""
Módulo de configuración global para el proyecto Voxelwise Encoding Model.
Centraliza rutas de sistema, umbrales biológicos, y parámetros computacionales
y de machine learning para todo el pipeline.
"""

import re
import numpy as np
from pathlib import Path

# ==============================================================================
# 1. RUTAS DEL SISTEMA Y DIRECTORIOS
# ==============================================================================

# Se asume una estructura donde los datos crudos/BIDS están separados
# de los datos procesados del proyecto para evitar confusión de carpetas
PROJECT_ROOT = Path("/home/almontao/proyectos/grammatical_tense_vem")
DATA_DIR = PROJECT_ROOT / "data" / "ds003020"

# --- Datos Crudos y Derivados BIDS ---
DIR_TEXTGRIDS = DATA_DIR / "derivatives" / "TextGrids"
DIR_FMRI = DATA_DIR / "derivatives" / "preprocessed_data"
DIR_STIMULI = DATA_DIR / "stimuli"

# --- Salidas del Pipeline (Derivados Propios) ---
# Decisión técnica: 'processed' almacena matrices, 'artifacts' almacena modelos entrenados
DIR_PROCESSED = PROJECT_ROOT / "data" / "processed"
DIR_ARTIFACTS = PROJECT_ROOT / "artifacts"

# Subdirectorios para el almacenamiento de matrices de características
# - HIGH_RESOLUTION: Matrices continuas a 100 Hz
# - FMRI_TR: Matrices convolucionadas y reducidas a 0.5 Hz
DIR_FEATURES_HIGH_RESOLUTION = DIR_PROCESSED / "features_high_resolution"
DIR_FEATURES_FMRI_TR = DIR_PROCESSED / "features_fmri_tr"

# --- Bases de Datos y Reportes ---
DB_DIR = PROJECT_ROOT / "db"
# Base de datos principal para metadatos y auditorías (Prefijo 'audit_')
DB_PATH = DB_DIR / "audit_metadata.sqlite"
# Base de datos secundaria exclusiva para presentación (matrices truncadas a 1 historia)
PRESENTATION_DB_PATH = DB_DIR / "features_presentation.sqlite"
# Directorio para centralizar los reportes CSV generados por las auditorías
DIR_AUDIT_REPORTS = DB_DIR / "csv_reports"


# ==============================================================================
# 2. PARÁMETROS EXPERIMENTALES Y DISEÑO DE LA COHORTE
# ==============================================================================

# Clave principal del dataset BOLD dentro de los archivos HDF5 de fMRI.
HDF5_DATASET_KEY = "data"

# Historias excluidas del análisis por fallas técnicas o decisiones metodológicas
EXCLUDED_STORIES = {'legacy', 'exorcism'}

# Historia reservada exclusivamente para la evaluación del modelo (Test Set).
# - "wheretheressmoke": Metodológicamente seleccionada por tener múltiples 
#   presentaciones, permitiendo calcular la confiabilidad y el techo de ruido.
TEST_STORY = "wheretheressmoke"

# Historia seleccionada para exportar a SQLite como muestra visual
PRESENTATION_STORY = "odetostepfather"


# ==============================================================================
# 3. PARÁMETROS NEUROBIOLÓGICOS Y TEMPORALES
# ==============================================================================

# Frecuencia de muestreo de alta resolución en Hz (10 ms)
HIGH_RES_FS = 100
# Tiempo de repetición del fMRI en segundos
TR_FMRI = 2.0
# Duración total de la respuesta hemodinámica simulada en segundos
HRF_LENGTH_SEC = 32.0

# Umbrales para pausas y marcas metalingüísticas (derivados de la auditoría acústica)
# 0.35s aísla pausas cortas evitando la fragmentación de sintagmas nominales.
# 0.65s representa el percentil 75, utilizado para delimitar fronteras clausales.
THRESHOLD_COMMA_SEC = 0.35
THRESHOLD_PERIOD_SEC = 0.65


# ==============================================================================
# 4. PROCESAMIENTO DE LENGUAJE NATURAL (NLP) Y EXTRACCIÓN
# ==============================================================================

SPACY_MODEL_NAME = "en_core_web_trf"

# Cantidad de oraciones procesadas simultáneamente por el pipeline de spaCy.
# - 256: Equilibrio ideal para exprimir procesadores de alto rendimiento (ej. i9) sin saturar RAM.
# - Disminuir (ej. 64-128) si hay cuellos de botella de memoria; aumentar si se usa GPU dedicada.
NLP_BATCH_SIZE = 256

# TODO: Refactorizar globalmente 'LSA_NOISE_PATTERN' a 'TEXTGRID_NOISE_PATTERN'
# Expresión regular para filtrar ruidos no lingüísticos en los archivos TextGrid.
# Detecta y elimina automáticamente:
# - Etiquetas de silencio o pausas cortas ('sil', 'sp').
# - Ruidos vocales y respiración ('spn', 'br', 'lg' para risas).
# - Cualquier anotación metalingüística encerrada en corchetes, llaves 
#   o paréntesis angulares (ej. [laugh], {cough}, <breath>).
# Vital para evitar que el modelo interprete el ruido como si fueran palabras reales.
LSA_NOISE_PATTERN = re.compile(r'\[|\]|\{|\}|\<|\>|spn|^sp$|^sil$|^br$|^lg$', re.IGNORECASE)

# --- Análisis Semántico Latente (LSA) ---
LATENT_SEMANTIC_COMPONENTS = 10

# Frecuencia mínima de aparición documental en TF-IDF (Filtro de rareza).
# - 3: Permite que la palabra exista si aparece al menos 3 veces. Filtra errores de transcripción o rarezas.
# - Aumentar (ej. 5 o 10) restringe el espacio semántico a un vocabulario mucho más común y repetitivo.
LSA_MIN_DF = 3

# Frecuencia máxima de aparición documental en TF-IDF (Filtro de stop-words de dominio).
# - 0.85: Ignora palabras que aparecen en más del 85% de las historias, ya que no discriminan temáticas.
# - Disminuir (ej. 0.70) recorta palabras transversales, forzando al modelo a enfocarse en tópicos más únicos.
LSA_MAX_DF = 0.85

# Palabras vacías personalizadas para omitir en el cálculo Semántico LSA
LSA_CUSTOM_STOP_WORDS = {
    'like', 'know', 'go', 'uh', 'say', 'um', 'think', 'get', 
    'come', 'look', 'to', 'thing', 'tell', 'want', 'start', 
    'feel', 'right', 'mean', 'kind', 'yeah', 'oh', 'well'
}


# ==============================================================================
# 5. MATRIZ DE CARACTERÍSTICAS (PREDICTORES)
# ==============================================================================

# Dimensiones documentadas en la auditoría metodológica (45 características en total)
# Decisión técnica: Centralizar esto evita inconsistencias entre los extractores y el modelo
SPACES_DIMENSIONS = {
    'phonological': 14,
    'lexical_stats': 4,
    'categorical': 8,
    'syntactic': 3,
    'semantic': 10,
    'tense': 6  
}


# ==============================================================================
# 6. COMPUTACIÓN Y MACHINE LEARNING (RIDGE REGRESSION)
# ==============================================================================

# Semilla global para garantizar la reproducibilidad exacta en procesos estocásticos 
# (ej. TruncatedSVD, permutaciones en Ridge, selección de particiones).
RANDOM_SEED = 10

# --- Infraestructura de Procesamiento ---
# Backend matemático. Usar "numpy" para forzar CPU multinúcleo en entorno local
# Opciones soportadas: "numpy" (CPU), "torch" (CPU), "torch_cuda" (GPU)
HIMALAYA_BACKEND = "numpy"

# Tamaño del lote para procesamiento espacial (Chunking).
# Decisión técnica: 10,000 vóxeles limitan el consumo de RAM de las matrices internas 
# de Himalaya a ~2 GB por iteración, evitando el colapso (Out Of Memory) del sistema
VOXEL_BATCH_SIZE = 10000


# --- Hiperparámetros del Modelo Ridge ---
# Espacio de búsqueda de penalización L2. Se usa escala logarítmica estándar
RIDGE_ALPHAS = np.logspace(-2, 4, 20)

# Número de particiones (folds) para la validación cruzada interna del modelo Ridge.
# Define en cuántos bloques se dividen los datos de entrenamiento para buscar el 
# mejor nivel de regularización (Alfa).
# - 3: Valor por defecto (Recomendado). Ofrece un balance óptimo entre tiempo de 
#   cómputo y estabilidad matemática. Ideal para fMRI naturalista dado el volumen 
#   masivo de datos (miles de TRs).
# - 5 o 10: Mayor rigor estadístico en la estimación del hiperparámetro. Reduce 
#   el riesgo de sobreajuste, pero multiplica linealmente el consumo de RAM y el 
#   tiempo de procesamiento. Sugerido solo si la muestra temporal es pequeña.
RIDGE_CV_FOLDS = 3


# --- Validación Estadística ---
# Iteraciones para la distribución nula (desplazamiento circular)
N_PERMUTATIONS = 1000

# Umbral de significancia estadística (Alpha) para control de falsos positivos (FDR).
# - 0.05: Estándar en neuroimagen. Balance adecuado entre sensibilidad y rigor.
# - 0.01: Estricto. Útil si se detecta mucho ruido o para inferencias muy conservadoras.
# - 0.10: Exploratorio. Relaja el filtro si el tamaño de la muestra reduce la potencia.
STATISTICAL_ALPHA = 0.05


# ==============================================================================
# 7. VISUALIZACIÓN CORTICAL (PYCORTEX)
# ==============================================================================

# Colormap por defecto para la visualización de los mapas de calor en PyCortex
CORTICAL_COLORMAP = "OrRd"
