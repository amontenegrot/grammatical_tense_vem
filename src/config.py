# src/config.py
"""
Módulo de configuración global para el proyecto Voxelwise Encoding Model.
Centraliza rutas de sistema, umbrales neurobiológicos, parámetros computacionales,
arquitectura de 43 predictores y especificaciones metodológicas del anteproyecto.
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

# Historia reservada exclusivamente para la evaluación fuera de muestra (Test Set).
# - "wheretheressmoke": Seleccionada metodológicamente por contar con múltiples 
#   presentaciones repetidas, permitiendo maximizar la relación señal-ruido (SNR)
#   mediante el promedio temporal de repeticiones BOLD estandarizadas.
TEST_STORY = "wheretheressmoke"

# Historia seleccionada para exportar a SQLite como muestra visual
PRESENTATION_STORY = "odetostepfather"


# ==============================================================================
# 3. PARÁMETROS NEUROBIOLÓGICOS Y TEMPORALES
# ==============================================================================

# Frecuencia de muestreo de alta resolución en Hz (10 ms)
HIGH_RES_FS = 100
# Tiempo de repetición del fMRI en segundos (0.5 Hz)
TR_FMRI = 2.0
# Duración total de la respuesta hemodinámica simulada en segundos
HRF_LENGTH_SEC = 32.0

# Umbrales para pausas y marcas metalingüísticas (derivados de la auditoría acústica)
# 0.35s aísla pausas cortas evitando la fragmentación de sintagmas nominales (comas).
# 0.65s representa el percentil 75, utilizado para delimitar fronteras clausales (puntos).
THRESHOLD_COMMA_SEC = 0.35
THRESHOLD_PERIOD_SEC = 0.65


# ==============================================================================
# 4. PROCESAMIENTO DE LENGUAJE NATURAL (NLP) Y REPRESENTACIÓN SEMÁNTICA
# ==============================================================================

SPACY_MODEL_NAME = "en_core_web_trf"

# Cantidad de oraciones procesadas simultáneamente por el pipeline de spaCy.
# - 256: Equilibrio ideal para exprimir procesadores de alto rendimiento sin saturar RAM.
NLP_BATCH_SIZE = 256

# Expresión regular para filtrar ruidos no lingüísticos en los archivos TextGrid.
# Detecta y elimina automáticamente:
# - Etiquetas de silencio o pausas cortas ('sil', 'sp').
# - Ruidos vocales y respiración ('spn', 'br', 'lg' para risas).
# - Cualquier anotación metalingüística encerrada en corchetes, llaves 
#   o paréntesis angulares (ej. [laugh], {cough}, <breath>).
LSA_NOISE_PATTERN = re.compile(r'\[|\]|\{|\}|\<|\>|spn|^sp$|^sil$|^br$|^lg$', re.IGNORECASE)

# --- Análisis Semántico Latente (LSA) ---
LATENT_SEMANTIC_COMPONENTS = 10

# Frecuencia mínima de aparición documental en TF-IDF (Filtro de rareza).
LSA_MIN_DF = 3

# Frecuencia máxima de aparición documental en TF-IDF (Filtro de stop-words de dominio).
# 0.85: Ignora palabras presentes en más del 85% de las unidades textuales.
LSA_MAX_DF = 0.85

# Palabras vacías personalizadas para omitir en el cálculo Semántico LSA
LSA_CUSTOM_STOP_WORDS = {
    'like', 'know', 'go', 'uh', 'say', 'um', 'think', 'get', 
    'come', 'look', 'to', 'thing', 'tell', 'want', 'start', 
    'feel', 'right', 'mean', 'kind', 'yeah', 'oh', 'well'
}


# ==============================================================================
# 5. MATRIZ DE CARACTERÍSTICAS (ARQUITECTURA FIJA DE 43 PREDICTORES)
# ==============================================================================

# Dimensiones exactas según el anteproyecto (39 control + 4 de interés = 43 total)
# Decisión técnica: Centralizar las dimensiones previene discrepancias entre extracción y modelos.
SPACES_DIMENSIONS = {
    'phonological': 14,
    'lexical_stats': 4,
    'categorical': 8,
    'syntactic': 3,
    'semantic': 10,
    'tense': 4  # Estrictamente: past_regular, past_irregular, non_past_marked_3sg, non_past_unmarked
}

TOTAL_FEATURES_COUNT = sum(SPACES_DIMENSIONS.values())  # Exactamente 43
CONTROL_FEATURES_COUNT = TOTAL_FEATURES_COUNT - SPACES_DIMENSIONS['tense']  # Exactamente 39

# Orden estricto y único de columnas en la matriz predictora consolidada
FEATURE_COLUMNS_ORDER = [
    # 1. Fonético-Fonológico (14)
    'vocalic', 'consonantal', 'voiceless', 'voiced', 'bilabial', 
    'labiodental', 'dental_alveolar', 'palatal_velar', 'stop', 
    'fricative', 'affricate', 'nasal', 'liquid', 'approximant',
    # 2. Léxico-Estadístico (4)
    'word_presence', 'lexical_frequency', 'word_length_chars', 'word_duration_secs',
    # 3. Léxico-Categorial (8)
    'noun', 'adjective', 'adverb', 'pronoun', 'preposition', 
    'conjunction', 'determiner', 'non_finite_verb',
    # 4. Sintáctico (3)
    'dependency_distance', 'syntactic_depth', 'is_root',
    # 5. Léxico-Semántico LSA (10)
    'semantic_dim_0', 'semantic_dim_1', 'semantic_dim_2', 'semantic_dim_3', 'semantic_dim_4',
    'semantic_dim_5', 'semantic_dim_6', 'semantic_dim_7', 'semantic_dim_8', 'semantic_dim_9',
    # 6. Tiempo Gramatical Finito (4)
    'past_regular', 'past_irregular', 'non_past_marked_3sg', 'non_past_unmarked'
]


# ==============================================================================
# 6. COMPUTACIÓN Y MACHINE LEARNING (RIDGE REGRESSION)
# ==============================================================================

# Semilla global para garantizar la reproducibilidad exacta en procesos estocásticos
RANDOM_SEED = 10

# Backend matemático para Himalaya
HIMALAYA_BACKEND = "numpy"

# Tamaño del lote para procesamiento espacial (Chunking).
# Decisión técnica: 10,000 vóxeles limitan el consumo de RAM de las matrices internas 
# de regresión a ~2 GB por iteración, evitando el colapso (Out Of Memory) en WSL/Linux.
VOXEL_BATCH_SIZE = 10000

# Espacio de búsqueda de penalización L2 común: 20 valores distribuidos logarítmicamente entre 0.01 y 10,000
RIDGE_ALPHAS = np.logspace(-2, 4, 20)

# Número de particiones para la validación cruzada interna (Story-Blocked CV)
RIDGE_CV_FOLDS = 3


# ==============================================================================
# 7. VALIDACIÓN ESTADÍSTICA (DESPLAZAMIENTOS CIRCULARES Y FDR)
# ==============================================================================

# Márgenes para exclusión de desplazamientos cercanos al orden original (evitar autocorrelación residual HRF)
CIRCULAR_SHIFT_MARGIN_MIN = 10         # Margen base: 10 TRs (20 segundos)
CIRCULAR_SHIFT_MARGIN_SENSITIVITY = 16  # Margen ampliado: 16 TRs (32 segundos, longitud biológica de HRF)

# Umbral de significancia estadística (Alpha) para control de falsos descubrimientos (FDR Benjamini-Hochberg)
STATISTICAL_ALPHA = 0.05


# ==============================================================================
# 8. VISUALIZACIÓN CORTICAL (PYCORTEX)
# ==============================================================================

# Colormaps para las capas corticales complementarias
CORTICAL_COLORMAP_GLOBAL = "OrRd"          # Capa 1: Desempeño global (R2 > 0)
CORTICAL_COLORMAP_TENSE_SIG = "OrRd"       # Capa 2a: Varianza única umbralizada (FDR < 0.05)
CORTICAL_COLORMAP_DIVERGING = "coolwarm"   # Capa 2b: Varianza única continua descriptiva (-vmax a +vmax)
