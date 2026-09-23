# scripts/03_apply_hrf.py
"""Orquestador de Transformación Hemodinámica.

Lee los espacios de alta resolución (100 Hz) exportados en la Fase 2, ejecuta 
la convolución FFT con la HRF canónica de doble gamma y el remuestreo polifásico 
a 0.5 Hz (TR = 2.0s) en paralelo. Consolida los 6 espacios en una matriz predictora
global de exactamente 43 características alineadas al escáner fMRI.
"""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DIR_FEATURES_FMRI_TR,
    DIR_FEATURES_HIGH_RESOLUTION,
    FEATURE_COLUMNS_ORDER,
    HIGH_RES_FS,
    HRF_LENGTH_SEC,
    SPACES_DIMENSIONS,
    TOTAL_FEATURES_COUNT,
    TR_FMRI,
)
from src.db_manager import load_table_to_dataframe, log_execution_time
from src.features.hemodynamics import apply_hrf_and_downsample, generate_double_gamma_hrf


FEATURES_IN_DIR = DIR_FEATURES_HIGH_RESOLUTION
FEATURES_OUT_DIR = DIR_FEATURES_FMRI_TR

# Lista ordenada de espacios según la configuración del anteproyecto
SPACES = ['phonological', 'lexical_stats', 'categorical', 'syntactic', 'semantic', 'tense']


def process_story_hrf(story_name: str, hrf_kernel: np.ndarray) -> str:
    """Consolida y transforma todos los espacios de características de una historia.
    
    Lee los 6 archivos Parquet a 100 Hz, los concatena en el orden estricto, 
    aplica la convolución HRF por FFT y remuestrea a 0.5 Hz.
    
    Args:
        story_name (str): Nombre de la historia a procesar.
        hrf_kernel (np.ndarray): Filtro HRF Doble-Gamma normalizado.
        
    Returns:
        str: Mensaje de registro (log) indicando el resultado de la transformación.
    """
    out_path = FEATURES_OUT_DIR / f"{story_name}_fmri_features.parquet"
    if out_path.exists():
        return f"Omitido: {story_name} ya transformada previamente (Checkpoint)."

    story_spaces = []
    
    try:
        # Cargar los 6 espacios generados en el script 02
        for space in SPACES:
            parquet_path = FEATURES_IN_DIR / f"{story_name}_{space}.parquet"
            if not parquet_path.exists():
                return f"[ERROR] En {story_name}: Espacio '{space}' no encontrado."
            
            df_space = pd.read_parquet(parquet_path)
            story_spaces.append(df_space)
            
        # Concatenación horizontal para formar la matriz global continua a 100 Hz
        df_global_high_res = pd.concat(story_spaces, axis=1)
        
        # Validación dimensional previa a la convolución
        if df_global_high_res.shape[1] != TOTAL_FEATURES_COUNT:
            return (
                f"[ERROR] En {story_name}: Dimensiones incorrectas ({df_global_high_res.shape[1]} cols, "
                f"se esperaban {TOTAL_FEATURES_COUNT})."
            )
            
        # Asegurar orden estricto de columnas
        df_global_high_res = df_global_high_res[FEATURE_COLUMNS_ORDER]
        
        # Aplicación de convolución hemodinámica y remuestreo polifásico al TR (0.5 Hz)
        df_global_fmri = apply_hrf_and_downsample(
            df_high_res=df_global_high_res, 
            hrf_kernel=hrf_kernel, 
            fs=HIGH_RES_FS, 
            tr=TR_FMRI
        )
        
        # Validación post-transformación
        assert df_global_fmri.shape[1] == TOTAL_FEATURES_COUNT, "Inconsistencia en columnas post-downsample"
        assert list(df_global_fmri.columns) == FEATURE_COLUMNS_ORDER, "Inconsistencia en orden post-downsample"
        
        # Exportar matriz predictora consolidada al TR del escáner en formato Parquet
        df_global_fmri.to_parquet(out_path, engine='pyarrow', index=True)
        
        return f"Éxito: {story_name} transformada y consolidada ({df_global_fmri.shape[0]} TRs, 43 cols)."
        
    except Exception as e:
        return f"[ERROR CRÍTICO] Procesando {story_name}: {str(e)}"


def run_hrf_pipeline() -> None:
    """Orquesta la aplicación de la HRF en paralelo para todo el corpus empírico."""
    FEATURES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_stories = load_table_to_dataframe('audit_stories')
    if df_stories is None or df_stories.empty:
        print("Error: No se encontró la tabla 'audit_stories'.")
        return
        
    stories = df_stories['story'].tolist()
    total_stories = len(stories)
    
    print("Generando kernel HRF Doble-Gamma Canónica (Friston et al., 1998, 32s)...")
    hrf_kernel = generate_double_gamma_hrf(HIGH_RES_FS, HRF_LENGTH_SEC)
    
    print(f"\nIniciando convolución FFT y remuestreo (0.5 Hz) en paralelo para {total_stories} historias...")
    
    max_workers = multiprocessing.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_story_hrf, story, hrf_kernel): story 
            for story in stories
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            print(f"[{completed}/{total_stories}] {result}")


if __name__ == "__main__":
    start_time = time.time()
    run_hrf_pipeline()
    log_execution_time("03_apply_hrf", time.time() - start_time)
