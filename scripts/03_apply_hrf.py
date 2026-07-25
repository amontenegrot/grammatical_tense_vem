# scripts/03_apply_hrf.py
#*
"""Orquestador de Transformación Hemodinámica.

Lee los espacios de alta resolución exportados en la Fase 2, ejecuta la 
convolución FFT y reducción de dimensionalidad en paralelo. Consolida todos 
los espacios en una matriz global de características alineada al TR del fMRI.
"""

import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.config import (
    HIGH_RES_FS, 
    TR_FMRI, 
    HRF_LENGTH_SEC,
    DIR_FEATURES_HIGH_RESOLUTION,
    DIR_FEATURES_FMRI_TR,
    SPACES_DIMENSIONS,
)
import time
from src.db_manager import save_dataframe_to_table, log_execution_time, load_table_to_dataframe
from src.features.hemodynamics import generate_double_gamma_hrf, apply_hrf_and_downsample


FEATURES_IN_DIR = DIR_FEATURES_HIGH_RESOLUTION
FEATURES_OUT_DIR = DIR_FEATURES_FMRI_TR

SPACES = list(SPACES_DIMENSIONS.keys())


def process_story_hrf(story_name: str, hrf_kernel: np.ndarray) -> str:
    """
    Consolida y transforma todos los espacios de características de una historia.
    
    Función diseñada para ser ejecutada concurrentemente. Lee los archivos Parquet, 
    los concatena en una matriz continua a 100 Hz, aplica la convolución BOLD 
    y remuestrea la frecuencia para coincidir con la adquisición fMRI.
    
    Args:
        story_name (str): Nombre de la historia a procesar.
        hrf_kernel (np.ndarray): Filtro HRF Doble-Gamma previamente instanciado.
        
    Returns:
        str: Mensaje de registro (log) indicando el éxito o fallo de la ejecución.
    """
    out_path = FEATURES_OUT_DIR / f"{story_name}_fmri_features.parquet"
    if out_path.exists():
        return f"Omitido: {story_name} ya transformada previamente (Checkpoint)."

    story_spaces = []
    
    try:
        # Cargar todos los espacios de la historia generados en el script 02
        for space in SPACES:
            parquet_path = FEATURES_IN_DIR / f"{story_name}_{space}.parquet"
            if not parquet_path.exists():
                return f"Error en {story_name}: Espacio {space} no encontrado."
            
            df_space = pd.read_parquet(parquet_path)
            story_spaces.append(df_space)
            
        # Concatenación horizontal (column-wise) para formar la matriz global X a 100 Hz
        df_global_high_res = pd.concat(story_spaces, axis=1)
        
        # Aplicar metamorfosis BOLD (Convolución + Downsampling)
        df_global_fmri = apply_hrf_and_downsample(
            df_global_high_res, 
            hrf_kernel, 
            HIGH_RES_FS, 
            TR_FMRI
        )
        
        # Exportar matriz global consolidada al TR del escáner
        out_path = FEATURES_OUT_DIR / f"{story_name}_fmri_features.parquet"
        df_global_fmri.to_parquet(out_path, engine='pyarrow', index=True)
        
        return f"Exito: {story_name} transformada y consolidada ({df_global_fmri.shape[0]} TRs)."
        
    except Exception as e:
        return f"Error procesando {story_name}: {str(e)}"

def run_hrf_pipeline() -> None:
    """
    Orquesta la aplicación de la HRF en paralelo para todo el corpus.
    
    Genera el kernel hemodinámico global y distribuye el procesamiento de las 
    historias a través de un pool de procesos (ProcessPoolExecutor), maximizando 
    el uso de la CPU.
    """
    FEATURES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_stories = load_table_to_dataframe('audit_stories')
    if df_stories is None or df_stories.empty:
        print("Error: No se encontró la tabla 'audit_stories'.")
        return
        
    stories = df_stories['story'].tolist()
    total_stories = len(stories)
    
    print("Generando kernel HRF Double-Gamma...")
    hrf_kernel = generate_double_gamma_hrf(HIGH_RES_FS, HRF_LENGTH_SEC)
    
    print(f"Iniciando convolución FFT y remuestreo en paralelo para {total_stories} historias...")
    
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
