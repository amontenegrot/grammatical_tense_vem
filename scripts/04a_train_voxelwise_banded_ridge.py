# scripts/04a_train_voxelwise_banded_ridge.py
#*
"""Orquestador de Entrenamiento a Nivel de Sujeto (Enfoque Banded Ridge).

Implementa una política estricta de punto de control (checkpointing).
Cruza las matrices predictoras con la señal fMRI individual utilizando 
Múltiples Kernels (Banded Ridge Regression) y guarda el mapa cortical 
resultante en disco. Permite reanudación automática ante interrupciones.
"""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DIR_FEATURES_FMRI_TR,
    DIR_PROCESSED,
    N_PERMUTATIONS,
    SPACES_DIMENSIONS,
    TEST_STORY,
)
from src.data_loader import load_fmri_data
from src.db_manager import load_table_to_dataframe, log_execution_time
from src.models.banded_ridge import VoxelwiseEncoder


FEATURES_IN_DIR = DIR_FEATURES_FMRI_TR
RESULTS_OUT_DIR = DIR_PROCESSED / "results_voxelwise"


def build_matrices(subject_sessions: pd.DataFrame) -> tuple:
    """Ensambla las matrices predictoras (X) y BOLD (Y) concatenando historias.
    
    Aplica conversión a float32 para optimización estricta de memoria, vital 
    para evitar colapsos (OOM) en la regresión de múltiples kernels.
    
    Args:
        subject_sessions (pd.DataFrame): Registros de historias para un sujeto.
        
    Returns:
        tuple: (x_train, y_train, x_test, y_test). Retorna (None, None, None, None) 
            si faltan archivos o datos.
    """
    x_train_list, y_train_list = [], []
    x_test, y_test = None, None
    
    for _, row in subject_sessions.iterrows():
        story = row['story']
        fmri_path = row['fmri_path']
        feature_path = FEATURES_IN_DIR / f"{story}_fmri_features.parquet"
        
        if not feature_path.exists():
            print(f"Advertencia: Características faltantes para {story}. Omitiendo.")
            continue
            
        y_data = load_fmri_data(fmri_path)
        if y_data is None:
            continue
            
        # Conversión temprana a precisión simple para dividir el peso en RAM a la mitad
        x_data = pd.read_parquet(feature_path).values.astype(np.float32)
        y_data = y_data.astype(np.float32)
        
        # Truncado de seguridad para alinear posibles diferencias mínimas de redondeo del TR
        min_samples = min(x_data.shape[0], y_data.shape[0])
        x_data = x_data[:min_samples, :]
        y_data = y_data[:min_samples, :]
        
        if story == TEST_STORY:
            if x_test is None: 
                x_test, y_test = x_data, y_data
        else:
            x_train_list.append(x_data)
            y_train_list.append(y_data)
            
    if not x_train_list or x_test is None:
        return None, None, None, None
        
    x_train_full = np.vstack(x_train_list)
    y_train_full = np.vstack(y_train_list)
    
    # Limpieza explícita de referencias intermedias en memoria
    del x_train_list, y_train_list
    gc.collect()
    
    return x_train_full, y_train_full, x_test, y_test

def run_subject_level_modeling() -> None:
    """Orquesta el ajuste de modelos por participante con checkpointing.
    
    Itera de forma ordenada sobre los sujetos, carga los datos transformados 
    y delega el entrenamiento y validación cruzada al VoxelwiseEncoder.
    Aplica liberación de memoria estricta al finalizar cada participante.
    """
    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_sessions = load_table_to_dataframe('audit_sessions')
    if df_sessions is None or df_sessions.empty:
        print("Error: No se encontró la tabla 'audit_sessions'.")
        return
        
    subjects = sorted(df_sessions['subject_id'].unique())
    encoder = VoxelwiseEncoder(SPACES_DIMENSIONS)
    
    for subject_id in subjects:
        start_time = time.time()
        print(f"\n--- Procesando Participante: {subject_id} ---")
        
        # Sistema de Checkpointing
        subject_out_file = RESULTS_OUT_DIR / f"{subject_id}_voxelwise_results.parquet"
        if subject_out_file.exists():
            print(f"[OMITIDO] El participante {subject_id} ya fue procesado previamente.")
            continue
            
        subject_records = df_sessions[df_sessions['subject_id'] == subject_id]
        
        print(f"Ensamblando matrices para {len(subject_records)} sesiones...")
        x_tr, y_tr, x_te, y_te = build_matrices(subject_records)
        
        if x_tr is None:
            print(f"[ERROR] Datos insuficientes para partición Train/Test en {subject_id}.")
            continue
            
        print(f"Entrenamiento: X={x_tr.shape}, Y={y_tr.shape}")
        print(f"Evaluación: X={x_te.shape}, Y={y_te.shape}")
        
        print(f"Entrenando Banded Ridge y calculando permutaciones ({N_PERMUTATIONS} iters)...")
        try:
            df_results = encoder.fit_and_evaluate(x_tr, y_tr, x_te, y_te, n_permutations=N_PERMUTATIONS)
            
            # Guardado atómico del progreso del participante
            df_results.to_parquet(subject_out_file, engine='pyarrow', index=False)
            print(f"[ÉXITO] Resultados persistidos en {subject_out_file.name}")
            
        except Exception as e:
            print(f"[ERROR] Fallo modelando al sujeto {subject_id}: {str(e)}")
            
        finally:
            # Limpieza de memoria
            del x_tr, y_tr, x_te, y_te
            if 'df_results' in locals():
                del df_results
            gc.collect()
            
            elapsed = time.time() - start_time
            log_execution_time("04a_banded_ridge", elapsed, subject_id)

if __name__ == "__main__":
    run_subject_level_modeling()
