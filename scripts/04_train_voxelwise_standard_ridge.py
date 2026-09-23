# scripts/04_train_voxelwise_standard_ridge.py
"""Orquestador de Entrenamiento a Nivel de Sujeto (Ridge Estándar / Primal).

Implementa extracción, ensamblaje, estandarización independiente por sesión,
promedio temporal de repeticiones BOLD en prueba y ajuste vóxel a vóxel
mediante procesamiento espacial por lotes para preservar los límites de RAM.
Mantiene política estricta de punto de control (checkpointing).
"""

import gc
import multiprocessing
import os
import time
from typing import Optional, Tuple

# Decisión técnica: Asegura que el backend en C subyacente (OpenBLAS/MKL)
# vectorice las operaciones matemáticas ocupando el 100% de la CPU lógica.
# Debe ejecutarse antes de la importación de Numpy/Pandas.
TOTAL_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = TOTAL_CORES
os.environ["OPENBLAS_NUM_THREADS"] = TOTAL_CORES
os.environ["MKL_NUM_THREADS"] = TOTAL_CORES
os.environ["VECLIB_MAXIMUM_THREADS"] = TOTAL_CORES
os.environ["NUMEXPR_NUM_THREADS"] = TOTAL_CORES

import numpy as np
import pandas as pd

from src.config import (
    DIR_FEATURES_FMRI_TR,
    DIR_PROCESSED,
    SPACES_DIMENSIONS,
    TEST_STORY,
    TOTAL_FEATURES_COUNT,
    VOXEL_BATCH_SIZE,
)
from src.data_loader import load_fmri_data
from src.db_manager import load_table_to_dataframe, log_execution_time
from src.models.standard_ridge import StandardVoxelwiseEncoder


FEATURES_IN_DIR = DIR_FEATURES_FMRI_TR
RESULTS_OUT_DIR = DIR_PROCESSED / "results_voxelwise"


def build_matrices(subject_sessions: pd.DataFrame) -> Tuple:
    """Ensambla las matrices de entrenamiento y evaluación fuera de muestra.
    
    Aplica una conversión estricta a float32 y estandarización (Z-score) 
    independiente por sesión para neutralizar las fluctuaciones de línea base del escáner.
    Para la historia reservada ('wheretheressmoke'), promedia temporalmente las 
    repeticiones BOLD estandarizadas para maximizar la relación señal-ruido (SNR).
    
    Args:
        subject_sessions (pd.DataFrame): Registros de auditoría del participante.
        
    Returns:
        Tuple: (x_train, y_train, x_test, y_test, story_ids_train). Retorna Nones si faltan datos.
    """
    x_train_list, y_train_list, story_ids_list = [], [], []
    test_bold_runs = []
    x_test = None
    
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
            
        # Coerción inmediata a precisión simple para reducir el peso en RAM al 50%
        x_data = pd.read_parquet(feature_path).values.astype(np.float32)
        y_data = y_data.astype(np.float32)
        
        # Truncado de seguridad para emparejar posibles discrepancias mínimas de TR
        min_samples = min(x_data.shape[0], y_data.shape[0])
        x_data = x_data[:min_samples, :]
        y_data = y_data[:min_samples, :]
        
        # Decisión técnica: Z-scoring INDEPENDIENTE POR SESIÓN/HISTORIA.
        # El escáner fMRI no tiene un cero absoluto y la línea base "flota" entre sesiones.
        # Si no se estandariza cada historia por separado antes de concatenarlas,
        # el modelo Ridge intentará predecir los saltos abruptos entre sesiones y fallará.
        
        # Estandarización de características predictoras (X)
        x_data = np.nan_to_num(x_data, nan=0.0)
        x_mean = x_data.mean(axis=0)
        x_std = x_data.std(axis=0)
        x_std[x_std == 0] = 1.0
        x_data = (x_data - x_mean) / x_std
        
        # Estandarización de la serie temporal BOLD (Y)
        y_data = np.nan_to_num(y_data, nan=0.0)
        y_mean = y_data.mean(axis=0)
        y_std = y_data.std(axis=0)
        y_std[y_std == 0] = 1.0
        y_data = (y_data - y_mean) / y_std
        
        if story == TEST_STORY:
            test_bold_runs.append(y_data)
            if x_test is None:
                x_test = x_data
        else:
            x_train_list.append(x_data)
            y_train_list.append(y_data)
            story_ids_list.append(np.full(min_samples, story))
            
    if not x_train_list or not test_bold_runs or x_test is None:
        return None, None, None, None, None
        
    # Ensamblaje de Entrenamiento
    x_train_full = np.vstack(x_train_list)
    y_train_full = np.vstack(y_train_list)
    story_ids_train = np.concatenate(story_ids_list)
    
    # Decisión metodológica del anteproyecto: Promedio temporal de repeticiones de prueba
    # Se calcula el promedio elemento a elemento de las matrices BOLD estandarizadas de test
    # para atenuar el ruido fisiológico no sincronizado con el estímulo.
    if len(test_bold_runs) > 1:
        # Asegurar longitud común entre repeticiones de prueba si difirieran por 1 TR
        min_test_len = min(run.shape[0] for run in test_bold_runs)
        test_bold_trimmed = [run[:min_test_len, :] for run in test_bold_runs]
        y_test_avg = np.mean(test_bold_trimmed, axis=0, dtype=np.float32)
        x_test = x_test[:min_test_len, :]
    else:
        y_test_avg = test_bold_runs[0]
        
    del x_train_list, y_train_list, test_bold_runs
    gc.collect()
    
    return x_train_full, y_train_full, x_test, y_test_avg, story_ids_train


def run_subject_level_modeling() -> None:
    """Orquesta el ajuste de modelos por participante con punto de control."""
    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_sessions = load_table_to_dataframe('audit_sessions')
    if df_sessions is None or df_sessions.empty:
        print("Error: No se encontró la tabla 'audit_sessions'. Ejecute 01a primero.")
        return
        
    subjects = sorted(df_sessions['subject_id'].unique())
    encoder = StandardVoxelwiseEncoder(SPACES_DIMENSIONS)
    
    for subject_id in subjects:
        start_time = time.time()
        print(f"\n=================================================================")
        print(f"PROCESANDO PARTICIPANTE: {subject_id}")
        print(f"=================================================================")
        
        subject_out_file = RESULTS_OUT_DIR / f"{subject_id}_voxelwise_results.parquet"
        if subject_out_file.exists():
            print(f"[OMITIDO] El participante {subject_id} ya fue procesado previamente.")
            continue
            
        subject_records = df_sessions[df_sessions['subject_id'] == subject_id]
        print(f"Ensamblando matrices para {len(subject_records)} sesiones...")
        
        x_tr, y_tr, x_te, y_te, story_ids = build_matrices(subject_records)
        if x_tr is None:
            print(f"[ERROR] Datos insuficientes para partición Train/Test en {subject_id}.")
            continue
            
        print(f"Entrenamiento: X={x_tr.shape} (43 predictores), Y={y_tr.shape}")
        print(f"Evaluación (Promedio SNR): X={x_te.shape}, Y={y_te.shape}")
        print(f"Historias en entrenamiento: {len(np.unique(story_ids))} narraciones para Story-Blocked CV.")
        
        try:
            df_results = encoder.fit_and_evaluate(
                x_train=x_tr, 
                y_train=y_tr, 
                x_test=x_te, 
                y_test=y_te, 
                story_ids_train=story_ids,
                batch_size=VOXEL_BATCH_SIZE
            )
            
            # Persistencia atómica de resultados
            df_results.to_parquet(subject_out_file, engine='pyarrow', index=False)
            print(f"[ÉXITO] Resultados persistidos en {subject_out_file.name}")
            
        except Exception as e:
            print(f"[ERROR CRÍTICO] Falló el modelamiento para {subject_id}: {str(e)}")
            
        finally:
            del x_tr, y_tr, x_te, y_te, story_ids
            if 'df_results' in locals():
                del df_results
            gc.collect()
            
            elapsed = time.time() - start_time
            log_execution_time("04_standard_ridge", elapsed, subject_id)


if __name__ == "__main__":
    print(f"Iniciando pipeline VEM. OpenBLAS configurado para usar {TOTAL_CORES} hilos lógicos.")
    run_subject_level_modeling()
