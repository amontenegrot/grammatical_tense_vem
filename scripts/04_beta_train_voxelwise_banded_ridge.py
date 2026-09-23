# scripts/04_beta_train_voxelwise_banded_ridge.py
"""Orquestador de Entrenamiento a Nivel de Sujeto (Banded Ridge / Múltiples Kernels).

Alternativa metodológica que implementa regularización diferenciada por espacio 
de características, conservando el promedio de repeticiones BOLD en prueba y Story-Blocked CV.
"""

import gc
import multiprocessing
import os
import time
from pathlib import Path

TOTAL_CORES = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = TOTAL_CORES
os.environ["OPENBLAS_NUM_THREADS"] = TOTAL_CORES
os.environ["MKL_NUM_THREADS"] = TOTAL_CORES

import numpy as np
import pandas as pd

from src.config import (
    DIR_PROCESSED,
    SPACES_DIMENSIONS,
)
from src.db_manager import load_table_to_dataframe, log_execution_time
from src.models.banded_ridge import VoxelwiseEncoder
from scripts.04_train_voxelwise_standard_ridge import build_matrices


RESULTS_OUT_DIR = DIR_PROCESSED / "results_voxelwise_banded"


def run_banded_subject_modeling() -> None:
    """Orquesta el ajuste de Banded Ridge por participante."""
    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_sessions = load_table_to_dataframe('audit_sessions')
    if df_sessions is None or df_sessions.empty:
        print("Error: No se encontró la tabla 'audit_sessions'.")
        return
        
    subjects = sorted(df_sessions['subject_id'].unique())
    encoder = VoxelwiseEncoder(SPACES_DIMENSIONS)
    
    for subject_id in subjects:
        start_time = time.time()
        print(f"\n--- Procesando Banded Ridge: {subject_id} ---")
        
        subject_out_file = RESULTS_OUT_DIR / f"{subject_id}_banded_results.parquet"
        if subject_out_file.exists():
            print(f"[OMITIDO] {subject_id} ya procesado en Banded Ridge.")
            continue
            
        subject_records = df_sessions[df_sessions['subject_id'] == subject_id]
        x_tr, y_tr, x_te, y_te, story_ids = build_matrices(subject_records)
        
        if x_tr is None:
            continue
            
        try:
            df_results = encoder.fit_and_evaluate(x_tr, y_tr, x_te, y_te, story_ids)
            df_results.to_parquet(subject_out_file, engine='pyarrow', index=False)
            print(f"[ÉXITO Banded] Persistido en {subject_out_file.name}")
        except Exception as e:
            print(f"[ERROR Banded] en {subject_id}: {str(e)}")
        finally:
            del x_tr, y_tr, x_te, y_te, story_ids
            gc.collect()
            elapsed = time.time() - start_time
            log_execution_time("04_banded_ridge", elapsed, subject_id)


if __name__ == "__main__":
    run_banded_subject_modeling()
