# src/models/banded_ridge.py
"""Módulo de Machine Learning Predictivo (Espacio Dual / Múltiples Kernels).

Encapsula la lógica de Banded Ridge Regression usando Himalaya y la validación
estadística mediante permutación de desplazamiento circular. Este modelo 
optimiza hiperparámetros separados para cada espacio de características.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from statsmodels.stats.multitest import multipletests

import himalaya
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV

from src.config import (
    HIMALAYA_BACKEND, 
    RANDOM_SEED, 
    RIDGE_ALPHAS, 
    RIDGE_CV_FOLDS,
    STATISTICAL_ALPHA
)


himalaya.backend.set_backend(HIMALAYA_BACKEND)


class VoxelwiseEncoder:
    """Modelo de codificación a nivel de vóxel basado en Banded Ridge Regression."""
    
    def __init__(self, spaces_dimensions: Dict[str, int], random_state: int = RANDOM_SEED):
        """Inicializa los estimadores y el particionador de espacios de características.
        
        Args:
            spaces_dimensions (Dict[str, int]): Diccionario con el nombre de cada 
                espacio y su cantidad de columnas (dimensiones).
            random_state (int, opcional): Semilla para reproducibilidad.
        """
        self.random_state = random_state
        self.spaces_dimensions = spaces_dimensions
        
        # Configuración de los límites de las bandas (espacios)
        sizes = list(spaces_dimensions.values())
        names = list(spaces_dimensions.keys())
        start_end = np.concatenate([[0], np.cumsum(sizes)])
        self.slices = [slice(s, e) for s, e in zip(start_end[:-1], start_end[1:])]
        
        # Kernelizadores para Modelo Global
        kernelizers_global = [
            (name, Kernelizer(kernel="linear"), slc) 
            for name, slc in zip(names, self.slices)
        ]
        self.col_kernelizer_global = ColumnKernelizer(kernelizers_global)
        
        # Kernelizadores para Modelo Restringido (Ablación del último espacio: 'tense')
        kernelizers_restr = kernelizers_global[:-1]
        self.col_kernelizer_restr = ColumnKernelizer(kernelizers_restr)
        
        # Configuración de Ridge de Múltiples Kernels
        # solver_params utiliza RIDGE_ALPHAS desde config para el grid search interno
        solver_params = dict(n_iter=20, alphas=RIDGE_ALPHAS)
        self.ridge_model = MultipleKernelRidgeCV(
            kernels="precomputed",
            solver="random_search",
            solver_params=solver_params,
            cv=RIDGE_CV_FOLDS,
            random_state=self.random_state
        )

    def fit_and_evaluate(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray, 
        n_permutations: int = 1000
    ) -> pd.DataFrame:
        """Ajusta los modelos, calcula la varianza única y ejecuta permutación nula.
        
        Aplica estandarización estricta IN-PLACE sobre los conjuntos para evitar 
        fuga de datos (data leakage) y proteger la memoria RAM (evita copias).
        
        Args:
            x_train (np.ndarray): Matriz predictora de entrenamiento.
            y_train (np.ndarray): Matriz fMRI de entrenamiento.
            x_test (np.ndarray): Matriz predictora de evaluación.
            y_test (np.ndarray): Matriz fMRI de evaluación.
            n_permutations (int): Número de desplazamientos circulares.
                
        Returns:
            pd.DataFrame: Tabla de resultados por vóxel con R2, Delta R2 y p-values.
        """
        # 1. Limpieza de NaNs (In-Place)
        np.nan_to_num(x_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(y_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(x_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(y_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. Estandarización Estricta (In-Place para proteger la RAM)
        # Train X
        x_tr_mean = x_train.mean(axis=0)
        x_tr_std = x_train.std(axis=0)
        x_tr_std[x_tr_std == 0] = 1.0
        x_train -= x_tr_mean
        x_train /= x_tr_std
        # Test X
        x_test -= x_tr_mean
        x_test /= x_tr_std
        
        # Train Y
        y_tr_mean = y_train.mean(axis=0)
        y_tr_std = y_train.std(axis=0)
        y_tr_std[y_tr_std == 0] = 1.0
        y_train -= y_tr_mean
        y_train /= y_tr_std
        # Test Y
        y_test -= y_tr_mean
        y_test /= y_tr_std
        
        # 3. Entrenamiento y Predicción: Modelo Global
        pipeline_global = make_pipeline(self.col_kernelizer_global, self.ridge_model)
        pipeline_global.fit(x_train, y_train)
        y_pred_global = pipeline_global.predict(x_test)
        r2_global = r2_score(y_test, y_pred_global, multioutput='raw_values')
        
        # 4. Entrenamiento y Predicción: Modelo Restringido
        pipeline_restr = make_pipeline(self.col_kernelizer_restr, self.ridge_model)
        pipeline_restr.fit(x_train, y_train)
        y_pred_restr = pipeline_restr.predict(x_test)
        r2_restr = r2_score(y_test, y_pred_restr, multioutput='raw_values')
        
        # 5. Cálculo de Varianza Predictiva Única
        delta_r2 = r2_global - r2_restr
        
        # 6. Validación Estadística: Desplazamiento Circular (Circular Shift)
        n_test_samples = y_test.shape[0]
        n_voxels = y_test.shape[1]
        valid_shifts = np.arange(10, n_test_samples - 10)
        
        null_distribution = np.zeros((n_permutations, n_voxels), dtype=np.float32)
        
        np.random.seed(self.random_state)
        for i in range(n_permutations):
            shift = np.random.choice(valid_shifts)
            y_pred_g_shifted = np.roll(y_pred_global, shift, axis=0)
            y_pred_r_shifted = np.roll(y_pred_restr, shift, axis=0)
            
            r2_g_null = r2_score(y_test, y_pred_g_shifted, multioutput='raw_values')
            r2_r_null = r2_score(y_test, y_pred_r_shifted, multioutput='raw_values')
            
            null_distribution[i, :] = r2_g_null - r2_r_null
            
        # 7. Cálculo de valor-p empírico y corrección FDR (usando STATISTICAL_ALPHA)
        exceedance = np.sum(null_distribution >= delta_r2, axis=0)
        p_raw = (exceedance + 1) / (n_permutations + 1)
        
        _, p_fdr, _, _ = multipletests(p_raw, alpha=STATISTICAL_ALPHA, method='fdr_bh')
        
        df_results = pd.DataFrame({
            'voxel_idx': np.arange(n_voxels),
            'r2_global': r2_global,
            'r2_restricted': r2_restr,
            'delta_r2_tense': delta_r2,
            'p_value_raw': p_raw,
            'p_value_fdr': p_fdr
        })
        
        return df_results
