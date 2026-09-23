"""Módulo de Machine Learning Predictivo (Espacio Dual / Banded Ridge).

Encapsula la regresión de cresta con múltiples kernels (Banded Ridge) utilizando 
Himalaya. Permite regularizaciones independientes por espacio de características, 
incorporando Story-Blocked CV y evaluación por desplazamientos circulares exhaustivos.
"""

import gc
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from statsmodels.stats.multitest import multipletests

import himalaya
from himalaya.kernel_ridge import ColumnKernelizer, Kernelizer, MultipleKernelRidgeCV

from src.config import (
    CIRCULAR_SHIFT_MARGIN_MIN,
    HIMALAYA_BACKEND,
    RANDOM_SEED,
    RIDGE_ALPHAS,
    RIDGE_CV_FOLDS,
    SPACES_DIMENSIONS,
    STATISTICAL_ALPHA,
)


himalaya.backend.set_backend(HIMALAYA_BACKEND)


class VoxelwiseEncoder:
    """Modelo de codificación a nivel de vóxel basado en Banded Ridge Regression."""
    
    def __init__(
        self, 
        spaces_dimensions: Dict[str, int] = SPACES_DIMENSIONS, 
        random_state: int = RANDOM_SEED
    ):
        """Inicializa los particionadores por espacio de características (6 bandas)."""
        self.random_state = random_state
        self.spaces_dimensions = spaces_dimensions
        
        sizes = list(spaces_dimensions.values())
        names = list(spaces_dimensions.keys())
        start_end = np.concatenate([[0], np.cumsum(sizes)])
        self.slices = [slice(s, e) for s, e in zip(start_end[:-1], start_end[1:])]
        
        # Kernelizadores para Modelo Global (6 espacios)
        kernelizers_global = [
            (name, Kernelizer(kernel="linear"), slc) 
            for name, slc in zip(names, self.slices)
        ]
        self.col_kernelizer_global = ColumnKernelizer(kernelizers_global)
        
        # Kernelizadores para Modelo Restringido (5 espacios de control, excluyendo 'tense')
        kernelizers_restr = kernelizers_global[:-1]
        self.col_kernelizer_restr = ColumnKernelizer(kernelizers_restr)

    def fit_and_evaluate(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray, 
        story_ids_train: np.ndarray
    ) -> pd.DataFrame:
        """Ajusta Banded Ridge con Story-Blocked CV y desplazamientos exhaustivos."""
        n_voxels = y_test.shape[1]
        n_test_samples = y_test.shape[0]
        
        # Configuración de partición Story-Blocked
        unique_stories = np.unique(story_ids_train)
        n_splits = min(RIDGE_CV_FOLDS, len(unique_stories))
        gkf = GroupKFold(n_splits=n_splits)
        cv_splits = list(gkf.split(x_train, groups=story_ids_train))
        
        solver_params = dict(n_iter=20, alphas=RIDGE_ALPHAS)
        ridge_global = MultipleKernelRidgeCV(
            kernels="precomputed",
            solver="random_search",
            solver_params=solver_params,
            cv=cv_splits,
            random_state=self.random_state
        )
        ridge_restr = MultipleKernelRidgeCV(
            kernels="precomputed",
            solver="random_search",
            solver_params=solver_params,
            cv=cv_splits,
            random_state=self.random_state
        )
        
        # Ajuste y Predicción Modelo Global
        pipeline_global = make_pipeline(self.col_kernelizer_global, ridge_global)
        pipeline_global.fit(x_train, y_train)
        y_pred_g = pipeline_global.predict(x_test)
        r2_g = r2_score(y_test, y_pred_g, multioutput='raw_values')
        
        # Ajuste y Predicción Modelo Restringido
        pipeline_restr = make_pipeline(self.col_kernelizer_restr, ridge_restr)
        pipeline_restr.fit(x_train, y_train)
        y_pred_r = pipeline_restr.predict(x_test)
        r2_r = r2_score(y_test, y_pred_r, multioutput='raw_values')
        
        delta_r2 = r2_g - r2_r
        
        # Desplazamientos circulares exhaustivos
        valid_shifts = np.arange(CIRCULAR_SHIFT_MARGIN_MIN, n_test_samples - CIRCULAR_SHIFT_MARGIN_MIN)
        n_shifts = len(valid_shifts)
        
        null_distribution = np.zeros((n_shifts, n_voxels), dtype=np.float32)
        for s_idx, shift in enumerate(valid_shifts):
            pred_g_shifted = np.roll(y_pred_g, shift, axis=0)
            pred_r_shifted = np.roll(y_pred_r, shift, axis=0)
            
            r2_g_null = r2_score(y_test, pred_g_shifted, multioutput='raw_values')
            r2_r_null = r2_score(y_test, pred_r_shifted, multioutput='raw_values')
            null_distribution[s_idx, :] = r2_g_null - r2_r_null
            
        exceedance = np.sum(null_distribution >= delta_r2, axis=0)
        p_raw = (exceedance + 1.0) / (n_shifts + 1.0)
        
        _, p_fdr, _, _ = multipletests(p_raw, alpha=STATISTICAL_ALPHA, method='fdr_bh')
        
        df_results = pd.DataFrame({
            'voxel_idx': np.arange(n_voxels),
            'r2_global': r2_g,
            'r2_restricted': r2_r,
            'delta_r2_tense': delta_r2,
            'p_value_raw': p_raw,
            'p_value_fdr': p_fdr,
            'n_exhaustive_shifts': n_shifts,
            'p_resolution_min': 1.0 / (n_shifts + 1.0)
        })
        
        return df_results
