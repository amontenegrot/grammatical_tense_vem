# src/models/standard_ridge.py
"""Módulo de Machine Learning Predictivo (Espacio Primal / Ridge Estándar).

Implementa la Regresión de Cresta con penalización L2 común y procesamiento por 
lotes espaciales (chunking de 10.000 vóxeles) para garantizar eficiencia de RAM.
Incorpora Story-Blocked Cross-Validation (3 particiones por historias completas)
y validación estadística mediante desplazamientos circulares exhaustivos deterministas.
"""

import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests

import himalaya
from himalaya.ridge import RidgeCV

from src.config import (
    CIRCULAR_SHIFT_MARGIN_MIN,
    HIMALAYA_BACKEND,
    RANDOM_SEED,
    RIDGE_ALPHAS,
    RIDGE_CV_FOLDS,
    SPACES_DIMENSIONS,
    STATISTICAL_ALPHA,
    TOTAL_FEATURES_COUNT,
)


himalaya.backend.set_backend(HIMALAYA_BACKEND)


class StandardVoxelwiseEncoder:
    """Modelo de codificación Voxelwise basado en Ridge Estándar con Story-Blocked CV.
    
    Attributes:
        random_state (int): Semilla para reproducibilidad.
        spaces_dimensions (Dict[str, int]): Mapeo de espacios y sus dimensiones.
        total_features (int): Total de características del modelo global (43).
        tense_features (int): Total de características de tiempo gramatical (4).
        restricted_limit (int): Índice de corte para el modelo restringido (39).
    """
    
    def __init__(
        self, 
        spaces_dimensions: Dict[str, int] = SPACES_DIMENSIONS, 
        random_state: int = RANDOM_SEED
    ):
        """Inicializa los límites de corte dimensional para la ablación."""
        self.random_state = random_state
        self.spaces_dimensions = spaces_dimensions
        
        self.total_features = sum(spaces_dimensions.values())
        self.tense_features = spaces_dimensions.get('tense', 4)
        self.restricted_limit = self.total_features - self.tense_features
        
        assert self.total_features == TOTAL_FEATURES_COUNT, (
            f"Dimensiones globales inconsistentes: {self.total_features} != {TOTAL_FEATURES_COUNT}"
        )
        assert self.restricted_limit == 39, (
            f"El modelo restringido debe tener exactamente 39 columnas de control, obtenido: {self.restricted_limit}"
        )

    def fit_and_evaluate(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray, 
        story_ids_train: np.ndarray,
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """Ajusta los modelos global y restringido y ejecuta desplazamientos exhaustivos.
        
        Args:
            x_train (np.ndarray): Matriz predictora de entrenamiento (TRs x 43).
            y_train (np.ndarray): Matriz BOLD de entrenamiento (TRs x Vóxeles).
            x_test (np.ndarray): Matriz predictora de prueba (TRs x 43).
            y_test (np.ndarray): Matriz BOLD de prueba promediada temporalmente (TRs x Vóxeles).
            story_ids_train (np.ndarray): Vector con el identificador de historia de cada TR.
            batch_size (int): Tamaño del lote espacial de vóxeles.
            
        Returns:
            pd.DataFrame: Resultados por vóxel con R2_global, R2_restringido, Delta_R2,
                p-value empírico y p-value corregido por FDR.
        """
        n_voxels = y_train.shape[1]
        n_test_samples = y_test.shape[0]
        
        # 1. Configuración de Story-Blocked Cross-Validation
        # Agrupa muestras temporales contiguas de una misma narración completa en el mismo pliegue
        # para evitar contaminación por autocorrelación hemodinámica entre TRs contiguos.
        unique_stories = np.unique(story_ids_train)
        n_splits = min(RIDGE_CV_FOLDS, len(unique_stories))
        
        gkf = GroupKFold(n_splits=n_splits)
        cv_splits = list(gkf.split(x_train, groups=story_ids_train))
        
        # Instanciar modelos Ridge independientes para la selección de hiperparámetros
        global_model = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv_splits)
        restricted_model = RidgeCV(alphas=RIDGE_ALPHAS, cv=cv_splits)
        
        # 2. Partición de características para el Modelo Restringido (Primeras 39 columnas de control)
        x_train_restricted = x_train[:, :self.restricted_limit]
        x_test_restricted = x_test[:, :self.restricted_limit]
        
        # 3. Definición de desplazamientos circulares exhaustivos admisibles
        # Se evalúan todas las rotaciones posibles excluyendo los márgenes de correlación residual
        valid_shifts = np.arange(CIRCULAR_SHIFT_MARGIN_MIN, n_test_samples - CIRCULAR_SHIFT_MARGIN_MIN)
        n_shifts = len(valid_shifts)
        
        if n_shifts < 1:
            raise ValueError(
                f"La longitud de prueba ({n_test_samples} TRs) es insuficiente para el margen de {CIRCULAR_SHIFT_MARGIN_MIN} TRs."
            )
            
        # 4. Pre-asignación de memoria para resultados
        r2_global_full = np.zeros(n_voxels, dtype=np.float32)
        r2_restricted_full = np.zeros(n_voxels, dtype=np.float32)
        delta_r2_full = np.zeros(n_voxels, dtype=np.float32)
        p_raw_full = np.zeros(n_voxels, dtype=np.float32)
        
        # 5. Procesamiento espacial por lotes (Chunking para control estricto de RAM)
        for start_idx in range(0, n_voxels, batch_size):
            end_idx = min(start_idx + batch_size, n_voxels)
            batch_len = end_idx - start_idx
            
            y_tr_batch = y_train[:, start_idx:end_idx]
            y_te_batch = y_test[:, start_idx:end_idx]
            
            # Ajuste y Predicción: Modelo Global (43 características)
            global_model.fit(x_train, y_tr_batch)
            y_pred_g = global_model.predict(x_test)
            r2_g = r2_score(y_te_batch, y_pred_g, multioutput='raw_values')
            
            # Ajuste y Predicción: Modelo Restringido (39 características de control)
            restricted_model.fit(x_train_restricted, y_tr_batch)
            y_pred_r = restricted_model.predict(x_test_restricted)
            r2_r = r2_score(y_te_batch, y_pred_r, multioutput='raw_values')
            
            # Aporte predictivo puntual del espacio de interés
            delta_r2 = r2_g - r2_r
            
            # 6. Distribución Nula: Desplazamientos Circulares Exhaustivos (Sin reemplazo)
            null_distribution = np.zeros((n_shifts, batch_len), dtype=np.float32)
            
            for s_idx, shift in enumerate(valid_shifts):
                pred_g_shifted = np.roll(y_pred_g, shift, axis=0)
                pred_r_shifted = np.roll(y_pred_r, shift, axis=0)
                
                r2_g_null = r2_score(y_te_batch, pred_g_shifted, multioutput='raw_values')
                r2_r_null = r2_score(y_te_batch, pred_r_shifted, multioutput='raw_values')
                
                null_distribution[s_idx, :] = r2_g_null - r2_r_null
                
            # Cálculo de valor p empírico con corrección de continuidad (+1)
            exceedance = np.sum(null_distribution >= delta_r2, axis=0)
            p_raw = (exceedance + 1.0) / (n_shifts + 1.0)
            
            # Almacenamiento en vectores consolidados
            r2_global_full[start_idx:end_idx] = r2_g
            r2_restricted_full[start_idx:end_idx] = r2_r
            delta_r2_full[start_idx:end_idx] = delta_r2
            p_raw_full[start_idx:end_idx] = p_raw
            
            # Liberación explícita de referencias y recolección de basura
            del y_tr_batch, y_te_batch, y_pred_g, y_pred_r, null_distribution
            gc.collect()
            
        # 7. Control de la Tasa de Falsos Descubrimientos (FDR Benjamini-Hochberg)
        _, p_fdr, _, _ = multipletests(p_raw_full, alpha=STATISTICAL_ALPHA, method='fdr_bh')
        
        # Resolución estadística empírica mínima alcanzable
        min_detectable_p = 1.0 / (n_shifts + 1.0)
        
        df_results = pd.DataFrame({
            'voxel_idx': np.arange(n_voxels),
            'r2_global': r2_global_full,
            'r2_restricted': r2_restricted_full,
            'delta_r2_tense': delta_r2_full,
            'p_value_raw': p_raw_full,
            'p_value_fdr': p_fdr,
            'n_exhaustive_shifts': n_shifts,
            'p_resolution_min': min_detectable_p
        })
        
        return df_results
