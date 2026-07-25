# src/models/standard_ridge.py
#*
"""Módulo de Machine Learning Predictivo (Espacio Primal).

Implementa la Regresión de Cresta (Ridge) con procesamiento por lotes (chunking)
sobre el espacio de vóxeles para garantizar eficiencia absoluta de RAM.
"""

import gc
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import multipletests

import himalaya
from himalaya.ridge import RidgeCV

from src.config import (
    HIMALAYA_BACKEND, 
    RANDOM_SEED,
    RIDGE_ALPHAS, 
    RIDGE_CV_FOLDS,
    STATISTICAL_ALPHA,
)


himalaya.backend.set_backend(HIMALAYA_BACKEND)


class StandardVoxelwiseEncoder:
    """Modelo de codificación basado en Ridge Estándar con Chunking espacial.
    
    Esta clase maneja el ajuste de hiperparámetros y la evaluación estadística 
    protegiendo la memoria RAM mediante el particionamiento de los datos fMRI.
    
    Attributes:
        random_state (int): Semilla para asegurar reproducibilidad.
        spaces_dimensions (Dict[str, int]): Mapeo de espacios y sus dimensiones.
        total_features (int): Suma de todas las dimensiones predictoras.
        tense_features (int): Cantidad de dimensiones del espacio de interés.
        restricted_limit (int): Índice de corte para el modelo restringido (ablación).
        global_model (RidgeCV): Instancia del modelo que usa todas las características.
        restricted_model (RidgeCV): Instancia del modelo que omite el tiempo gramatical.
    """
    
    def __init__(self, spaces_dimensions: Dict[str, int], random_state: int = RANDOM_SEED):
        """Inicializa el modelo calculando dinámicamente los índices de ablación.
        
        Args:
            spaces_dimensions (Dict[str, int]): Diccionario con nombres de los espacios 
                y sus dimensiones predictoras correspondientes.
            random_state (int, opcional): Semilla para la generación de particiones 
                y permutaciones. Por defecto asume la variable global RANDOM_SEED.
        """
        self.random_state = random_state
        self.spaces_dimensions = spaces_dimensions
        
        # Cálculo dinámico para evitar cortes manuales frágiles
        self.total_features = sum(spaces_dimensions.values())
        self.tense_features = spaces_dimensions.get('tense', 6)
        self.restricted_limit = self.total_features - self.tense_features
        
        # Modelos independientes con validación cruzada interna dinámica
        self.global_model = RidgeCV(alphas=RIDGE_ALPHAS, cv=RIDGE_CV_FOLDS)
        self.restricted_model = RidgeCV(alphas=RIDGE_ALPHAS, cv=RIDGE_CV_FOLDS)

    def fit_and_evaluate(
        self, 
        x_train: np.ndarray, 
        y_train: np.ndarray, 
        x_test: np.ndarray, 
        y_test: np.ndarray, 
        n_permutations: int,
        batch_size: int
    ) -> pd.DataFrame:
        """Ajusta modelos y evalúa significancia estadística operando por lotes.
        
        Decisión técnica fundamental: La matriz fMRI (`y_train`) se fragmenta en 
        lotes (`batch_size`) a lo largo del eje espacial (vóxeles). Esto impide 
        que Himalaya genere matrices tridimensionales de error que colapsen la RAM.
        
        Args:
            x_train: Matriz predictora de entrenamiento (muestras temporales, características).
            y_train: Matriz fMRI de entrenamiento (muestras temporales, vóxeles totales).
            x_test: Matriz predictora de evaluación.
            y_test: Matriz fMRI de evaluación.
            n_permutations: Cantidad de desplazamientos circulares para distribución nula.
            batch_size: Cantidad máxima de vóxeles a procesar por iteración.
            
        Returns:
            pd.DataFrame: Tabla con vóxeles, R2 de ambos modelos, Delta R2, y valores p.
        """
        n_voxels = y_train.shape[1]
        n_test_samples = y_test.shape[0]
        
        # 1. Limpieza y estandarización de X 
        # Decisión técnica: X es muy pequeña (45 cols), se limpia globalmente sin riesgo de RAM.
        x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        x_tr_mean = x_train.mean(axis=0)
        x_tr_std = x_train.std(axis=0)
        x_tr_std[x_tr_std == 0] = 1.0  # Prevención de división por cero
        
        # Operaciones "in-place" para no crear copias en memoria
        x_train -= x_tr_mean
        x_train /= x_tr_std
        x_test -= x_tr_mean
        x_test /= x_tr_std
        
        x_train_restricted = x_train[:, :self.restricted_limit]
        x_test_restricted = x_test[:, :self.restricted_limit]
        
        # 2. Pre-asignación de arrays unidimensionales para los resultados finales
        r2_global_full = np.zeros(n_voxels, dtype=np.float32)
        r2_restricted_full = np.zeros(n_voxels, dtype=np.float32)
        delta_r2_full = np.zeros(n_voxels, dtype=np.float32)
        p_raw_full = np.zeros(n_voxels, dtype=np.float32)
        
        # Exclusión de bordes para permutaciones válidas (evita correlación residual de HRF)
        valid_shifts = np.arange(10, n_test_samples - 10)
        
        # 3. Procesamiento Espacial por Lotes (Chunking)
        for start_idx in range(0, n_voxels, batch_size):
            end_idx = min(start_idx + batch_size, n_voxels)
            print(f"      -> Procesando bloque de vóxeles [{start_idx}:{end_idx}] de {n_voxels}...")
            
            # Extracción y limpieza exclusiva del bloque (ahorro masivo de RAM)
            y_tr_batch = np.nan_to_num(y_train[:, start_idx:end_idx], nan=0.0).astype(np.float32)
            y_te_batch = np.nan_to_num(y_test[:, start_idx:end_idx], nan=0.0).astype(np.float32)
            
            # Estandarización estricta del bloque
            y_mean = y_tr_batch.mean(axis=0)
            y_std = y_tr_batch.std(axis=0)
            y_std[y_std == 0] = 1.0
            
            y_tr_batch -= y_mean
            y_tr_batch /= y_std
            y_te_batch -= y_mean
            y_te_batch /= y_std
            
            # Entrenamiento y Predicción: Modelo Global
            self.global_model.fit(x_train, y_tr_batch)
            y_pred_g = self.global_model.predict(x_test)
            r2_g = r2_score(y_te_batch, y_pred_g, multioutput='raw_values')
            
            # Entrenamiento y Predicción: Modelo Restringido
            self.restricted_model.fit(x_train_restricted, y_tr_batch)
            y_pred_r = self.restricted_model.predict(x_test_restricted)
            r2_r = r2_score(y_te_batch, y_pred_r, multioutput='raw_values')
            
            # Variancia Predictiva Única
            delta_r2 = r2_g - r2_r
            
            # Validación Estadística mediante Desplazamiento Circular
            null_dist = np.zeros((n_permutations, end_idx - start_idx), dtype=np.float32)
            np.random.seed(self.random_state)
            
            for i in range(n_permutations):
                shift = np.random.choice(valid_shifts)
                pred_g_shifted = np.roll(y_pred_g, shift, axis=0)
                pred_r_shifted = np.roll(y_pred_r, shift, axis=0)
                
                r2_g_null = r2_score(y_te_batch, pred_g_shifted, multioutput='raw_values')
                r2_r_null = r2_score(y_te_batch, pred_r_shifted, multioutput='raw_values')
                
                null_dist[i, :] = r2_g_null - r2_r_null
                
            # Cálculo de valor p empírico
            exceedance = np.sum(null_dist >= delta_r2, axis=0)
            p_raw = (exceedance + 1) / (n_permutations + 1)
            
            # Almacenamiento en vectores globales
            r2_global_full[start_idx:end_idx] = r2_g
            r2_restricted_full[start_idx:end_idx] = r2_r
            delta_r2_full[start_idx:end_idx] = delta_r2
            p_raw_full[start_idx:end_idx] = p_raw
            
            # Decisión técnica: Recolección explícita de basura para mitigar el 90% de ocupación de RAM.
            # Limpia los arrays temporales de Himalaya al finalizar cada lote.
            del y_tr_batch, y_te_batch, y_pred_g, y_pred_r, null_dist
            gc.collect()
            
        # 4. Corrección de Múltiples Comparaciones (FDR) con parámetro dinámico
        print("      -> Calculando corrección FDR global...")
        _, p_fdr, _, _ = multipletests(p_raw_full, alpha=STATISTICAL_ALPHA, method='fdr_bh')
        
        df_results = pd.DataFrame({
            'voxel_idx': np.arange(n_voxels),
            'r2_global': r2_global_full,
            'r2_restricted': r2_restricted_full,
            'delta_r2_tense': delta_r2_full,
            'p_value_raw': p_raw_full,
            'p_value_fdr': p_fdr
        })
        
        return df_results
