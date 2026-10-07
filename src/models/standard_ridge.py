# src/models/standard_ridge.py
"""Módulo de Machine Learning Predictivo (Espacio Primal / Ridge Estándar).

Implementa la Regresión de Cresta con penalización L2 común y procesamiento por 
lotes espaciales (chunking de 10.000 vóxeles) para garantizar eficiencia de RAM.
Incorpora Story-Blocked Cross-Validation (3 particiones por historias completas)
y validación estadística mediante desplazamientos circulares exhaustivos deterministas.
"""

import gc
from typing import Dict, List, Optional, Tuple

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
    NOISE_CEILING_MIN_THRESHOLD,
    NULL_DISTRIBUTION_CHUNK_BYTES,
    RANDOM_SEED,
    RIDGE_ALPHAS,
    RIDGE_CV_FOLDS,
    SPACES_DIMENSIONS,
    STATISTICAL_ALPHA,
    TOTAL_FEATURES_COUNT,
)


himalaya.backend.set_backend(HIMALAYA_BACKEND)


def _r2_score_batch(ss_res: np.ndarray, ss_tot: np.ndarray) -> np.ndarray:
    """Calcula R2 vectorizado para múltiples desplazamientos a la vez.

    Replica exactamente la semántica de sklearn.metrics.r2_score(multioutput='raw_values'),
    incluyendo sus casos límite cuando la varianza total (ss_tot) es cero:
    - ss_tot != 0: score = 1 - ss_res/ss_tot.
    - ss_tot == 0 y ss_res != 0: score = 0.0 (evita división por cero / -inf).
    - ss_tot == 0 y ss_res == 0: score = 1.0 (ajuste perfecto trivial).

    Args:
        ss_res (np.ndarray): Suma de cuadrados residual, forma (n_shifts, n_voxels).
        ss_tot (np.ndarray): Suma de cuadrados total (invariante al desplazamiento), forma (n_voxels,).

    Returns:
        np.ndarray: Puntajes R2, forma (n_shifts, n_voxels).
    """
    nonzero_denominator = ss_tot != 0
    nonzero_numerator = ss_res != 0

    with np.errstate(divide='ignore', invalid='ignore'):
        scores = 1.0 - ss_res / ss_tot[None, :]

    scores = np.where(nonzero_denominator[None, :], scores, 0.0)
    perfect_fit = (~nonzero_denominator[None, :]) & (~nonzero_numerator)
    scores = np.where(perfect_fit, 1.0, scores)
    return scores.astype(np.float32)


def compute_noise_ceiling(repeats: List[np.ndarray]) -> Optional[np.ndarray]:
    """Estima el techo de ruido (R2 máximo alcanzable) por vóxel a partir de repeticiones.

    Implementa el estimador de potencia de señal con corrección de sesgo de
    Sahani & Linden (2003): descompone la varianza temporal del promedio entre
    repeticiones en una componente de señal reproducible y una componente de
    ruido independiente entre repeticiones, y expresa el techo como la fracción
    de la varianza del promedio atribuible a la señal reproducible. Este es el
    R2 máximo que cualquier modelo (incluso uno perfecto) podría alcanzar contra
    el objetivo BOLD promediado entre repeticiones.

    Args:
        repeats (List[np.ndarray]): Matrices BOLD estandarizadas (TRs x Vóxeles),
            una por cada presentación repetida de la historia de prueba, ya
            recortadas a una longitud temporal común.

    Returns:
        Optional[np.ndarray]: Vector (Vóxeles,) con el R2 techo por vóxel, o None
            si hay menos de 2 repeticiones (la reproducibilidad no es estimable).
    """
    n_repeats = len(repeats)
    if n_repeats < 2:
        return None

    stacked = np.stack(repeats, axis=0)  # (R, T, V)
    y_mean = stacked.mean(axis=0)  # (T, V)

    var_mean = np.var(y_mean, axis=0)  # (V,)
    resid_var_per_repeat = np.var(stacked - y_mean[None, :, :], axis=1)  # (R, V)
    mean_resid_var = resid_var_per_repeat.mean(axis=0)  # (V,)

    # Corrección de sesgo: el residual de cada repetición respecto al promedio de
    # TODAS las repeticiones (que la incluye a sí misma) subestima la varianza de
    # ruido real en un factor (R-1)/R.
    noise_var = mean_resid_var * n_repeats / (n_repeats - 1)
    signal_var = np.clip(var_mean - noise_var / n_repeats, a_min=0.0, a_max=None)

    with np.errstate(divide='ignore', invalid='ignore'):
        ceiling = np.where(var_mean > 0, signal_var / var_mean, np.nan)

    return np.clip(ceiling, 0.0, 1.0).astype(np.float32)


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
        
        # Decisión técnica: se usa `raise` explícito en vez de `assert` porque esta
        # invariante debe cumplirse incluso si el proceso se ejecuta con `python -O`
        # (que descarta los `assert`).
        if self.total_features != TOTAL_FEATURES_COUNT:
            raise ValueError(
                f"Dimensiones globales inconsistentes: {self.total_features} != {TOTAL_FEATURES_COUNT}"
            )
        if self.restricted_limit != 39:
            raise ValueError(
                f"El modelo restringido debe tener exactamente 39 columnas de control, obtenido: {self.restricted_limit}"
            )

    def fit_and_evaluate(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        story_ids_train: np.ndarray,
        batch_size: int = 10000,
        noise_ceiling: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Ajusta los modelos global y restringido y ejecuta desplazamientos exhaustivos.

        Args:
            x_train (np.ndarray): Matriz predictora de entrenamiento (TRs x 43).
            y_train (np.ndarray): Matriz BOLD de entrenamiento (TRs x Vóxeles).
            x_test (np.ndarray): Matriz predictora de prueba (TRs x 43).
            y_test (np.ndarray): Matriz BOLD de prueba promediada temporalmente (TRs x Vóxeles).
            story_ids_train (np.ndarray): Vector con el identificador de historia de cada TR.
            batch_size (int): Tamaño del lote espacial de vóxeles.
            noise_ceiling (Optional[np.ndarray]): Vector (Vóxeles,) con el R2 techo estimado
                a partir de repeticiones de la historia de prueba (ver compute_noise_ceiling).
                Si es None (ej. participante con una sola presentación), las columnas
                normalizadas se reportan como NaN.

        Returns:
            pd.DataFrame: Resultados por vóxel con R2_global, R2_restringido, Delta_R2,
                p-value empírico, p-value corregido por FDR, el techo de ruido estimado
                y las versiones de R2/Delta_R2 normalizadas por dicho techo.
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

        # Métricas normalizadas por el techo de ruido (NaN por defecto: "no estimable").
        ceiling_full = np.full(n_voxels, np.nan, dtype=np.float32)
        r2_global_normalized_full = np.full(n_voxels, np.nan, dtype=np.float32)
        r2_restricted_normalized_full = np.full(n_voxels, np.nan, dtype=np.float32)
        delta_r2_normalized_full = np.full(n_voxels, np.nan, dtype=np.float32)

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

            # Normalización por techo de ruido (R2 máximo estimable con repeticiones BOLD)
            if noise_ceiling is not None:
                ceiling_batch = noise_ceiling[start_idx:end_idx]
                reliable = ceiling_batch > NOISE_CEILING_MIN_THRESHOLD
                with np.errstate(divide='ignore', invalid='ignore'):
                    r2_g_norm = np.where(reliable, r2_g / ceiling_batch, np.nan)
                    r2_r_norm = np.where(reliable, r2_r / ceiling_batch, np.nan)
                delta_r2_norm = r2_g_norm - r2_r_norm
            else:
                ceiling_batch = np.full(batch_len, np.nan, dtype=np.float32)
                r2_g_norm = np.full(batch_len, np.nan, dtype=np.float32)
                r2_r_norm = np.full(batch_len, np.nan, dtype=np.float32)
                delta_r2_norm = np.full(batch_len, np.nan, dtype=np.float32)

            # 6. Distribución Nula: Desplazamientos Circulares Exhaustivos (Sin reemplazo)
            # Vectorizado: SS_tot depende únicamente de y_te_batch (invariante al
            # desplazamiento), así que se calcula una sola vez en vez de recomputarlo
            # en cada una de las (potencialmente cientos de) iteraciones. Las predicciones
            # desplazadas se materializan en bloques como un arreglo 3D
            # (desplazamientos x tiempo x vóxeles) y R2 se calcula en bloque con numpy,
            # en vez de invocar np.roll + r2_score en un bucle puro de Python por desplazamiento.
            null_distribution = np.zeros((n_shifts, batch_len), dtype=np.float32)

            ss_tot_batch = np.sum(
                (y_te_batch - y_te_batch.mean(axis=0, keepdims=True)) ** 2, axis=0
            )
            time_idx = np.arange(n_test_samples)

            # Tamaño de bloque de desplazamientos simultáneos: acota la memoria de los
            # dos arreglos 3D temporales (global y restringido) al presupuesto configurado.
            bytes_per_shift_pair = n_test_samples * batch_len * 4 * 2
            shift_chunk_size = max(
                1, min(n_shifts, NULL_DISTRIBUTION_CHUNK_BYTES // max(bytes_per_shift_pair, 1))
            )

            for chunk_start in range(0, n_shifts, shift_chunk_size):
                chunk_end = min(chunk_start + shift_chunk_size, n_shifts)
                shift_chunk = valid_shifts[chunk_start:chunk_end]

                # Índices circulares: shifted[c, t, :] == np.roll(pred, shift_chunk[c], axis=0)[t, :]
                roll_idx = (time_idx[None, :] - shift_chunk[:, None]) % n_test_samples

                pred_g_shifted = y_pred_g[roll_idx]
                pred_r_shifted = y_pred_r[roll_idx]

                ss_res_g = np.sum((y_te_batch[None, :, :] - pred_g_shifted) ** 2, axis=1)
                ss_res_r = np.sum((y_te_batch[None, :, :] - pred_r_shifted) ** 2, axis=1)

                r2_g_null = _r2_score_batch(ss_res_g, ss_tot_batch)
                r2_r_null = _r2_score_batch(ss_res_r, ss_tot_batch)

                null_distribution[chunk_start:chunk_end, :] = r2_g_null - r2_r_null

                del roll_idx, pred_g_shifted, pred_r_shifted, ss_res_g, ss_res_r

            # Cálculo de valor p empírico con corrección de continuidad (+1)
            exceedance = np.sum(null_distribution >= delta_r2, axis=0)
            p_raw = (exceedance + 1.0) / (n_shifts + 1.0)
            
            # Almacenamiento en vectores consolidados
            r2_global_full[start_idx:end_idx] = r2_g
            r2_restricted_full[start_idx:end_idx] = r2_r
            delta_r2_full[start_idx:end_idx] = delta_r2
            p_raw_full[start_idx:end_idx] = p_raw

            ceiling_full[start_idx:end_idx] = ceiling_batch
            r2_global_normalized_full[start_idx:end_idx] = r2_g_norm
            r2_restricted_normalized_full[start_idx:end_idx] = r2_r_norm
            delta_r2_normalized_full[start_idx:end_idx] = delta_r2_norm
            
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
            'p_resolution_min': min_detectable_p,
            'r2_noise_ceiling': ceiling_full,
            'r2_global_normalized': r2_global_normalized_full,
            'r2_restricted_normalized': r2_restricted_normalized_full,
            'delta_r2_tense_normalized': delta_r2_normalized_full
        })
        
        return df_results
