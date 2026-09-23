# src/features/hemodynamics.py
"""Módulo de transformación neurovascular temporal.

Implementa la función de respuesta hemodinámica (HRF) canónica de doble gamma
(estándar Friston / SPM) y la convolución temporal en el dominio de la frecuencia 
(FFT) con remuestreo polifásico anti-aliasing hacia la escala del TR fMRI (0.5 Hz).
"""

from typing import List

import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.signal import fftconvolve, resample_poly


def generate_double_gamma_hrf(fs: int, duration: float = 32.0) -> np.ndarray:
    """Genera una función HRF Doble-Gamma canónica (estándar SPM / Friston et al., 1998).
    
    Modela biológicamente el ascenso retardado de sangre oxigenada (pico aprox. a los 6s) 
    y el descenso/undershoot posterior (aprox. a los 16s) durante una ventana de 32 segundos.
    
    Args:
        fs (int): Frecuencia de muestreo original en Hz (100 Hz = paso de 10 ms).
        duration (float, opcional): Duración total de la ventana en segundos. Por defecto 32.0.
            
    Returns:
        np.ndarray: Vector 1D normalizado (suma = 1) que representa el filtro temporal HRF.
    """
    t = np.arange(0, duration, 1.0 / fs)
    
    # Parámetros biológicos canónicos de doble gamma
    a1, b1 = 6.0, 1.0  # Parámetros del pico principal
    a2, b2 = 16.0, 1.0 # Parámetros del undershoot
    c = 1.0 / 6.0      # Ratio de dispersión
    
    peak = (t**(a1 - 1) * np.exp(-t / b1)) / (b1**a1 * sp.gamma(a1))
    undershoot = (t**(a2 - 1) * np.exp(-t / b2)) / (b2**a2 * sp.gamma(a2))
    
    hrf = peak - (c * undershoot)
    
    # Normalización para evitar alterar artificialmente la escala de los predictores
    return hrf / np.sum(hrf)


def apply_hrf_and_downsample(
    df_high_res: pd.DataFrame, 
    hrf_kernel: np.ndarray, 
    fs: int, 
    tr: float
) -> pd.DataFrame:
    """Aplica convolución hemodinámica por FFT y reduce la resolución temporal al TR.
    
    Toma la matriz continua de características (100 Hz), convoluciona cada columna 
    con la HRF en el dominio de la frecuencia para máxima eficiencia computacional, 
    y aplica remuestreo polifásico con filtro anti-aliasing reduciendo la señal a 0.5 Hz.
    
    Args:
        df_high_res (pd.DataFrame): Matriz original de características a 100 Hz (43 cols).
        hrf_kernel (np.ndarray): Filtro temporal HRF normalizado.
        fs (int): Frecuencia de muestreo original de los predictores (100 Hz).
        tr (float): Tiempo de repetición del escáner fMRI (2.0 segundos).
        
    Returns:
        pd.DataFrame: Matriz hemodinámica transformada a 0.5 Hz (TR), conservando
            exactamente el orden y nombres de las 43 columnas originales.
    """
    n_samples, n_features = df_high_res.shape
    convolved_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    
    # Convolución independiente por columna vía Transformada Rápida de Fourier (FFT)
    for col_idx in range(n_features):
        feature_signal = df_high_res.iloc[:, col_idx].values
        convolved_signal = fftconvolve(feature_signal, hrf_kernel, mode='full')
        # Se descarta la cola transitoria posterior conservando la duración original
        convolved_matrix[:, col_idx] = convolved_signal[:n_samples]
        
    # Remuestreo polifásico anti-aliasing (Downsampling de 100 Hz a 0.5 Hz)
    down_factor = int(fs * tr)  # 100 * 2.0 = 200
    up_factor = 1
    
    downsampled_matrix = resample_poly(convolved_matrix, up=up_factor, down=down_factor, axis=0)
    
    n_tr_samples = downsampled_matrix.shape[0]
    tr_time_axis = np.arange(n_tr_samples) * tr
    
    df_fmri_space = pd.DataFrame(
        downsampled_matrix, 
        columns=df_high_res.columns, 
        index=tr_time_axis
    )
    df_fmri_space.index.name = 'time_tr_seconds'
    
    return df_fmri_space


def apply_fir_and_downsample(
    df_high_res: pd.DataFrame, 
    fs: int, 
    tr: float, 
    delays_in_trs: List[int] = [1, 2, 3, 4]
) -> pd.DataFrame:
    """Aplica transformación de Respuesta al Impulso Finito (FIR) mediante retardos.
    
    Nota metodológica: El anteproyecto fija la HRF canónica uniforme como aproximación 
    parsimoniosa principal para mantener 43 características y viabilidad en CPU. 
    Esta función se conserva como referencia metodológica auxiliar.
    """
    down_factor = int(fs * tr)
    up_factor = 1
    
    downsampled_matrix = resample_poly(df_high_res.values, up=up_factor, down=down_factor, axis=0)
    n_tr_samples = downsampled_matrix.shape[0]
    tr_time_axis = np.arange(n_tr_samples) * tr
    
    df_downsampled = pd.DataFrame(
        downsampled_matrix, 
        columns=df_high_res.columns, 
        index=tr_time_axis
    )
    
    lagged_dataframes = []
    for delay in delays_in_trs:
        df_shifted = df_downsampled.shift(delay, fill_value=0.0)
        df_shifted.columns = [f"{col}_lag{delay}" for col in df_downsampled.columns]
        lagged_dataframes.append(df_shifted)
        
    df_fir_space = pd.concat(lagged_dataframes, axis=1)
    df_fir_space.index.name = 'time_tr_seconds'
    return df_fir_space
