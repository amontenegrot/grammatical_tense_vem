# src/features/hemodynamics.py
"""Módulo de transformación neurovascular temporal.

Contiene las funciones matemáticas para modelar la relación entre el estímulo
y la señal BOLD, ofreciendo dos enfoques:
1. Convolución con Respuesta Hemodinámica (HRF) Canónica (Double-Gamma).
2. Expansión por Respuesta al Impulso Finito (FIR) mediante retardos temporales.
"""

from typing import List

import numpy as np
import pandas as pd
import scipy.special as sp
from scipy.signal import fftconvolve, resample_poly


def generate_double_gamma_hrf(fs: int, duration: float = 32.0) -> np.ndarray:
    """Genera una función HRF Double-Gamma canónica (estándar SPM).
    
    Modela biológicamente el pico inicial de sangre oxigenada (aprox. a los 6s) 
    y la caída posterior o undershoot (aprox. a los 16s). Los parámetros de la 
    curva se mantienen estáticos como constantes neurobiológicas.
    
    Args:
        fs (int): Frecuencia de muestreo original en Hz (resolución de las características).
        duration (float, opcional): Duración total de la ventana de la respuesta 
            en segundos. Por defecto es 32.0.
            
    Returns:
        np.ndarray: Vector 1D normalizado (suma = 1) que representa el filtro temporal HRF.
    """
    t = np.arange(0, duration, 1.0 / fs)
    
    # Parámetros biológicos estándar de SPM
    a1, b1 = 6.0, 1.0  # Parámetros del pico principal
    a2, b2 = 16.0, 1.0 # Parámetros del undershoot
    c = 1.0 / 6.0      # Ratio de dispersión
    
    peak = (t**(a1 - 1) * np.exp(-t / b1)) / (b1**a1 * sp.gamma(a1))
    undershoot = (t**(a2 - 1) * np.exp(-t / b2)) / (b2**a2 * sp.gamma(a2))
    
    hrf = peak - (c * undershoot)
    
    # Normalización para evitar escalar artificialmente la varianza del estímulo original
    return hrf / np.sum(hrf)


def apply_hrf_and_downsample(
    df_high_res: pd.DataFrame, 
    hrf_kernel: np.ndarray, 
    fs: int, 
    tr: float
) -> pd.DataFrame:
    """Aplica convolución hemodinámica por FFT y reduce la resolución al TR.
    
    Toma la matriz continua de características (ej. a 100 Hz), convoluciona cada 
    columna con la HRF en el dominio de la frecuencia para máxima eficiencia (FFT), 
    y finalmente aplica un filtro anti-aliasing reduciendo la señal a la escala 
    del escáner fMRI.
    
    Args:
        df_high_res (pd.DataFrame): Matriz original de características a alta resolución.
        hrf_kernel (np.ndarray): Filtro temporal de la función HRF.
        fs (int): Frecuencia de muestreo original de los datos (Hz).
        tr (float): Tiempo de repetición de la adquisición fMRI en segundos.
        
    Returns:
        pd.DataFrame: Matriz hemodinámica transformada y remuestreada, cuyo 
            índice de tiempo corresponde exactamente al TR del escáner.
    """
    n_samples, n_features = df_high_res.shape
    convolved_matrix = np.zeros((n_samples, n_features), dtype=np.float32)
    
    # Convolución independiente por característica usando Transformada Rápida de Fourier
    for col_idx in range(n_features):
        feature_signal = df_high_res.iloc[:, col_idx].values
        convolved_signal = fftconvolve(feature_signal, hrf_kernel, mode='full')
        convolved_matrix[:, col_idx] = convolved_signal[:n_samples]
        
    # Filtro anti-aliasing y reducción (Downsampling polifásico)
    down_factor = int(fs * tr)
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
    """Aplica transformación de Respuesta al Impulso Finito (FIR).
    
    Optimización: Reduce la resolución de la matriz predictora al TR antes 
    de crear los retardos (lags) para evitar explosiones de memoria RAM.
    
    Nota Metodológica: Si la matriz original tiene 45 características y se 
    solicitan 4 retardos, la matriz resultante tendrá 180 características.
    
    Args:
        df_high_res (pd.DataFrame): Matriz continua de características a alta resolución.
        fs (int): Frecuencia de muestreo original de los datos (Hz).
        tr (float): Tiempo de repetición del fMRI en segundos.
        delays_in_trs (List[int], opcional): Lista de retardos a generar (en cantidad de TRs).
            Por defecto [1, 2, 3, 4], cubriendo desde 2s hasta 8s (si TR=2.0s).
            
    Returns:
        pd.DataFrame: Matriz remuestreada y expandida con las columnas retardadas.
    """
    # 1. Reducción de resolución inmediata (Downsampling) para proteger memoria
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
    
    # 2. Generación de retardos (Lags)
    lagged_dataframes = []
    
    for delay in delays_in_trs:
        # shift() mueve las filas hacia abajo. fill_value=0.0 asegura 
        # que el padding no introduzca NaNs que rompan el Ridge Regression.
        df_shifted = df_downsampled.shift(delay, fill_value=0.0)
        
        # Renombrar columnas para mantener trazabilidad (ej. 'past_regular_lag2')
        df_shifted.columns = [f"{col}_lag{delay}" for col in df_downsampled.columns]
        lagged_dataframes.append(df_shifted)
        
    # 3. Concatenación horizontal masiva
    df_fir_space = pd.concat(lagged_dataframes, axis=1)
    df_fir_space.index.name = 'time_tr_seconds'
    
    return df_fir_space
