# src/data_loader.py
"""Módulo de carga de datos fMRI.

Maneja la lectura eficiente de archivos espaciales HDF5, asegurando un bajo 
consumo de memoria RAM mediante conversiones tempranas y explícitas.
"""

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from src.config import HDF5_DATASET_KEY


def load_fmri_data(hf5_path: str, dataset_key: str = HDF5_DATASET_KEY) -> Optional[np.ndarray]:
    """Carga la matriz de series de tiempo BOLD desde un archivo HDF5.
    
    Abre el archivo en modo de solo lectura y extrae la matriz de vóxeles.
    Asume que el archivo ya fue preprocesado y convertido a una matriz 2D
    (tiempo x vóxeles).
    
    Args:
        hf5_path (str): Ruta absoluta al archivo .hf5.
        dataset_key (str, opcional): Clave del dataset dentro del HDF5. 
            Por defecto utiliza la llave configurada en src.config.
            
    Returns:
        Optional[np.ndarray]: Matriz BOLD en formato numpy (float32) o None 
            si ocurre un error de lectura o el archivo no existe.
    """
    path_obj = Path(hf5_path)
    if not path_obj.exists():
        return None
        
    try:
        with h5py.File(path_obj, 'r') as f:
            # Se extrae como float32 para optimizar el consumo masivo de memoria 
            # antes de inyectarlo en los pipelines de regresión.
            fmri_matrix = np.array(f[dataset_key], dtype=np.float32)
            return fmri_matrix
            
    except Exception as e:
        print(f"Error cargando HDF5 en {hf5_path}: {e}")
        return None
