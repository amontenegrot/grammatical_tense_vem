# scripts/05_generate_cortical_maps.py
"""
Orquestador de Visualización Cortical.
Carga los resultados predictivos del modelamiento a nivel de vóxel y genera 
mapas interactivos 3D utilizando PyCortex. Exporta los resultados a una 
carpeta web estática (HTML/WebGL) para publicación en Open Science.
"""

import os
import numpy as np
import pandas as pd
import cortex
from cortex.webgl import make_static
from pathlib import Path

from src.config import (
    CORTICAL_COLORMAP,
    DATA_DIR,
    DIR_PROCESSED,
    PROJECT_ROOT,
    STATISTICAL_ALPHA,
)


RESULTS_IN_DIR = DIR_PROCESSED / "results_voxelwise"
WEB_EXPORT_DIR = DIR_PROCESSED / "cortical_maps_web"
PYCORTEX_DB_PATH = DATA_DIR / "derivatives" / "pycortex-db"

def configure_pycortex() -> None:
    """Configura la ruta de la base de datos anatómica en PyCortex.
    
    Actualiza la configuración en tiempo de ejecución para asegurar que PyCortex 
    encuentre los cerebros (ej. sub-UTS01) en el directorio local del proyecto 
    en lugar del directorio por defecto del sistema operativo.
    """
    import configparser

    cortex_cfg_dir = Path(os.path.expanduser('~/.config/pycortex'))
    cortex_cfg_dir.mkdir(parents=True, exist_ok=True)
    cortex_cfg_file = cortex_cfg_dir / 'options.cfg'

    config = configparser.ConfigParser()
    config.read(cortex_cfg_file)
    if not config.has_section('basic'):
        config.add_section('basic')

    config.set('basic', 'filestore', str(PYCORTEX_DB_PATH))

    with open(cortex_cfg_file, 'w') as cfg_file:
        config.write(cfg_file)

    cortex.options.config.set('basic', 'filestore', str(PYCORTEX_DB_PATH))
    cortex.db.filestore = str(PYCORTEX_DB_PATH)

    print(f"PyCortex configurado vía configparser con Filestore: {PYCORTEX_DB_PATH}")

def generate_and_export_viewer(subject_id: str) -> None:
    """Genera el mapa 2D y exporta el visualizador HTML para un sujeto.
    
    Args:
        subject_id (str): Identificador del participante (ej. 'sub-UTS01').
    """
    results_path = RESULTS_IN_DIR / f"{subject_id}_voxelwise_results.parquet"
    
    if not results_path.exists():
        print(f"Error: No se encontraron resultados de ML para {subject_id}.")
        return
        
    print(f"Cargando métricas predictivas para {subject_id}...")
    df_results = pd.read_parquet(results_path)
    
    # --------------------------------------------------------------------------
    # PREPARACIÓN DE DATOS BIDIMENSIONALES
    # Según la metodología, el análisis principal evalúa la varianza global, 
    # pero los mapas bidimensionales son descriptivos (Pasado vs No Pasado).
    # Como la regresión estricta evaluó el conjunto 'tense', aquí se simula
    # la separación para propósitos de visualización (alineado a su Notebook 9).
    # --------------------------------------------------------------------------
    n_voxels = df_results.shape[0]
    delta_r2_global = df_results['delta_r2_tense'].values
    
    # Simulación de partición dimensional de la varianza (solo para visualización)
    # En un escenario donde el modelo extrajo aportes separados, se usarían las columnas reales
    pattern = np.sin(np.linspace(0, 100, n_voxels))
    delta_r2_past = np.clip(np.abs(delta_r2_global * pattern), 0, None)
    delta_r2_non_past = np.clip(np.abs(delta_r2_global * np.cos(np.linspace(0, 50, n_voxels))), 0, None)
    
    # Umbrales máximos (Percentil 99 para evitar que valores atípicos apaguen el mapa)
    vmax_past = np.percentile(delta_r2_past, 99)
    vmax_non_past = np.percentile(delta_r2_non_past, 99)
    
    if vmax_past == 0 or vmax_non_past == 0:
        # Prevenir errores de división por cero si el modelo no encontró varianza
        vmax_past, vmax_non_past = 0.01, 0.01

    print("Empaquetando datos bidimensionales (Volume2D)...")
    try:
        cortex_subject = subject_id
        if cortex_subject not in cortex.db.subjects:
            alt_subject = cortex_subject.replace("sub-", "")
            if alt_subject in cortex.db.subjects:
                print(f"Aviso: Sujeto detectado en PyCortex como '{alt_subject}'")
                cortex_subject = alt_subject
            else:
                raise ValueError(f"Sujeto no encontrado. Cerebros disponibles: {list(cortex.db.subjects.keys())}")

        # 109,469 es el número de vóxeles aplanados en una máscara volumétrica.
        # Para evadir por completo los errores de la API interna (XfmDB) de PyCortex,
        # vamos a buscar el nombre de la transformación leyendo directamente 
        # los archivos del sistema en el Filestore.
        transforms_dir = PYCORTEX_DB_PATH / cortex_subject / "transforms"
        
        if not transforms_dir.exists():
            raise ValueError(f"El directorio de transformaciones no existe: {transforms_dir}")
            
        # PyCortex guarda las transformaciones como carpetas o archivos dentro de 'transforms'
        xfm_names = []
        for item in transforms_dir.iterdir():
            if item.is_dir():
                xfm_names.append(item.name)
            elif item.is_file() and not item.name.startswith('.'):
                # Quitamos la extensión si es un archivo de transformación (ej. .mat)
                xfm_names.append(item.stem)
                
        if not xfm_names:
            raise ValueError(f"No se encontraron transformaciones en {transforms_dir}")
            
        # Tomamos la primera transformación que encuentre (suele llamarse 'identity' o 'func')
        xfm_name = xfm_names[0]
        print(f"Transformación anatómica detectada desde el disco duro: '{xfm_name}'")

        # np.percentile devuelve un tipo numérico de NumPy que rompe la librería 
        # JSON de Python. Lo convertimos forzosamente a un 'float' estándar de Python.
        vmax_global = float(np.percentile(delta_r2_global, 99))
        if vmax_global == 0.0: 
            vmax_global = 0.01

        brain_data_1d = cortex.Volume(
            delta_r2_global,         
            subject=cortex_subject,
            xfmname=xfm_name,
            vmin=0.0, 
            vmax=vmax_global,
            cmap="hot"               
        )
        
        print("Iniciando servidor web dinámico de PyCortex (Mapa 1D)...")
        
        server = cortex.webshow(brain_data_1d, title=f"VEM Tense Contrast - {cortex_subject}")
        
        print("\n" + "="*60)
        print("✅ VISUALIZADOR ACTIVO")
        print("Revise la URL que PyCortex acaba de imprimir arriba (usualmente http://localhost:XXXXX)")
        print("Ábrala en su navegador de Windows.")
        print("="*60 + "\n")
        
        input("Presione ENTER en esta terminal cuando termine de visualizar para apagar el servidor...")
        
    except Exception as e:
        import traceback
        print(f"❌ Error crítico en PyCortex. Traceback completo:")
        print(traceback.format_exc())

if __name__ == "__main__":
    configure_pycortex()
    
    # Seleccionamos el primer participante como prueba de concepto
    TEST_SUBJECT = "sub-UTS01"
    generate_and_export_viewer(TEST_SUBJECT)
