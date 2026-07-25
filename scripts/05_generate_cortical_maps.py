# scripts/05_generate_cortical_maps.py
#*
"""Orquestador de Visualización Cortical Interactiva.

Carga resultados del modelo predictivo, aplica enmascarado estadístico estricto 
y genera mapas HTML multicapa. Proporciona un flujo dual: levanta un servidor 
en vivo para un solo sujeto o procesa en lotes una lista de participantes de 
forma silenciosa.
"""

import configparser
import os
from pathlib import Path
from typing import List, Union

import cortex
import numpy as np
import pandas as pd

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
    
    Actualiza la configuración en tiempo de ejecución modificando el archivo 
    'options.cfg' para asegurar que la librería encuentre los cerebros locales 
    del proyecto en lugar del directorio por defecto del sistema operativo.
    """
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


def generate_and_export_viewer(subject_id: str, start_server: bool = False) -> None:
    """
    Genera mapas multicapa enmascarados y opcionalmente inicia un servidor web.
    
    Extrae las métricas predictivas, aplica una máscara de transparencia a los 
    vóxeles que no superan el umbral FDR, y construye objetos volumétricos 3D.
    
    Args:
        subject_id (str): Identificador del participante (ej. 'sub-UTS01').
        start_server (bool, opcional): Si es True, abre la visualización interactiva 
            en el navegador local pausando la ejecución de la terminal. Por defecto 
            es False.
    """
    results_path = RESULTS_IN_DIR / f"{subject_id}_voxelwise_results.parquet"
    
    if not results_path.exists():
        print(f"Omitiendo {subject_id}: No se encontraron resultados en {results_path}")
        return
        
    print(f"\nProcesando mapas corticales para {subject_id}...")
    df_results = pd.read_parquet(results_path)
    
    # ----------------------------------------------------------------------
    # Enmascarado de Significancia Estadística Dinámico
    # ----------------------------------------------------------------------
    # Utiliza el mismo alfa configurado durante el cálculo de la prueba estadística
    non_significant_mask = df_results['p_value_fdr'].values >= STATISTICAL_ALPHA
    
    r2_global = df_results['r2_global'].values.astype(float)
    r2_restricted = df_results['r2_restricted'].values.astype(float)
    delta_r2 = df_results['delta_r2_tense'].values.astype(float)
    
    # Renderización transparente para los no significativos
    delta_r2[non_significant_mask] = np.nan
    r2_global[non_significant_mask] = np.nan
    r2_restricted[non_significant_mask] = np.nan
    
    n_significativos = np.sum(~non_significant_mask)
    print(f"  -> Vóxeles significativos (FDR < {STATISTICAL_ALPHA}): {n_significativos} / {len(delta_r2)}")
    
    if n_significativos == 0:
        print(f"  -> Advertencia: No hay vóxeles significativos para {subject_id}. Mapa transparente.")

    try:
        # Validación de sujeto en la base de PyCortex
        cortex_subject = subject_id
        if cortex_subject not in cortex.db.subjects:
            cortex_subject = cortex_subject.replace("sub-", "")
            if cortex_subject not in cortex.db.subjects:
                raise ValueError(f"Sujeto no encontrado en DB.")

        transforms_dir = PYCORTEX_DB_PATH / cortex_subject / "transforms"
        xfm_names = [item.stem if item.is_file() else item.name 
                     for item in transforms_dir.iterdir() if not item.name.startswith('.')]
        xfm_name = xfm_names[0]
        
        # ----------------------------------------------------------------------
        # Creación de Capas (Volumes)
        # ----------------------------------------------------------------------
        vmax_delta = max(float(np.nanpercentile(delta_r2, 99)), 0.01) if n_significativos > 0 else 0.01
        vmax_global = max(float(np.nanpercentile(r2_global, 99)), 0.01) if n_significativos > 0 else 0.01
        vmax_restr = max(float(np.nanpercentile(r2_restricted, 99)), 0.01) if n_significativos > 0 else 0.01

        vol_delta = cortex.Volume(
            delta_r2, subject=cortex_subject, xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP, vmin=0.0, vmax=vmax_delta
        )
        
        vol_global = cortex.Volume(
            r2_global, subject=cortex_subject, xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP, vmin=0.0, vmax=vmax_global
        )
        
        vol_restr = cortex.Volume(
            r2_restricted, subject=cortex_subject, xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP, vmin=0.0, vmax=vmax_restr
        )

        layer_dic = {
            "1. Delta R2 (Aporte Único Tense)": vol_delta,
            "2. R2 Global (Modelo Completo)": vol_global,
            "3. R2 Restringido (Sin Tense)": vol_restr
        }
        
        # Exportación HTML Estática
        subject_export_dir = WEB_EXPORT_DIR / subject_id
        subject_export_dir.mkdir(parents=True, exist_ok=True)
        
        cortex.webgl.make_static(
            outpath=str(subject_export_dir), 
            data=layer_dic, 
            title=f"VEM - {subject_id}"
        )
        print(f"  Mapa exportado con éxito en: {subject_export_dir}/index.html")

        # Flujo condicional de servidor en vivo
        if start_server:
            print("\n" + "="*65)
            print("INICIANDO SERVIDOR WEB EN VIVO")
            print("Revise la URL generada a continuación (ej. http://localhost:XXXXX).")
            print("="*65 + "\n")
            server = cortex.webshow(layer_dic, title=f"VEM Predictive Variance - {subject_id}")
            input("Presione ENTER en esta terminal cuando termine para apagar el servidor y continuar...")
        
    except Exception as e:
        import traceback
        print(f"Error modelando a {subject_id}:")
        print(traceback.format_exc())


if __name__ == "__main__":
    configure_pycortex()
    
    # --------------------------------------------------------------------------
    # FLUJO DUAL (Inteligencia de Entrada)
    # --------------------------------------------------------------------------
    # Si defines un String ("sub-UTS03"), procesará uno y abrirá el navegador.
    # Si defines una Lista (["sub-UTS03", "sub-UTS04"]), procesará en lote de forma silenciosa.
    
    SUBJECT_INPUT: Union[str, List[str]] = "sub-UTS09"
    # SUBJECT_INPUT: Union[str, List[str]] = [
    #     "sub-UTS01",
    #     "sub-UTS02",
    #     "sub-UTS03",
    #     "sub-UTS04",
    #     "sub-UTS05",
    #     "sub-UTS06",
    #     "sub-UTS07",
    #     "sub-UTS08",
    #     "sub-UTS09"
    # ]
    
    if isinstance(SUBJECT_INPUT, str):
        # Modo Interactivo: Un solo sujeto
        print(f"Modo Interactivo activado para: {SUBJECT_INPUT}")
        generate_and_export_viewer(SUBJECT_INPUT, start_server=True)
        
    elif isinstance(SUBJECT_INPUT, list):
        # Modo Lote (Batch): Múltiples sujetos silenciosamente
        print(f"Modo Batch activado. Generando mapas estáticos para {len(SUBJECT_INPUT)} sujetos...")
        for subj in SUBJECT_INPUT:
            generate_and_export_viewer(subj, start_server=False)
            
        print("\n" + "="*65)
        print("PROCESO POR LOTES COMPLETADO")
        print(f"Los mapas han sido exportados en formato estático a: {WEB_EXPORT_DIR}")
        print("Haga doble clic en el archivo 'index.html' del participante deseado.")
        print("="*65 + "\n")
    else:
        print("Error: SUBJECT_INPUT debe ser un String o una Lista de Strings.")
