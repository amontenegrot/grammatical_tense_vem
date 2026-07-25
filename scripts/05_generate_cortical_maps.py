# scripts/05_generate_cortical_maps.py
#*
"""Orquestador de Visualización Cortical Interactiva.

Carga los resultados del modelo predictivo y genera mapas HTML enfocados
únicamente en el aporte del tiempo gramatical (Delta R2).
Proporciona un flujo dual: levanta un servidor en vivo para un solo sujeto
o procesa en lotes una lista de participantes de forma silenciosa.
"""

import configparser
import os
import traceback
from pathlib import Path
from typing import List, Union

import cortex
import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    DIR_PROCESSED,
    STATISTICAL_ALPHA,
)


RESULTS_IN_DIR = DIR_PROCESSED / "results_voxelwise"
WEB_EXPORT_DIR = DIR_PROCESSED / "cortical_maps_web"
PYCORTEX_DB_PATH = DATA_DIR / "derivatives" / "pycortex-db"


def configure_pycortex() -> None:
    """Configura la ruta de la base de datos anatómica local en PyCortex.

    Actualiza la configuración en tiempo de ejecución modificando el archivo
    'options.cfg' para asegurar que la librería encuentre los cerebros locales
    del proyecto en lugar del directorio por defecto del sistema operativo.
    """
    cortex_cfg_dir = Path(os.path.expanduser("~/.config/pycortex"))
    cortex_cfg_dir.mkdir(parents=True, exist_ok=True)
    cortex_cfg_file = cortex_cfg_dir / "options.cfg"

    config = configparser.ConfigParser()
    config.read(cortex_cfg_file)
    
    if not config.has_section("basic"):
        config.add_section("basic")

    config.set("basic", "filestore", str(PYCORTEX_DB_PATH))

    with open(cortex_cfg_file, "w", encoding="utf-8") as cfg_file:
        config.write(cfg_file)

    cortex.options.config.set("basic", "filestore", str(PYCORTEX_DB_PATH))
    cortex.db.filestore = str(PYCORTEX_DB_PATH)


def generate_and_export_viewer(subject_id: str, start_server: bool = False) -> None:
    """Genera el mapa cortical enfocado en Delta R2 y opcionalmente inicia el servidor web.

    Extrae las métricas predictivas, aplica una máscara de transparencia (alpha) a los
    vóxeles que no superan el umbral FDR y construye los objetos volumétricos 3D.

    Args:
        subject_id (str): Identificador del participante (ej. 'sub-UTS01').
        start_server (bool, optional): Si es True, abre la visualización interactiva
            en el navegador local pausando la ejecución de la terminal. Por defecto es False.

    Raises:
        ValueError: Si el sujeto no se encuentra en la base de datos de PyCortex.
    """
    results_path = RESULTS_IN_DIR / f"{subject_id}_voxelwise_results.parquet"

    if not results_path.exists():
        print(f"Omitiendo {subject_id}: No se encontraron resultados en {results_path}")
        return

    print(f"\nProcesando mapas corticales para {subject_id}...")
    df_results = pd.read_parquet(results_path)

    # ----------------------------------------------------------------------
    # 1. Extracción y Enmascarado Sencillo (Solo lo que importa)
    # ----------------------------------------------------------------------
    delta_r2 = df_results["delta_r2_tense"].values.astype(float)
    p_values = df_results["p_value_fdr"].values

    # Arreglo booleano de significancia estadística
    is_significant = p_values < STATISTICAL_ALPHA
    n_significant_voxels = np.sum(is_significant)

    # La máscara alpha le dice a PyCortex nativamente qué pintar (1.0) y qué ocultar (0.0)
    alpha_mask = is_significant.astype(float)

    print(f"  -> Vóxeles significativos (FDR < {STATISTICAL_ALPHA}): {n_significant_voxels} / {len(delta_r2)}")
    if n_significant_voxels == 0:
        print(f"  -> Advertencia: No hay vóxeles significativos para {subject_id}. El mapa estará vacío.")

    try:
        # Validación del sujeto en la base de datos anatómica
        cortex_subject = subject_id
        if cortex_subject not in cortex.db.subjects:
            cortex_subject = cortex_subject.replace("sub-", "")
            if cortex_subject not in cortex.db.subjects:
                raise ValueError("Sujeto no encontrado en DB.")

        transforms_dir = PYCORTEX_DB_PATH / cortex_subject / "transforms"
        xfm_names = [
            item.stem if item.is_file() else item.name
            for item in transforms_dir.iterdir()
            if not item.name.startswith(".")
        ]
        xfm_name = xfm_names[0]

        # ----------------------------------------------------------------------
        # 2. Creación de la Capa de Visualización (Simplificada)
        # ----------------------------------------------------------------------
        # Se calcula el valor máximo del colormap (vmax) ignorando el ruido de fondo
        if n_significant_voxels > 0:
            vmax_delta = max(float(np.percentile(delta_r2[is_significant], 99)), 0.01)
        else:
            vmax_delta = 0.01

        # Generación del volumen simple con el mapa de colores solicitado
        vol_delta = cortex.Volume(
            delta_r2,
            subject=cortex_subject,
            xfmname=xfm_name,
            cmap="OrRd",          # Colores naranjas/rojos (Unidireccional)
            vmin=0.0,
            vmax=vmax_delta,
            alpha=alpha_mask      # Transparencia gestionada internamente por PyCortex
        )

        layer_dict = {
            "Delta R2 (Tense)": vol_delta
        }

        # 3. Exportación HTML Estática
        subject_export_dir = WEB_EXPORT_DIR / subject_id
        subject_export_dir.mkdir(parents=True, exist_ok=True)

        cortex.webgl.make_static(
            outpath=str(subject_export_dir),
            data=layer_dict,
            title=f"VEM - {subject_id}"
        )
        print(f"  Mapa exportado con éxito en: {subject_export_dir}/index.html")

        # 4. Flujo de servidor en vivo (Opcional)
        if start_server:
            print("\n" + "=" * 65)
            print("INICIANDO SERVIDOR WEB EN VIVO")
            print("Revise la URL generada a continuación (ej. http://localhost:XXXXX).")
            print("=" * 65 + "\n")
            
            _ = cortex.webshow(layer_dict, title=f"VEM Predictive Variance - {subject_id}")
            input("Presione ENTER en esta terminal cuando termine para apagar el servidor y continuar...")

    except Exception:
        print(f"Error modelando a {subject_id}:")
        print(traceback.format_exc())


if __name__ == "__main__":
    configure_pycortex()

    # --------------------------------------------------------------------------
    # FLUJO DUAL (Interactivo o por Lotes)
    # --------------------------------------------------------------------------
    SUBJECT_INPUT: Union[str, List[str]] = "sub-UTS09"
    # SUBJECT_INPUT: Union[str, List[str]] = [
    #     "sub-UTS01", "sub-UTS02", "sub-UTS03", "sub-UTS04",
    #     "sub-UTS05", "sub-UTS06", "sub-UTS07", "sub-UTS08", "sub-UTS09"
    # ]

    if isinstance(SUBJECT_INPUT, str):
        print(f"Modo Interactivo activado para: {SUBJECT_INPUT}")
        generate_and_export_viewer(SUBJECT_INPUT, start_server=True)

    elif isinstance(SUBJECT_INPUT, list):
        print(f"Modo Batch activado. Generando mapas estáticos para {len(SUBJECT_INPUT)} sujetos...")
        for subject in SUBJECT_INPUT:
            generate_and_export_viewer(subject, start_server=False)

        print("\n" + "=" * 65)
        print("PROCESO POR LOTES COMPLETADO")
        print(f"Mapas exportados en: {WEB_EXPORT_DIR}")
        print("=" * 65 + "\n")
    else:
        print("Error: SUBJECT_INPUT debe ser un String o una Lista de Strings.")
    