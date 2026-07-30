# scripts/06_generate_cortical_maps.py
"""Orquestador de Visualización Cortical 3D.

Genera mapas corticales interactivos utilizando PyCortex. Proyecta los 
resultados del modelo predictivo en la superficie del cerebro, creando 
dos capas (layers): una para la red global del lenguaje y otra específica 
para el tiempo gramatical.
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
    CORTICAL_COLORMAP, 
    DATA_DIR,
    DIR_PROCESSED,
    STATISTICAL_ALPHA,
)

RESULTS_IN_DIR = DIR_PROCESSED / "results_voxelwise"
WEB_EXPORT_DIR = DIR_PROCESSED / "cortical_maps_web"
PYCORTEX_DB_PATH = DATA_DIR / "derivatives" / "pycortex-db"


def configure_pycortex() -> None:
    """Configura de manera forzada y segura la ruta local de PyCortex.

    Sobrescribe la configuración nativa de PyCortex en ~/.config/pycortex
    para apuntar estrictamente a la base de datos derivada (pycortex-db) 
    dentro de la carpeta del proyecto. Previene errores de sujeto no encontrado.
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
    
    print(f"[INFO] PyCortex configurado con Filestore en: {PYCORTEX_DB_PATH}")


def generate_and_export_viewer(subject_id: str, start_server: bool = False) -> None:
    """Proyecta los datos de ML en la corteza y genera el visor web.

    Extrae R2 Global y Delta R2, aplica una máscara estricta binaria basada 
    en FDR y crea dos capas de visualización intercambiables en la interfaz.

    Args:
        subject_id (str): Identificador del participante (ej. 'sub-UTS01').
        start_server (bool, opcional): Si es True, despliega el servidor 
            interactivo local en el navegador. Por defecto es False.
    """
    results_path = RESULTS_IN_DIR / f"{subject_id}_voxelwise_results.parquet"

    if not results_path.exists():
        print(f"[OMITIDO] {subject_id}: No se encontraron resultados en {results_path}")
        return

    print(f"\n--- Modelando Superficie Cortical para {subject_id} ---")
    df_results = pd.read_parquet(results_path)

    # 1. Extracción de Vectores de Varianza
    r2_global = df_results["r2_global"].values.astype(float)
    delta_r2 = df_results["delta_r2_tense"].values.astype(float)
    p_values = df_results["p_value_fdr"].values

    # 2. Máscara de Transparencia Estricta (Binary Alpha Mask)
    # Se usa float(1.0) para vóxeles significativos y 0.0 para no significativos.
    # El color (intensidad) es independiente de esta máscara y se conservará intacto.
    is_significant = p_values < STATISTICAL_ALPHA
    alpha_mask = is_significant.astype(float)
    n_sig_voxels = np.sum(is_significant)

    print(f"[INFO] Vóxeles significativos a graficar: {n_sig_voxels:,} (FDR < {STATISTICAL_ALPHA})")

    try:
        # 3. Validación Anatómica
        cortex_subject = subject_id
        if cortex_subject not in cortex.db.subjects:
            cortex_subject = cortex_subject.replace("sub-", "")
            if cortex_subject not in cortex.db.subjects:
                raise ValueError(f"Sujeto '{subject_id}' no hallado en {PYCORTEX_DB_PATH}")

        transforms_dir = PYCORTEX_DB_PATH / cortex_subject / "transforms"
        if not transforms_dir.exists():
            raise FileNotFoundError(f"Carpeta de transformaciones ausente: {transforms_dir}")
            
        xfm_names = [
            item.stem if item.is_file() else item.name
            for item in transforms_dir.iterdir()
            if not item.name.startswith(".")
        ]
        
        if not xfm_names:
            raise FileNotFoundError("No se encontró ningún archivo de transformación anatómica.")
            
        xfm_name = xfm_names[0]

        # 4. Cálculo de Umbrales Visuales Dinámicos
        vmax_global = 0.01
        vmax_delta = 0.01
        
        if n_sig_voxels > 0:
            vmax_global = max(float(np.percentile(r2_global[is_significant], 99)), 0.01)
            vmax_delta = max(float(np.percentile(delta_r2[is_significant], 99)), 0.01)

        # 5. Construcción de Capas Volumétricas (Layers)
        vol_global = cortex.Volume(
            r2_global,
            subject=cortex_subject,
            xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP, 
            vmin=0.0,
            vmax=vmax_global,
            alpha=alpha_mask
        )

        vol_tense = cortex.Volume(
            delta_r2,
            subject=cortex_subject,
            xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP, 
            vmin=0.0,
            vmax=vmax_delta,
            alpha=alpha_mask
        )

        layer_dict = {
            "1. Global Language Network (R2)": vol_global,
            "2. Tense Specific Network (Delta R2)": vol_tense
        }

        # 6. Exportación y Visualización
        subject_export_dir = WEB_EXPORT_DIR / subject_id
        subject_export_dir.mkdir(parents=True, exist_ok=True)

        cortex.webgl.make_static(
            outpath=str(subject_export_dir),
            data=layer_dict,
            title=f"VEM Results - {subject_id}"
        )
        print(f"[EXITO] Mapa estático web exportado en: {subject_export_dir}")

        if start_server:
            print("\n" + "=" * 70)
            print("[SISTEMA] INICIANDO SERVIDOR WEB LOCAL")
            print("Verifique la URL generada debajo (ej. http://localhost:XXXXX).")
            print("=" * 70)
            
            _ = cortex.webshow(layer_dict, title=f"VEM Predictors - {subject_id}")
            input("\n[PAUSA] Presione ENTER en esta terminal para detener el servidor...")

    except Exception as e:
        print(f"[ERROR] Falló el mapeo cortical para {subject_id}:")
        print(traceback.format_exc())


def print_methodological_guide() -> None:
    """Imprime la interpretación metodológica de los mapas en consola."""
    print("\n" + "=" * 75)
    print("GUIA DE INTERPRETACION CORTICAL (Para redaccion del manuscrito)")
    print("=" * 75)
    print("\nAl abrir el visualizador de PyCortex, utilice el panel 'Data' (derecha)")
    print("para alternar entre las dos capas generadas:\n")
    
    print("Capa 1: Global Language Network (R2 Global)")
    print("  - Descripcion: Precision total del modelo usando las 45 caracteristicas.")
    print("  - Implicacion metodologica: Actua como un 'Sanity Check'. Debe revelar")
    print("    actividad robusta en regiones clasicas del lenguaje (STG, Broca, Wernicke).")
    print("    Confirma que la senal fMRI y el paradigma naturalista son viables.\n")
    
    print("Capa 2: Tense Specific Network (Delta R2)")
    print("  - Descripcion: Varianza predictiva unica del Tiempo Gramatical Finito.")
    print("  - Implicacion metodologica: Responde de forma directa a la interrogante:")
    print("    '¿Que areas del cerebro procesan el tiempo gramatical en general?'")
    print("    Aisla espacialmente el efecto morfosintactico tras controlar estadisticamente")
    print("    los dominios semantico, sintactico, lexico y fonologico.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    configure_pycortex()

    # Configuracion de ejecucion
    SUBJECT_INPUT: Union[str, List[str]] = "sub-UTS01"
    # SUBJECT_INPUT: Union[str, List[str]] = [
    #     "sub-UTS01", "sub-UTS02", "sub-UTS03"
    # ]

    if isinstance(SUBJECT_INPUT, str):
        generate_and_export_viewer(SUBJECT_INPUT, start_server=True)
    elif isinstance(SUBJECT_INPUT, list):
        print(f"[INFO] Modo Batch. Procesando {len(SUBJECT_INPUT)} sujetos de forma silenciosa...")
        for subject in SUBJECT_INPUT:
            generate_and_export_viewer(subject, start_server=False)
            
    print_methodological_guide()
