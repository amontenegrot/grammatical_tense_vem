# scripts/06_generate_cortical_maps.py
"""Orquestador de Visualización Cortical 3D en PyCortex.

Genera proyecciones corticales individuales en la anatomía del participante,
organizadas en tres capas complementarias:
1. Capa 1: Desempeño del Modelo Global (R2 > 0.01) en escala cálida secuencial ('OrRd').
2. Capa 2a: Aporte del Tiempo Gramatical Umbralizado (FDR < 0.05) en escala secuencial ('OrRd').
3. Capa 2b: Aporte del Tiempo Gramatical Descriptivo Continuo (sin umbral) en escala divergente ('coolwarm').
Exporta visores web estáticos desacoplados de la evaluación numérica principal.
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
    CORTICAL_COLORMAP_DIVERGING,
    CORTICAL_COLORMAP_GLOBAL,
    CORTICAL_COLORMAP_TENSE_SIG,
    DATA_DIR,
    DIR_PROCESSED,
    STATISTICAL_ALPHA,
)


RESULTS_IN_DIR = DIR_PROCESSED / "results_voxelwise"
WEB_EXPORT_DIR = DIR_PROCESSED / "cortical_maps_web"
PYCORTEX_DB_PATH = DATA_DIR / "derivatives" / "pycortex-db"


def configure_pycortex() -> None:
    """Configura de manera forzada y segura el filestore local de PyCortex."""
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
    """Proyecta los resultados del participante en la superficie cortical."""
    results_path = RESULTS_IN_DIR / f"{subject_id}_voxelwise_results.parquet"

    if not results_path.exists():
        print(f"[OMITIDO] {subject_id}: No se encontraron resultados en {results_path}")
        return

    print(f"\n--- Modelando Superficie Cortical para {subject_id} ---")
    df_results = pd.read_parquet(results_path)

    r2_global = df_results["r2_global"].values.astype(float)
    delta_r2 = df_results["delta_r2_tense"].values.astype(float)
    p_values = df_results["p_value_fdr"].values

    is_significant = p_values < STATISTICAL_ALPHA
    n_sig_voxels = np.sum(is_significant)
    print(f"[INFO] Vóxeles significativos (FDR < {STATISTICAL_ALPHA}): {n_sig_voxels:,} de {len(df_results):,}")

    # 1. Enmascarado de Datos (Reemplazando por NaN para transparencia nativa en WebGL)
    # Capa 1: Sanity Check (Modelo Global R2 > 0.01)
    r2_global_masked = np.where(r2_global > 0.01, r2_global, np.nan)
    
    # Capa 2a: Tiempo Gramatical Umbralizado por FDR
    delta_r2_sig_masked = np.where(is_significant, delta_r2, np.nan)
    
    # Capa 2b: Tiempo Gramatical Continuo Descriptivo (sin filtro estadístico)
    delta_r2_unthresholded = delta_r2.copy()

    try:
        # 2. Validación Anatómica
        cortex_subject = subject_id
        if cortex_subject not in cortex.db.subjects:
            cortex_subject = cortex_subject.replace("sub-", "")
            if cortex_subject not in cortex.db.subjects:
                raise ValueError(f"Sujeto '{subject_id}' no hallado en el filestore {PYCORTEX_DB_PATH}")

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

        # 3. Límites Dinámicos de Escala
        vmax_global = 0.05
        valid_global = r2_global_masked[~np.isnan(r2_global_masked)]
        if len(valid_global) > 0:
            vmax_global = max(float(np.percentile(valid_global, 99)), 0.05)

        # Umbral para Capa 2a (Sig)
        vmax_sig = 0.01
        valid_sig = delta_r2_sig_masked[~np.isnan(delta_r2_sig_masked)]
        if len(valid_sig) > 0:
            vmax_sig = max(float(np.percentile(valid_sig, 99)), 0.01)

        # Umbral simétrico para Capa 2b (Divergente centrado en 0)
        vmax_div = float(max(np.percentile(np.abs(delta_r2_unthresholded), 99), 0.005))

        # 4. Construcción de Capas Volumétricas
        # Capa 1: Modelo Global (OrRd)
        vol_global = cortex.Volume(
            r2_global_masked,
            subject=cortex_subject,
            xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP_GLOBAL, 
            vmin=0.01,
            vmax=vmax_global
        )

        # Capa 2a: Tiempo Gramatical Umbralizado FDR (OrRd)
        vol_tense_sig = cortex.Volume(
            delta_r2_sig_masked,
            subject=cortex_subject,
            xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP_TENSE_SIG, 
            vmin=0.0001,
            vmax=vmax_sig
        )

        # Capa 2b: Tiempo Gramatical Descriptivo Continuo (coolwarm simétrico)
        vol_tense_div = cortex.Volume(
            delta_r2_unthresholded,
            subject=cortex_subject,
            xfmname=xfm_name,
            cmap=CORTICAL_COLORMAP_DIVERGING,
            vmin=-vmax_div,
            vmax=vmax_div
        )

        layer_dict = {
            "1. Modelo Global (R2 > 0.01)": vol_global,
            "2a. Tiempo Gramatical (FDR < 0.05)": vol_tense_sig,
            "2b. Tiempo Gramatical (Continuo Descriptivo)": vol_tense_div
        }

        # 5. Exportación a Visor Web Estático
        subject_export_dir = WEB_EXPORT_DIR / subject_id
        subject_export_dir.mkdir(parents=True, exist_ok=True)

        cortex.webgl.make_static(
            outpath=str(subject_export_dir),
            data=layer_dict,
            title=f"VEM Resultados Corticales - {subject_id}"
        )
        print(f"[ÉXITO] Visor estático web exportado en: {subject_export_dir}")

        if start_server:
            print("\n" + "=" * 70)
            print(f"[SISTEMA] INICIANDO SERVIDOR WEB LOCAL PARA {subject_id}")
            print("=" * 70)
            _ = cortex.webshow(layer_dict, title=f"VEM - {subject_id}")
            input("\n[PAUSA] Presione ENTER en esta terminal para detener el servidor...")

    except Exception:
        print(f"[ERROR] Falló el mapeo cortical para {subject_id}:")
        print(traceback.format_exc())


if __name__ == "__main__":
    configure_pycortex()
    
    # Configuración de ejecución (Participante individual o lista)
    SUBJECT_TO_PLOT: str = "sub-UTS01"
    generate_and_export_viewer(SUBJECT_TO_PLOT, start_server=False)
