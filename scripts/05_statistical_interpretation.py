# scripts/05_statistical_interpretation.py
"""Orquestador de Interpretación Estadística.

Lee los resultados del modelo Ridge Voxelwise (Nivel de Sujeto), calcula 
métricas globales de varianza particionada (Delta R2) y tamaño del efecto, 
y genera automáticamente un reporte en formato Markdown listo para ser 
incluido en un manuscrito científico (LaTeX/Word).
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import (
    DIR_PROCESSED,
    PROJECT_ROOT,
    STATISTICAL_ALPHA
)
import time
from src.db_manager import log_execution_time

RESULTS_IN_DIR = DIR_PROCESSED / "results_voxelwise"
REPORTS_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "reports" / "statistical"

# =============================================================================
# UMBRALES BASADOS EN LITERATURA FMRI NATURALISTA
# =============================================================================
# Puedes ajustar estos valores si los revisores te piden criterios más estrictos.

# Umbrales para Varianza Única Explicada (Delta R2 medio en vóxeles significativos)
# 0.01 significa que la característica explica un 1% extra de varianza sobre el ruido base.
EFFECT_SIZE_MODERATE = 0.01  # Típico en fMRI naturalista
EFFECT_SIZE_STRONG = 0.05    # Alto/Fuerte para un solo rasgo lingüístico

# Umbrales para Extensión Espacial (% de la corteza escaneada)
SPATIAL_FOCAL = 1.0     # Efecto altamente localizado (ej. solo giro temporal superior)
SPATIAL_EXTENDED = 5.0  # Efecto distribuido moderado
SPATIAL_MASSIVE = 15.0  # Efecto masivo (Poco común para rasgos gramaticales específicos)


def evaluate_effect_size(mean_delta_r2: float) -> str:
    """Evalúa el tamaño del efecto (Delta R2) basándose en umbrales estándar.
    
    Args:
        mean_delta_r2 (float): Promedio de Delta R2 en vóxeles significativos.
        
    Returns:
        str: Párrafo interpretativo del tamaño del efecto.
    """
    if mean_delta_r2 == 0.0:
        return "Nulo. No se detectó varianza única explicada."
    elif mean_delta_r2 < EFFECT_SIZE_MODERATE:
        return (
            "Débil / Marginal. Es un efecto sutil, completamente esperable en "
            "paradigmas de fMRI naturalista (historias), pero se encuentra en el límite "
            "inferior de detectabilidad."
        )
    elif mean_delta_r2 < EFFECT_SIZE_STRONG:
        return (
            "Moderado / Típico. El tamaño del efecto es robusto y se alinea perfectamente "
            "con los estándares de la literatura de Voxelwise Encoding Models."
        )
    else:
        return (
            "Fuerte. Un efecto notablemente alto para una sola característica lingüística. "
            "Sugiere una fuerte sintonización cortical hacia el tiempo gramatical."
        )


def evaluate_spatial_extent(perc_sig: float) -> str:
    """Evalúa la extensión espacial del efecto en la corteza total.
    
    Args:
        perc_sig (float): Porcentaje de vóxeles significativos.
        
    Returns:
        str: Párrafo interpretativo de la extensión cortical.
    """
    if perc_sig == 0.0:
        return "Ausente. El modelo no logró predecir la actividad cortical por encima del azar."
    elif perc_sig < SPATIAL_FOCAL:
        return (
            "Altamente Focalizado. El procesamiento del tiempo gramatical está restringido "
            "a poblaciones neuronales muy específicas (sugiere especialización regional estricta)."
        )
    elif perc_sig < SPATIAL_EXTENDED:
        return (
            "Focalizado a Moderado. Sugiere que el procesamiento ocurre en regiones locales "
            "clásicas del lenguaje sin reclutar redes de dominio general."
        )
    elif perc_sig < SPATIAL_MASSIVE:
        return (
            "Extendido / Distribuido. El efecto abarca múltiples áreas corticales, sugiriendo "
            "una red distribuida para el procesamiento morfosintáctico continuo."
        )
    else:
        return (
            "Masivo / Global. Altamente inusual para una característica tan específica. "
            "Recomendación metodológica: Revisar si existe colinealidad con características "
            "acústicas (como la intensidad del audio) o semánticas generales."
        )


def generate_subject_report(df: pd.DataFrame, subject_id: str) -> Tuple[str, Dict]:
    """Calcula estadísticas para un sujeto y redacta su interpretación.
    
    Args:
        df (pd.DataFrame): DataFrame con resultados de Ridge.
        subject_id (str): Identificador del sujeto.
        
    Returns:
        Tuple[str, Dict]: 
            - Texto en formato Markdown con el reporte del sujeto.
            - Diccionario con los estadísticos calculados para el reporte grupal.
    """
    total_voxels = len(df)
    
    # Filtro de falsos positivos (False Discovery Rate)
    sig_mask = df['p_value_fdr'] < STATISTICAL_ALPHA
    sig_voxels = int(sig_mask.sum())
    perc_sig = (sig_voxels / total_voxels) * 100
    
    # Cálculos de varianza (Solo en vóxeles donde hubo significancia)
    if sig_voxels > 0:
        mean_delta_r2 = float(df.loc[sig_mask, 'delta_r2_tense'].mean())
        max_delta_r2 = float(df.loc[sig_mask, 'delta_r2_tense'].max())
    else:
        mean_delta_r2 = 0.0
        max_delta_r2 = 0.0

    # Generación de interpretaciones condicionales
    spatial_interp = evaluate_spatial_extent(perc_sig)
    effect_interp = evaluate_effect_size(mean_delta_r2)

    # Redacción del manuscrito (f-string)
    md_text = f"### Participant: {subject_id}\n\n"
    md_text += f"- **Corteza Analizada:** {total_voxels:,} vóxeles totales.\n"
    md_text += f"- **Vóxeles Significativos (FDR < {STATISTICAL_ALPHA}):** {sig_voxels:,} vóxeles.\n"
    md_text += f"- **Extensión Espacial:** {perc_sig:.2f}%. *Interpretación:* {spatial_interp}\n"
    md_text += f"- **Varianza Única Media ($\\Delta R^2$):** {mean_delta_r2:.4f}. *Interpretación:* {effect_interp}\n"
    md_text += f"- **Pico Máximo de Varianza (Max $\\Delta R^2$):** {max_delta_r2:.4f} (Este es el vóxel que mejor codifica el tiempo gramatical en este sujeto).\n\n"

    # Diccionario para retorno
    stats = {
        'subject_id': subject_id,
        'total_voxels': total_voxels,
        'sig_voxels': sig_voxels,
        'perc_sig': perc_sig,
        'mean_delta_r2': mean_delta_r2
    }
    
    return md_text, stats


def generate_group_report(all_stats: List[Dict]) -> str:
    """Calcula el consenso muestral a partir de las estadísticas individuales.
    
    Nota: Se aclara explícitamente en el reporte que es descriptivo debido al n pequeño.
    
    Args:
        all_stats (List[Dict]): Lista de diccionarios con estadísticas por sujeto.
        
    Returns:
        str: Texto en formato Markdown con el consenso grupal.
    """
    n_subjects = len(all_stats)
    
    # Contamos cuántos sujetos mostraron al menos 0.1% de la corteza activa 
    # (Filtro para considerar que el efecto existe y no es un artefacto de 1 o 2 vóxeles perdidos)
    subjects_with_effects = sum(1 for s in all_stats if s['perc_sig'] >= 0.1)
    
    avg_perc_sig = sum(s['perc_sig'] for s in all_stats) / n_subjects
    
    # Promedio del tamaño de efecto solo tomando sujetos donde sí hubo efecto
    valid_r2s = [s['mean_delta_r2'] for s in all_stats if s['mean_delta_r2'] > 0]
    avg_mean_delta_r2 = sum(valid_r2s) / len(valid_r2s) if valid_r2s else 0.0

    md_text = "## Sample-Level Consensus (Descriptivo)\n\n"
    md_text += f"> **Nota Metodológica sobre Tamaño Muestral:** Debido a la naturaleza intensiva de la recolección de datos fMRI naturalistas (múltiples horas de escaneo por participante), el tamaño de la muestra (*n*={n_subjects}) es estadísticamente modesto. Por lo tanto, el siguiente reporte grupal representa una caracterización muestral cualitativa de consistencia, y no una inferencia poblacional estricta (Random Effects).\n\n"
    
    md_text += f"- **Consistencia de Detección:** Se encontraron representaciones significativas del tiempo gramatical en **{subjects_with_effects} de {n_subjects}** participantes estudiados.\n"
    md_text += f"- **Extensión Promedio:** A través de la cohorte, el rasgo activa en promedio un **{avg_perc_sig:.2f}%** de la corteza medida.\n"
    md_text += f"- **Tamaño de Efecto Promedio:** La varianza única media extraída globalmente es de **$\\Delta R^2$ = {avg_mean_delta_r2:.4f}**.\n\n"
    
    return md_text


def run_statistical_interpretation() -> None:
    """Orquesta la lectura, análisis y exportación del reporte estadístico."""
    REPORTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_files = list(RESULTS_IN_DIR.glob("*_voxelwise_results.parquet"))
    
    if not parquet_files:
        print(f"Error: No se encontraron resultados parquet en {RESULTS_IN_DIR}.")
        return
        
    print(f"Iniciando interpretación estadística para {len(parquet_files)} sujetos...")
    
    manuscript_content = "# Statistical Interpretation Report\n"
    manuscript_content += "Generado automáticamente por el Pipeline de Voxelwise Encoding.\n\n"
    manuscript_content += "## Single-Subject Level Analysis\n\n"
    
    all_subject_stats = []
    
    # Procesamiento por sujeto
    for file_path in sorted(parquet_files):
        # Extraer el sub-id (ej. de 'sub-UTS01_voxelwise_results.parquet')
        subject_id = file_path.name.split('_voxelwise')[0]
        
        try:
            df = pd.read_parquet(file_path)
            md_text, stats = generate_subject_report(df, subject_id)
            
            manuscript_content += md_text
            all_subject_stats.append(stats)
            
            # También lo imprimimos en consola para retroalimentación inmediata
            print(f"\n--- {subject_id} ---")
            print(f"Vóxeles Sig: {stats['sig_voxels']}/{stats['total_voxels']} ({stats['perc_sig']:.2f}%)")
            print(f"Delta R2 Medio: {stats['mean_delta_r2']:.4f}")
            
        except Exception as e:
            print(f"Error procesando estadísticas para {subject_id}: {str(e)}")

    # Procesamiento Grupal
    if all_subject_stats:
        group_md_text = generate_group_report(all_subject_stats)
        manuscript_content += group_md_text
        
        # Guardado en disco
        report_file = REPORTS_OUT_DIR / "statistical_manuscript_draft.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(manuscript_content)
            
        print(f"\n[ÉXITO] Reporte Markdown exportado en: {report_file}")
        print("Puedes abrir este archivo en VSCode/Typora y copiar el contenido a LaTeX/Word.")


if __name__ == "__main__":
    start_time = time.time()
    run_statistical_interpretation()
    log_execution_time("05_statistical_interpretation", time.time() - start_time)
