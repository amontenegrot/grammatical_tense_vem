# scripts/05_statistical_interpretation.py
"""Orquestador de Interpretación Estadística.

Lee los resultados de la regresión Ridge a nivel de vóxel por participante,
calcula la distribución completa del aporte predictivo (Delta R2), caracteriza
los vóxeles que superan la corrección FDR (reportando 'N/A' ante la ausencia de efecto)
y genera un reporte en formato Markdown estructurado según los 3 escenarios del anteproyecto.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import (
    DIR_PROCESSED,
    PROJECT_ROOT,
    STATISTICAL_ALPHA,
    TEST_STORY,
)
from src.db_manager import log_execution_time, save_dataframe_to_table


RESULTS_IN_DIR = DIR_PROCESSED / "results_voxelwise"
REPORTS_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "reports" / "statistical"


def calculate_voxelwise_metrics(df: pd.DataFrame) -> Dict[str, Union[int, float, str]]:
    """Calcula las métricas estadísticas a los 3 niveles requeridos por el anteproyecto.
    
    1. Desempeño global (R2_global).
    2. Distribución continua de Delta R2 en la totalidad de la corteza.
    3. Métricas en vóxeles significativos tras FDR (o 'N/A' si no hay ninguno).
    """
    total_voxels = len(df)
    r2_global = df['r2_global'].values
    delta_r2 = df['delta_r2_tense'].values
    p_fdr = df['p_value_fdr'].values
    
    # 1. Nivel Global (R2)
    r2_global_positive_pct = float(np.mean(r2_global > 0) * 100)
    r2_global_median = float(np.median(r2_global))
    r2_global_max = float(np.max(r2_global))
    
    # 2. Nivel Distribución Completa Delta R2 (Todos los vóxeles)
    delta_median = float(np.median(delta_r2))
    q25, q75 = np.percentile(delta_r2, [25, 75])
    delta_iqr = float(q75 - q25)
    delta_p5 = float(np.percentile(delta_r2, 5))
    delta_p95 = float(np.percentile(delta_r2, 95))
    delta_min = float(np.min(delta_r2))
    delta_max = float(np.max(delta_r2))
    delta_pos_pct = float(np.mean(delta_r2 > 0) * 100)
    
    # 3. Nivel Vóxeles Significativos (FDR < 0.05)
    sig_mask = p_fdr < STATISTICAL_ALPHA
    sig_voxels = int(np.sum(sig_mask))
    sig_pct = float((sig_voxels / total_voxels) * 100)
    
    # Decisión metodológica: Si no hay vóxeles significativos, reportar como 'N/A'
    # para no confundir la falta de potencia estadística con un Delta R2 de 0.0 exacto.
    if sig_voxels > 0:
        sig_delta_mean = f"{float(np.mean(delta_r2[sig_mask])):.5f}"
        sig_delta_median = f"{float(np.median(delta_r2[sig_mask])):.5f}"
        sig_delta_max = f"{float(np.max(delta_r2[sig_mask])):.5f}"
    else:
        sig_delta_mean = "N/A"
        sig_delta_median = "N/A"
        sig_delta_max = "N/A"
        
    return {
        'total_voxels': total_voxels,
        'r2_global_median': r2_global_median,
        'r2_global_max': r2_global_max,
        'r2_global_positive_pct': r2_global_positive_pct,
        'delta_median': delta_median,
        'delta_iqr': delta_iqr,
        'delta_p5': delta_p5,
        'delta_p95': delta_p95,
        'delta_min': delta_min,
        'delta_max': delta_max,
        'delta_positive_pct': delta_pos_pct,
        'sig_voxels_count': sig_voxels,
        'sig_voxels_pct': sig_pct,
        'sig_delta_mean': sig_delta_mean,
        'sig_delta_median': sig_delta_median,
        'sig_delta_max': sig_delta_max
    }


def generate_subject_markdown(metrics: Dict, subject_id: str) -> str:
    """Genera la sección individual del manuscrito en formato Markdown."""
    sig_count = metrics['sig_voxels_count']
    sig_pct = metrics['sig_voxels_pct']
    
    md = f"### Participante: {subject_id}\n\n"
    md += f"- **Volumen Cortical Analizado:** {metrics['total_voxels']:,} vóxeles.\n"
    md += (
        f"- **Desempeño del Modelo Global (43 características):** "
        f"Mediana $R^2 = {metrics['r2_global_median']:.4f}$, "
        f"Máximo $R^2 = {metrics['r2_global_max']:.4f}$ "
        f"({metrics['r2_global_positive_pct']:.2f}% de vóxeles con ajuste positivo).\n"
    )
    md += (
        f"- **Distribución de $\\Delta R^2$ en Toda la Corteza:** "
        f"Mediana = {metrics['delta_median']:.5f}, "
        f"IQR = {metrics['delta_iqr']:.5f}, "
        f"Percentil 5 = {metrics['delta_p5']:.5f}, "
        f"Percentil 95 = {metrics['delta_p95']:.5f}, "
        f"Rango = [{metrics['delta_min']:.5f}, {metrics['delta_max']:.5f}], "
        f"Vóxeles con $\\Delta R^2 > 0$ = {metrics['delta_positive_pct']:.2f}%.\n"
    )
    md += (
        f"- **Vóxeles Significativos (FDR Benjamini-Hochberg $q < {STATISTICAL_ALPHA}$):** "
        f"{sig_count:,} vóxeles ({sig_pct:.2f}% de la corteza).\n"
    )
    
    if sig_count > 0:
        md += (
            f"  - *Aporte en vóxeles significativos:* Media $\\Delta R^2 = {metrics['sig_delta_mean']}$, "
            f"Mediana $\\Delta R^2 = {metrics['sig_delta_median']}$, "
            f"Pico Máximo $\\Delta R^2 = {metrics['sig_delta_max']}$.\n"
        )
    else:
        md += (
            f"  - *Aporte en vóxeles significativos:* **N/A** (Ningún vóxel superó el criterio corregido). "
            f"Bajo la configuración implementada, no se identificó un aporte predictivo separable "
            f"de los 5 espacios de control para la historia '{TEST_STORY}'.\n"
        )
        
    md += "\n"
    return md


def run_statistical_interpretation() -> None:
    """Orquesta el análisis cuantitativo y la exportación de reportes."""
    REPORTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted(list(RESULTS_IN_DIR.glob("*_voxelwise_results.parquet")))
    if not parquet_files:
        print(f"Error: No se encontraron resultados parquet en {RESULTS_IN_DIR}.")
        return
        
    print(f"Iniciando interpretación estadística formal para {len(parquet_files)} participantes...")
    
    summary_records = []
    markdown_content = "# Reporte de Interpretación Estadística del Modelo VEM\n\n"
    markdown_content += (
        "**Estudio:** Contribución predictiva del tiempo gramatical finito a la señal BOLD "
        "durante la comprensión de habla naturalista continua.\n\n"
    )
    markdown_content += "## 1. Análisis a Nivel de Participante Individual\n\n"
    
    for file_path in parquet_files:
        subject_id = file_path.name.split('_voxelwise')[0]
        try:
            df = pd.read_parquet(file_path)
            metrics = calculate_voxelwise_metrics(df)
            metrics['subject_id'] = subject_id
            summary_records.append(metrics)
            
            md_subject = generate_subject_markdown(metrics, subject_id)
            markdown_content += md_subject
            
            print(f"[{subject_id}] FDR Sig: {metrics['sig_voxels_count']} ({metrics['sig_voxels_pct']:.2f}%) | "
                  f"Delta R2 Mediana: {metrics['delta_median']:.5f} | "
                  f"Sig Delta Media: {metrics['sig_delta_mean']}")
                  
        except Exception as e:
            print(f"[ERROR] Procesando participante {subject_id}: {str(e)}")

    # 2. Resumen Muestral Descriptivo (Consistencia entre participantes)
    df_summary = pd.DataFrame(summary_records)
    
    n_subj = len(df_summary)
    subj_with_effects = sum(1 for r in summary_records if r['sig_voxels_count'] > 0)
    
    markdown_content += "## 2. Caracterización Descriptiva de la Muestra\n\n"
    markdown_content += (
        "> **Nota Metodológica:** Los resultados grupales constituyen una síntesis descriptiva "
        "de consistencia en la muestra analizada y no una inferencia poblacional de efectos aleatorios.\n\n"
    )
    markdown_content += (
        f"- **Consistencia Muestral:** {subj_with_effects} de {n_subj} participantes presentaron "
        f"vóxeles donde el tiempo gramatical superó la corrección FDR ($q < {STATISTICAL_ALPHA}$).\n"
    )
    markdown_content += (
        f"- **Desempeño Promedio del Modelo Global:** Mediana $R^2 = {df_summary['r2_global_median'].mean():.4f}$, "
        f"con un promedio de {df_summary['r2_global_positive_pct'].mean():.2f}% de vóxeles con ajuste positivo.\n"
    )
    markdown_content += (
        f"- **Distribución de $\\Delta R^2$ Muestral:** Mediana transversal = {df_summary['delta_median'].mean():.5f}, "
        f"Proporción de vóxeles con $\\Delta R^2 > 0$ = {df_summary['delta_positive_pct'].mean():.2f}%.\n"
    )
    
    # 3. Exportación de artefactos
    report_md_file = REPORTS_OUT_DIR / "statistical_interpretation_report.md"
    report_csv_file = REPORTS_OUT_DIR / "statistical_summary_table.csv"
    
    with open(report_md_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
        
    df_summary.to_csv(report_csv_file, index=False)
    save_dataframe_to_table(df_summary, 'audit_statistical_summary')
    
    print(f"\n[ÉXITO] Reporte Markdown exportado en: {report_md_file.name}")
    print(f"[ÉXITO] Tabla resumen exportada en CSV y persistida en SQLite.")


if __name__ == "__main__":
    start_time = time.time()
    run_statistical_interpretation()
    log_execution_time("05_statistical_interpretation", time.time() - start_time)
