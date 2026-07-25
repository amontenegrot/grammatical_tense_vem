# scripts/01a_audit_dataset.py
#*
"""Orquestador de Auditoría del Conjunto de Datos.

Cruza archivos fMRI HDF5, TextGrids y audios WAV para generar un mapeo 
analítico integral de la cohorte experimental. Extrae métricas eficientes 
en hardware (ej. tiempo acumulado de exposición por sujeto) y verifica 
la completitud de los datos para el flujo de modelamiento predictivo.
"""

import re
import wave
import pandas as pd
from pathlib import Path

from src.config import (
    DIR_TEXTGRIDS, 
    DIR_FMRI, 
    DIR_STIMULI, 
    DIR_AUDIT_REPORTS, 
    EXCLUDED_STORIES,
)
import time
from src.db_manager import save_dataframe_to_table, log_execution_time


def extract_wav_duration(filepath: Path) -> float:
    """Extrae la duración del audio usando metadatos de cabecera (O(1)).

    Args:
        filepath (Path): Ruta absoluta o relativa al archivo .wav.

    Returns:
        float: Duración del audio en segundos. Retorna 0.0 si el archivo 
            está corrupto o ausente.
    """
    try:
        with wave.open(str(filepath), 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def audit_experimental_corpus() -> None:
    """Ejecuta la auditoría cruzada entre estímulos, transcripciones y fMRI.

    Cruza los archivos disponibles, puebla DataFrames agregados a nivel de 
    sujeto, historia y sesión, y los serializa en la base de datos SQLite 
    central y en archivos de reporte CSV.
    """
    print("Iniciando auditoría del corpus empírico...")

    # 1. Auditoría de TextGrids (Inventario lingüístico disponible)
    textgrids = list(DIR_TEXTGRIDS.rglob("*.TextGrid"))
    valid_stories = {tg.stem for tg in textgrids if tg.stem not in EXCLUDED_STORIES}
    print(f"Detectados {len(valid_stories)} TextGrids válidos.")

    # 2. Auditoría de duraciones de audio (Estímulos)
    wav_files = list(DIR_STIMULI.rglob("*.wav"))
    story_durations = {}
    for wav in wav_files:
        # Lógica de extracción de nombre basada en el estándar BIDS esperado
        story_name = wav.stem.replace('task-', '').replace('_audio', '')
        if story_name in valid_stories:
            story_durations[story_name] = extract_wav_duration(wav)
            
    # 3. Auditoría de archivos fMRI (Inventario neurofuncional disponible)
    hf5_files = list(DIR_FMRI.rglob("*.hf5"))
    
    sessions_data = []

    for hf5 in hf5_files:
        # Captura de identificador (ej. UTS01) ignorando mayúsculas/minúsculas
        subject_match = re.search(r'(?:sub-)?(UTS\d{2})', str(hf5), re.IGNORECASE)
        
        story_match = None
        for story in valid_stories:
            if story.lower() in hf5.name.lower():
                story_match = story
                break
                
        if subject_match and story_match:
            # Normalización del identificador forzando el prefijo 'sub-'
            subject_id = f"sub-{subject_match.group(1).upper()}"
            sessions_data.append({
                'subject_id': subject_id,
                'story': story_match,
                'fmri_path': str(hf5),
                'has_textgrid': True,
                'duration_sec': story_durations.get(story_match, 0.0)
            })

    df_sessions = pd.DataFrame(sessions_data)

    if df_sessions.empty:
        print("Advertencia: No se detectaron sesiones fMRI compatibles. Verifique rutas.")
        return

    # 4. Agregación y cálculo de métricas analíticas
    df_subjects = df_sessions.groupby('subject_id').agg(
        total_stories_listened=('story', 'count'),
        unique_stories_listened=('story', 'nunique'),
        total_listening_time_sec=('duration_sec', 'sum')
    ).reset_index()
    
    df_subjects['total_listening_time_min'] = df_subjects['total_listening_time_sec'] / 60.0

    df_stories = df_sessions.groupby('story').agg(
        total_plays=('subject_id', 'count'),
        unique_listeners=('subject_id', 'nunique'),
        duration_sec=('duration_sec', 'max')
    ).reset_index()
    
    # 5. Serialización (SQLite + CSV)
    print("Persistiendo resultados en la base de datos SQLite 'audit_metadata'...")
    save_dataframe_to_table(df_sessions, 'audit_sessions')
    save_dataframe_to_table(df_subjects, 'audit_subjects')
    save_dataframe_to_table(df_stories, 'audit_stories')
    
    # Exportación de reportes CSV centralizados
    DIR_AUDIT_REPORTS.mkdir(parents=True, exist_ok=True)
    df_subjects.to_csv(DIR_AUDIT_REPORTS / "audit_subjects.csv", index=False)
    df_stories.to_csv(DIR_AUDIT_REPORTS / "audit_stories.csv", index=False)
    df_sessions.to_csv(DIR_AUDIT_REPORTS / "audit_sessions.csv", index=False)
    
    print("\n--- Resumen de Auditoría ---")
    print(f"Total Participantes: {df_subjects.shape[0]}")
    print(f"Total Sesiones (Cruces fMRI): {df_sessions.shape[0]}")
    print("\nMuestra de Métricas por Participante:")
    print(df_subjects[['subject_id', 'total_stories_listened', 'total_listening_time_min']])


if __name__ == "__main__":
    start_time = time.time()
    audit_experimental_corpus()
    log_execution_time("01a_audit_dataset", time.time() - start_time)
