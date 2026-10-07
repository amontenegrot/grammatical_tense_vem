# scripts/01a_audit_dataset.py
"""Orquestador de Auditoría del Conjunto de Datos.

Cruza archivos fMRI HDF5, TextGrids y audios WAV para generar un mapeo 
analítico integral de la cohorte experimental. Extrae de forma sistemática
los identificadores BIDS (sujeto, sesión, corrida, presentación y narración),
verificando la completitud de los datos para el flujo de modelamiento predictivo.
"""

import re
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tgt

from src.config import (
    DIR_AUDIT_REPORTS,
    DIR_FMRI,
    DIR_STIMULI,
    EXCLUDED_STORIES,
)
from src.db_manager import log_execution_time, save_dataframe_to_table
from src.features.text_parser import discover_available_stories, resolve_textgrid_path


def extract_wav_duration(filepath: Path) -> float:
    """Extrae la duración del audio usando metadatos de cabecera en O(1)."""
    try:
        with wave.open(str(filepath), 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def parse_bids_fmri_filename(hf5_path: Path) -> Optional[Dict]:
    """Extrae los identificadores BIDS de un archivo funcional HDF5.
    
    Captura: subject_id, session_id, run_id, presentation_id y el nombre de la historia.
    """
    filename = hf5_path.stem
    
    # Captura de sujeto (ej. sub-UTS01 o UTS01)
    sub_match = re.search(r'(?:sub-)?(UTS\d{2})', str(hf5_path), re.IGNORECASE)
    if not sub_match:
        return None
    subject_id = f"sub-{sub_match.group(1).upper()}"
    
    # Captura de sesión BIDS (ej. ses-1, ses-02)
    ses_match = re.search(r'ses-(\d+)', filename, re.IGNORECASE)
    session_id = f"ses-{int(ses_match.group(1))}" if ses_match else "ses-unknown"
    
    # Captura de corrida/run (ej. run-1, run-02)
    run_match = re.search(r'run-(\d+)', filename, re.IGNORECASE)
    run_id = f"run-{int(run_match.group(1))}" if run_match else "run-1"
    
    return {
        'subject_id': subject_id,
        'session_id': session_id,
        'run_id': run_id,
        'filename': filename,
        'fmri_path': str(hf5_path)
    }


def validate_textgrid_content(story: str) -> bool:
    """Verifica que el TextGrid de una historia sea realmente parseable y utilizable.

    A diferencia de una comprobación por nombre de archivo, intenta leer el
    TextGrid con tgt (priorizando la versión reparada vía resolve_textgrid_path)
    y exige la presencia de las capas 'word' y 'phone', que son las que
    scripts/02_extract_features.py requiere más adelante en el pipeline.
    """
    tg_path = resolve_textgrid_path(story)
    if not tg_path.exists():
        return False

    try:
        textgrid = tgt.io.read_textgrid(str(tg_path), include_empty_intervals=True)
        tier_names = [name.lower() for name in textgrid.get_tier_names()]
        has_word_tier = any('word' in name for name in tier_names)
        has_phone_tier = any('phone' in name for name in tier_names)
        return has_word_tier and has_phone_tier
    except Exception:
        return False


def audit_experimental_corpus() -> None:
    """Ejecuta la auditoría cruzada entre estímulos, transcripciones y fMRI."""
    print("Iniciando auditoría del corpus empírico (ds003020 v3.1.1)...")

    # 1. Auditoría de TextGrids (Inventario lingüístico disponible)
    # Decisión técnica: se usa discover_available_stories (fuente cruda o reparada)
    # en vez de listar únicamente DIR_TEXTGRIDS, para no perder historias cuyo
    # TextGrid original no es parseable pero sí cuentan con una versión reparada
    # (generada por scripts/00b_fix_chronological_textgrids.py).
    available_stories = discover_available_stories()
    valid_stories = {s for s in available_stories if s not in EXCLUDED_STORIES}
    print(f"Detectadas {len(valid_stories)} historias con TextGrid disponible (Historias no excluidas).")

    # 1b. Validación de contenido real (no solo existencia del archivo por nombre)
    print("Validando legibilidad real de cada TextGrid (parseo + capas 'word'/'phone')...")
    textgrid_validity = {story: validate_textgrid_content(story) for story in valid_stories}
    n_valid_content = sum(textgrid_validity.values())
    print(f"TextGrids efectivamente parseables: {n_valid_content}/{len(valid_stories)}.")

    # 2. Auditoría de duraciones de audio (Estímulos)
    wav_files = list(DIR_STIMULI.rglob("*.wav"))
    story_durations = {}
    for wav in wav_files:
        story_name = wav.stem.replace('task-', '').replace('_audio', '')
        if story_name in valid_stories:
            story_durations[story_name] = extract_wav_duration(wav)

    # 3. Auditoría de archivos fMRI (Inventario neurofuncional efectivo)
    hf5_files = list(DIR_FMRI.rglob("*.hf5"))
    sessions_data: List[Dict] = []

    for hf5 in hf5_files:
        bids_meta = parse_bids_fmri_filename(hf5)
        if not bids_meta:
            continue
            
        story_match = None
        for story in valid_stories:
            if story.lower() in hf5.name.lower():
                story_match = story
                break
                
        if story_match:
            sessions_data.append({
                'subject_id': bids_meta['subject_id'],
                'session_id': bids_meta['session_id'],
                'run_id': bids_meta['run_id'],
                'story': story_match,
                'fmri_path': bids_meta['fmri_path'],
                'has_textgrid': True,
                'duration_sec': story_durations.get(story_match, 0.0)
            })

    df_sessions = pd.DataFrame(sessions_data)

    if df_sessions.empty:
        print("[ERROR CRÍTICO] No se detectaron sesiones fMRI compatibles. Verifique rutas.")
        return

    # 4. Agregación y métricas por participante e historia
    df_subjects = df_sessions.groupby('subject_id').agg(
        total_sessions_listened=('story', 'count'),
        unique_stories_listened=('story', 'nunique'),
        total_listening_time_sec=('duration_sec', 'sum')
    ).reset_index()
    df_subjects['total_listening_time_min'] = df_subjects['total_listening_time_sec'] / 60.0

    df_stories = df_sessions.groupby('story').agg(
        total_plays=('subject_id', 'count'),
        unique_listeners=('subject_id', 'nunique'),
        duration_sec=('duration_sec', 'max')
    ).reset_index()
    df_stories['has_valid_textgrid'] = df_stories['story'].map(textgrid_validity).fillna(False)

    # 5. Persistencia en SQLite y exportación a CSV
    print("Persistiendo metadatos en SQLite y CSV...")
    save_dataframe_to_table(df_sessions, 'audit_sessions')
    save_dataframe_to_table(df_subjects, 'audit_subjects')
    save_dataframe_to_table(df_stories, 'audit_stories')

    DIR_AUDIT_REPORTS.mkdir(parents=True, exist_ok=True)
    df_subjects.to_csv(DIR_AUDIT_REPORTS / "audit_subjects.csv", index=False)
    df_stories.to_csv(DIR_AUDIT_REPORTS / "audit_stories.csv", index=False)
    df_sessions.to_csv(DIR_AUDIT_REPORTS / "audit_sessions.csv", index=False)

    print("\n--- Resumen de Auditoría ---")
    n_invalid = int((~df_stories['has_valid_textgrid']).sum())
    if n_invalid > 0:
        invalid_list = df_stories.loc[~df_stories['has_valid_textgrid'], 'story'].tolist()
        print(f"[ALERTA] {n_invalid} historia(s) con sesiones fMRI pero TextGrid no parseable: {invalid_list}")
    print(f"Total Participantes: {df_subjects.shape[0]}")
    print(f"Total Sesiones fMRI compatibles: {df_sessions.shape[0]}")
    print("\nMuestra de Métricas por Participante:")
    print(df_subjects[['subject_id', 'total_sessions_listened', 'unique_stories_listened', 'total_listening_time_min']])


if __name__ == "__main__":
    start_time = time.time()
    audit_experimental_corpus()
    log_execution_time("01a_audit_dataset", time.time() - start_time)
