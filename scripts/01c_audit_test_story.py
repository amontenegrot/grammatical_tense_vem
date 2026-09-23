# scripts/01c_audit_test_story.py
"""Orquestador de Auditoría de la Historia de Prueba.

Analiza la disponibilidad, sesiones, corridas y repeticiones de la historia 
de evaluación fuera de muestra ('wheretheressmoke') por participante.
Documenta formalmente el desbalance lingüístico (303 formas finitas: 20.13% pasado vs 79.87% no pasado)
y la base para el promedio temporal de repeticiones BOLD.
"""

import time
import pandas as pd

from src.config import TEST_STORY
from src.db_manager import load_table_to_dataframe, log_execution_time, save_dataframe_to_table


# Métricas documentadas en la auditoría lingüística formal del anteproyecto
TEST_STORY_LINGUISTIC_PROFILE = {
    'story': TEST_STORY,
    'total_finite_forms': 303,
    'past_forms_count': 61,
    'past_forms_pct': 20.13,
    'non_past_forms_count': 242,
    'non_past_forms_pct': 79.87,
    'note': 'Predominio no pasado; reservada para evaluación con promedio temporal de repeticiones'
}


def audit_test_story(test_story: str = TEST_STORY) -> None:
    """Evalúa la viabilidad metodológica y repeticiones de la historia de prueba."""
    df_sessions = load_table_to_dataframe('audit_sessions')
    
    if df_sessions is None or df_sessions.empty:
        print("Error: No se encontró la tabla 'audit_sessions'. Ejecute 01a primero.")
        return
        
    print(f"\n=================================================================")
    print(f"AUDITORÍA DE LA HISTORIA DE EVALUACIÓN FUERA DE MUESTRA: '{test_story}'")
    print(f"=================================================================")
    
    # 1. Filtrar sesiones correspondientes a la historia de prueba
    df_test = df_sessions[df_sessions['story'] == test_story]
    
    if df_test.empty:
        print(f"[ALERTA CRÍTICA] Ningún participante tiene registros para '{test_story}'.")
        return
        
    # 2. Conteo de presentaciones / repeticiones por participante
    play_counts = df_test.groupby('subject_id').agg(
        presentations_count=('run_id', 'count'),
        sessions_list=('session_id', lambda x: ", ".join(sorted(x.unique()))),
        runs_list=('run_id', lambda x: ", ".join(x))
    ).reset_index()
    
    all_subjects = sorted(df_sessions['subject_id'].unique())
    df_all_subjects = pd.DataFrame({'subject_id': all_subjects})
    
    report = pd.merge(df_all_subjects, play_counts, on='subject_id', how='left')
    report['presentations_count'] = report['presentations_count'].fillna(0).astype(int)
    
    print("\nDisponibilidad de presentaciones por participante:")
    print(report[['subject_id', 'presentations_count', 'sessions_list']].to_string(index=False))
    
    # 3. Diagnóstico de repeticiones para SNR Boosting (Promedio Temporal)
    single_rep = report[report['presentations_count'] == 1]
    multi_rep = report[report['presentations_count'] > 1]
    
    print("\n--- DIAGNÓSTICO METODOLÓGICO DE SNR ---")
    print(f"Participantes con repeticiones múltiples (Permiten promedio de señal BOLD): {len(multi_rep)}")
    print(f"Participantes con presentación única: {len(single_rep)}")
    
    # 4. Registro formal del perfil lingüístico de la historia de prueba
    df_profile = pd.DataFrame([TEST_STORY_LINGUISTIC_PROFILE])
    save_dataframe_to_table(report, 'audit_test_story_coverage')
    save_dataframe_to_table(df_profile, 'audit_test_story_linguistic_profile')
    
    print("\n--- PERFIL LINGÜÍSTICO DOCUMENTADO (Anteproyecto) ---")
    print(f"Formas finitas totales: {TEST_STORY_LINGUISTIC_PROFILE['total_finite_forms']}")
    print(f"Pasado: {TEST_STORY_LINGUISTIC_PROFILE['past_forms_count']} ({TEST_STORY_LINGUISTIC_PROFILE['past_forms_pct']}%)")
    print(f"No Pasado: {TEST_STORY_LINGUISTIC_PROFILE['non_past_forms_count']} ({TEST_STORY_LINGUISTIC_PROFILE['non_past_forms_pct']}%)")
    print("Nota: El desbalance será considerado formalmente en la interpretación de los escenarios de resultado.")


if __name__ == "__main__":
    start_time = time.time()
    audit_test_story()
    log_execution_time("01c_audit_test_story", time.time() - start_time)
