# scripts/01c_audit_test_story.py
"""Orquestador de Auditoría de la Historia de Prueba.

Analiza la disponibilidad y frecuencia de la historia candidata para evaluación 
(test set) en cada participante de la cohorte. Garantiza que el conjunto de 
evaluación sea metodológicamente viable.
"""

import pandas as pd
import time
from src.db_manager import save_dataframe_to_table, log_execution_time, load_table_to_dataframe
from src.config import TEST_STORY


def audit_test_story(test_story: str = TEST_STORY) -> None:
    """Evalúa la viabilidad de una historia como conjunto de prueba.
    
    Extrae la tabla de sesiones de la base de datos SQLite y genera un reporte 
    detallado de las reproducciones por participante. Identifica carencias 
    de datos que puedan comprometer la validación cruzada.
    
    Args:
        test_story (str, opcional): Nombre de la historia a auditar. 
            Por defecto toma la variable centralizada TEST_STORY.
    """
    df_sessions = load_table_to_dataframe('audit_sessions')
    
    if df_sessions is None or df_sessions.empty:
        print("Error: No se encontró la tabla 'audit_sessions'.")
        return
        
    print(f"\n--- AUDITORÍA DE HISTORIA DE PRUEBA: '{test_story}' ---")
    
    # 1. Filtramos solo las sesiones donde se escuchó la historia objetivo
    df_test = df_sessions[df_sessions['story'] == test_story]
    
    if df_test.empty:
        print(f"[ALERTA CRÍTICA] Ningún participante escuchó '{test_story}'.")
        return
        
    # 2. Contamos cuántas veces la escuchó cada sujeto
    play_counts = df_test.groupby('subject_id').size().reset_index(name='play_count')
    
    # 3. Cruzamos con la lista total de sujetos
    all_subjects = df_sessions['subject_id'].unique()
    df_all_subjects = pd.DataFrame({'subject_id': all_subjects})
    
    report = pd.merge(df_all_subjects, play_counts, on='subject_id', how='left')
    report['play_count'] = report['play_count'].fillna(0).astype(int)
    
    # 4. Impresión del reporte analítico
    print("\nFrecuencia de escucha por participante:")
    print(report.to_string(index=False))
    
    # 5. Diagnóstico de viabilidad
    subjects_with_zero = report[report['play_count'] == 0]['subject_id'].tolist()
    
    print("\n--- DIAGNÓSTICO METODOLÓGICO ---")
    if subjects_with_zero:
        print(f"[CUIDADO] Los siguientes sujetos NO escucharon la historia: {subjects_with_zero}")
        print("No podrá evaluar el modelo (Test Set) en estos participantes usando esta historia.")
    else:
        print("[VIABILIDAD CONFIRMADA] Todos los participantes escucharon la historia al menos una vez.")
        
    avg_plays = report['play_count'].mean()
    print(f"Promedio de reproducciones por participante: {avg_plays:.2f}")
    
    multiple_plays = report[report['play_count'] > 1]
    if not multiple_plays.empty:
        print("\nParticipantes con presentaciones repetidas (Útiles para calcular el Noise Ceiling):")
        print(multiple_plays.to_string(index=False))


if __name__ == "__main__":
    start_time = time.time()
    audit_test_story()
    log_execution_time("01c_audit_test_story", time.time() - start_time)
