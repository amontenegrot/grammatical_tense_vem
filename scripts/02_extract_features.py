# scripts/02_extract_features.py
"""Orquestador de Extracción de Espacios de Características (Secuencial).

Extrae las propiedades lingüísticas y acústicas del corpus, delegando la tarea
a 6 extractores independientes. Genera matrices temporales de alta resolución (100 Hz),
verifica programáticamente la arquitectura estricta de exactamente 43 columnas 
(39 de control + 4 de tiempo finito) y exporta una muestra a SQLite para inspección.
"""

import time
from pathlib import Path

import joblib
import pandas as pd
import spacy
import tgt

from src.config import (
    DIR_ARTIFACTS,
    DIR_FEATURES_HIGH_RESOLUTION,
    DIR_TEXTGRIDS,
    FEATURE_COLUMNS_ORDER,
    PRESENTATION_DB_PATH,
    PRESENTATION_STORY,
    SPACY_MODEL_NAME,
    TOTAL_FEATURES_COUNT,
)
from src.db_manager import load_table_to_dataframe, log_execution_time, save_dataframe_to_table
from src.features.extractors import (
    FiniteTenseExtractor,
    LexicalCategoricalExtractor,
    LexicalStatsExtractor,
    PhonologicalExtractor,
    SemanticLSAExtractor,
    SyntacticExtractor,
)
from src.features.text_parser import reconstruct_and_map_text


FEATURES_OUT_DIR = DIR_FEATURES_HIGH_RESOLUTION
MODEL_OUT_PATH = DIR_ARTIFACTS / "semantic_lsa_model.joblib"


def run_extraction_pipeline() -> None:
    """Ejecuta el flujo secuencial de extracción global de características a 100 Hz.
    
    Itera sobre todas las historias validadas, carga los modelos de lenguaje 
    y semántica (ajustado en Train), y delega la proyección temporal a cada extractor.
    Garantiza la consistencia dimensional de 43 columnas antes de serializar.
    """
    FEATURES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_stories = load_table_to_dataframe('audit_stories')
    if df_stories is None or df_stories.empty:
        print("Error: No se encontró la tabla 'audit_stories'. Ejecute scripts/01a primero.")
        return
        
    stories = df_stories['story'].tolist()
    total = len(stories)
    
    if not MODEL_OUT_PATH.exists():
        print("Error: El modelo semántico no existe. Ejecute scripts/01b primero.")
        return
        
    print(f"Cargando modelo semántico global LSA entrenado en historias de ajuste...")
    semantic_pipeline = joblib.load(MODEL_OUT_PATH)
    
    print(f"Cargando modelo de lenguaje principal ({SPACY_MODEL_NAME})...")
    nlp = spacy.load(SPACY_MODEL_NAME, disable=["ner"])
    
    # Instanciación de los 6 extractores
    phono_ext = PhonologicalExtractor()
    lex_stats_ext = LexicalStatsExtractor()
    lex_cat_ext = LexicalCategoricalExtractor()
    syntactic_ext = SyntacticExtractor()
    tense_ext = FiniteTenseExtractor()
    sem_lsa_ext = SemanticLSAExtractor(semantic_pipeline, nlp)
    
    print(f"\nIniciando extracción secuencial para {total} historias (Arquitectura: 43 predictores)...")

    for i, story in enumerate(stories, 1):
        try:
            check_file = FEATURES_OUT_DIR / f"{story}_semantic.parquet"
            if check_file.exists():
                print(f"[{i}/{total}] Omitido: {story} ya fue procesada previamente.")
                continue

            tg_path = DIR_TEXTGRIDS / f"{story}.TextGrid"
            if not tg_path.exists():
                print(f"[{i}/{total}] Advertencia: TextGrid no encontrado para {story}.")
                continue
                
            full_text, df_align = reconstruct_and_map_text(str(tg_path))
            
            tg = tgt.io.read_textgrid(str(tg_path), include_empty_intervals=True)
            word_tier = next(t for t in tg.get_tier_names() if 'word' in t.lower())
            phone_tier = next(t for t in tg.get_tier_names() if 'phone' in t.lower())
            
            total_duration = tg.get_tier_by_name(word_tier).end_time
            
            # Inferencia morfosintáctica y de dependencias con spaCy
            doc = nlp(full_text)
            
            # Extracción de los 6 espacios independientes (a 100 Hz)
            df_phono = phono_ext.extract(tg.get_tier_by_name(phone_tier).intervals, total_duration)
            df_lex_stat = lex_stats_ext.extract(df_align, total_duration)
            df_lex_cat = lex_cat_ext.extract(df_align, doc, total_duration)
            df_syn = syntactic_ext.extract(df_align, doc, total_duration)
            df_tense = tense_ext.extract(df_align, doc, total_duration)
            df_sem = sem_lsa_ext.extract(tg.get_tier_by_name(word_tier).intervals, total_duration)
            
            # Verificación programática de dimensiones exactas
            combined_high_res = pd.concat(
                [df_phono, df_lex_stat, df_lex_cat, df_syn, df_sem, df_tense], 
                axis=1
            )
            
            # Validación estricta según el anteproyecto (Exactamente 43 columnas ordenadas)
            assert combined_high_res.shape[1] == TOTAL_FEATURES_COUNT, (
                f"Error en {story}: Se esperaban {TOTAL_FEATURES_COUNT} características pero se obtuvieron {combined_high_res.shape[1]}"
            )
            assert list(combined_high_res.columns) == FEATURE_COLUMNS_ORDER, (
                f"Error en {story}: El orden o nombre de las columnas no coincide con FEATURE_COLUMNS_ORDER"
            )
            
            # 1. Exportación principal en Parquet independiente por espacio
            df_phono.to_parquet(FEATURES_OUT_DIR / f"{story}_phonological.parquet")
            df_lex_stat.to_parquet(FEATURES_OUT_DIR / f"{story}_lexical_stats.parquet")
            df_lex_cat.to_parquet(FEATURES_OUT_DIR / f"{story}_categorical.parquet")
            df_syn.to_parquet(FEATURES_OUT_DIR / f"{story}_syntactic.parquet")
            df_sem.to_parquet(FEATURES_OUT_DIR / f"{story}_semantic.parquet")
            df_tense.to_parquet(FEATURES_OUT_DIR / f"{story}_tense.parquet")
            
            # 2. Exportación secundaria a SQLite solo para la historia de presentación
            # Decisión técnica: reset_index() transforma el índice 'time_seconds' en una columna 
            # visible y explícita, ideal para consultar en visores como DBeaver/DBBrowser.
            if story == PRESENTATION_STORY:
                print(f"      -> Guardando muestra representativa de {story} en SQLite...")
                spaces = {
                    'phonological': df_phono,
                    'lexical_stats': df_lex_stat,
                    'categorical': df_lex_cat,
                    'syntactic': df_syn,
                    'semantic': df_sem,
                    'tense': df_tense
                }
                for space_name, df_space in spaces.items():
                    df_presentation = df_space.reset_index()
                    save_dataframe_to_table(
                        df=df_presentation, 
                        table_name=space_name, 
                        db_path=PRESENTATION_DB_PATH
                    )
            
            print(f"[{i}/{total}] Éxito: {story} procesada ({combined_high_res.shape[0]} muestras a 100 Hz, 43 cols).")
            
        except Exception as e:
            print(f"[{i}/{total}] [ERROR CRÍTICO] en {story}: {str(e)}")


if __name__ == "__main__":
    start_time = time.time()
    run_extraction_pipeline()
    log_execution_time("02_extract_features", time.time() - start_time)
