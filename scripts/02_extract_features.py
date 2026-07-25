# scripts/02_extract_features.py
#*
"""Orquestador de Extracción de Espacios de Características (Secuencial).

Extrae propiedades lingüísticas y acústicas del corpus, delegando la tarea
a 6 espacios de características independientes. Genera matrices temporales 
de alta resolución y exporta una muestra a SQLite para presentación visual.
"""

import spacy
import tgt
import joblib
from pathlib import Path

from src.config import (
    DIR_TEXTGRIDS, 
    SPACY_MODEL_NAME, 
    DIR_FEATURES_HIGH_RESOLUTION,
    DIR_ARTIFACTS,
    PRESENTATION_STORY,
    PRESENTATION_DB_PATH,
)
import time
from src.db_manager import save_dataframe_to_table, log_execution_time, load_table_to_dataframe
from src.features.text_parser import reconstruct_and_map_text
from src.features.extractors import (
    PhonologicalExtractor, LexicalStatsExtractor, 
    LexicalCategoricalExtractor, SyntacticExtractor, 
    FiniteTenseExtractor, SemanticLSAExtractor
)


FEATURES_OUT_DIR = DIR_FEATURES_HIGH_RESOLUTION
MODEL_OUT_PATH = DIR_ARTIFACTS / "semantic_lsa_model.joblib"


def run_extraction_pipeline() -> None:
    """Ejecuta el flujo secuencial de extracción global de características.
    
    Itera sobre todas las historias validadas, carga los modelos de lenguaje 
    y semántica, y delega la proyección temporal a cada extractor. Guarda 
    los resultados en formato Parquet y exporta una muestra en SQLite.
    """
    FEATURES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_stories = load_table_to_dataframe('audit_stories')
    if df_stories is None or df_stories.empty:
        print("Error: No se encontró la tabla 'audit_stories'.")
        return
        
    stories = df_stories['story'].tolist()
    total = len(stories)
    
    if not MODEL_OUT_PATH.exists():
        print("Error: El modelo semántico no existe. Ejecute scripts/01b primero.")
        return
        
    print(f"Cargando modelo semántico global LSA...")
    semantic_pipeline = joblib.load(MODEL_OUT_PATH)
    
    print(f"Cargando modelo de lenguaje principal ({SPACY_MODEL_NAME})...")
    nlp = spacy.load(SPACY_MODEL_NAME, disable=["ner"])
    
    # Instanciar extractores
    phono_ext = PhonologicalExtractor()
    lex_stats_ext = LexicalStatsExtractor()
    lex_cat_ext = LexicalCategoricalExtractor()
    syntactic_ext = SyntacticExtractor()
    tense_ext = FiniteTenseExtractor()
    sem_lsa_ext = SemanticLSAExtractor(semantic_pipeline, nlp)
    
    print(f"Iniciando procesamiento secuencial de {total} historias...")

    for i, story in enumerate(stories, 1):
        try:
            check_file = FEATURES_OUT_DIR / f"{story}_semantic.parquet"
            if check_file.exists():
                print(f"[{i}/{total}] Omitido: {story} ya fue procesada previamente.")
                continue

            tg_path = DIR_TEXTGRIDS / f"{story}.TextGrid"
            if not tg_path.exists():
                continue
                
            full_text, df_align = reconstruct_and_map_text(str(tg_path))
            
            tg = tgt.io.read_textgrid(str(tg_path), include_empty_intervals=True)
            word_tier = next(t for t in tg.get_tier_names() if 'word' in t.lower())
            phone_tier = next(t for t in tg.get_tier_names() if 'phone' in t.lower())
            
            total_duration = tg.get_tier_by_name(word_tier).end_time
            
            # Inferencia principal
            doc = nlp(full_text)
            
            # Delegación a extractores (Devuelven DataFrames con el índice temporal)
            df_phono = phono_ext.extract(tg.get_tier_by_name(phone_tier).intervals, total_duration)
            df_lex_stat = lex_stats_ext.extract(df_align, total_duration)
            df_lex_cat = lex_cat_ext.extract(df_align, doc, total_duration)
            df_syn = syntactic_ext.extract(df_align, doc, total_duration)
            df_tense = tense_ext.extract(df_align, doc, total_duration)
            df_sem = sem_lsa_ext.extract(tg.get_tier_by_name(word_tier).intervals, total_duration)
            
            # 1. Exportación principal en Parquet (Rápida y comprimida para el pipeline)
            df_phono.to_parquet(FEATURES_OUT_DIR / f"{story}_phonological.parquet")
            df_lex_stat.to_parquet(FEATURES_OUT_DIR / f"{story}_lexical_stats.parquet")
            df_lex_cat.to_parquet(FEATURES_OUT_DIR / f"{story}_categorical.parquet")
            df_syn.to_parquet(FEATURES_OUT_DIR / f"{story}_syntactic.parquet")
            df_tense.to_parquet(FEATURES_OUT_DIR / f"{story}_tense.parquet")
            df_sem.to_parquet(FEATURES_OUT_DIR / f"{story}_semantic.parquet")
            
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
                    'tense': df_tense,
                    'semantic': df_sem
                }
                for space_name, df_space in spaces.items():
                    df_presentation = df_space.reset_index()
                    save_dataframe_to_table(
                        df=df_presentation, 
                        table_name=space_name, 
                        db_path=PRESENTATION_DB_PATH
                    )
            
            print(f"[{i}/{total}] Procesamiento completo para: {story}")
            
        except Exception as e:
            print(f"[{i}/{total}] Error en {story}: {str(e)}")

if __name__ == "__main__":
    start_time = time.time()
    run_extraction_pipeline()
    log_execution_time("02_extract_features", time.time() - start_time)
