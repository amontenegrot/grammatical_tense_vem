# scripts/debug_story.py
"""Script de diagnóstico puntual para historias con error."""
import sys
from pathlib import Path

# --- GARANTIZAR QUE PYTHON ENCUENTRE EL MÓDULO 'src' ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -------------------------------------------------------

import traceback
import spacy
import tgt

from src.config import DIR_TEXTGRIDS, SPACY_MODEL_NAME
from src.features.text_parser import reconstruct_and_map_text
from src.features.extractors import (
    PhonologicalExtractor,
    LexicalStatsExtractor,
    LexicalCategoricalExtractor,
    SyntacticExtractor,
    FiniteTenseExtractor,
)

STORIES_TO_TEST = ['legacy', 'exorcism']

def debug_stories():
    print("Iniciando diagnóstico puntual...\n")
    print(f"Ruta raíz del proyecto: {PROJECT_ROOT}\n")
    
    try:
        nlp = spacy.load(SPACY_MODEL_NAME, disable=["ner"])
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo spaCy '{SPACY_MODEL_NAME}': {e}")
        return
    
    phono_ext = PhonologicalExtractor()
    lex_stats_ext = LexicalStatsExtractor()
    lex_cat_ext = LexicalCategoricalExtractor()
    syntactic_ext = SyntacticExtractor()
    tense_ext = FiniteTenseExtractor()
    
    for story in STORIES_TO_TEST:
        print(f"==================================================")
        print(f"DIAGNOSTICANDO: {story}")
        print(f"==================================================")
        
        tg_path = DIR_TEXTGRIDS / f"{story}.TextGrid"
        if not tg_path.exists():
            print(f"[ERROR] No existe el archivo: {tg_path}\n")
            continue
            
        # 1. Prueba de lectura con tgt
        try:
            tg = tgt.io.read_textgrid(str(tg_path), include_empty_intervals=True)
            tier_names = tg.get_tier_names()
            print(f"1. Lectura Praat (tgt): OK. Tiers detectados: {tier_names}")
        except Exception as e:
            print(f"1. Lectura Praat (tgt): FALLÓ")
            print(traceback.format_exc())
            continue

        # 2. Prueba de detección de tiers
        try:
            word_tier = next(t for t in tier_names if 'word' in t.lower())
            phone_tier = next(t for t in tier_names if 'phone' in t.lower())
            total_duration = tg.get_tier_by_name(word_tier).end_time
            print(f"2. Detección de Tiers: OK (Word: '{word_tier}', Phone: '{phone_tier}', Duración: {total_duration:.2f}s)")
        except Exception as e:
            print(f"2. Detección de Tiers: FALLÓ")
            print(traceback.format_exc())
            continue

        # 3. Prueba de reconstrucción de texto y mapeo de caracteres
        try:
            full_text, df_align = reconstruct_and_map_text(str(tg_path))
            print(f"3. Reconstrucción de texto: OK ({len(full_text)} caracteres, {len(df_align)} palabras alineadas)")
            print(f"   Muestra de texto: '{full_text[:80]}...'")
        except Exception as e:
            print(f"3. Reconstrucción de texto: FALLÓ")
            print(traceback.format_exc())
            continue

        # 4. Inferencia con spaCy
        try:
            doc = nlp(full_text)
            print(f"4. spaCy NLP: OK ({len(doc)} tokens procesados)")
        except Exception as e:
            print(f"4. spaCy NLP: FALLÓ")
            print(traceback.format_exc())
            continue

        # 5. Extracción de los espacios
        try:
            df_phono = phono_ext.extract(tg.get_tier_by_name(phone_tier).intervals, total_duration)
            df_lex_stat = lex_stats_ext.extract(df_align, total_duration)
            df_lex_cat = lex_cat_ext.extract(df_align, doc, total_duration)
            df_syn = syntactic_ext.extract(df_align, doc, total_duration)
            df_tense = tense_ext.extract(df_align, doc, total_duration)
            print("5. Extracción de características (5 espacios): OK")
        except Exception as e:
            print("5. Extracción de características: FALLÓ")
            print(traceback.format_exc())
            continue

        print(f"\n>> RESULTADO: {story} se procesó correctamente en esta prueba.\n")

if __name__ == "__main__":
    debug_stories()
