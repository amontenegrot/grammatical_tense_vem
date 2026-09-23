# scripts/01b_train_semantic_model.py
"""Orquestador de Entrenamiento del Modelo Semántico Latente (LSA).

Entrena un pipeline scikit-learn (TF-IDF + TruncatedSVD a 10 componentes)
exclusivamente sobre las historias de entrenamiento, garantizando la total 
prevención de fuga de datos (Data Leakage) respecto a la historia de prueba.
Registra formalmente un reporte de trazabilidad y cobertura léxica.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import spacy
import tgt

from src.config import (
    DIR_ARTIFACTS,
    DIR_TEXTGRIDS,
    EXCLUDED_STORIES,
    LATENT_SEMANTIC_COMPONENTS,
    LSA_CUSTOM_STOP_WORDS,
    LSA_MAX_DF,
    LSA_MIN_DF,
    LSA_NOISE_PATTERN,
    NLP_BATCH_SIZE,
    RANDOM_SEED,
    SPACY_MODEL_NAME,
    TEST_STORY,
    THRESHOLD_PERIOD_SEC,
)
from src.db_manager import load_table_to_dataframe, log_execution_time, save_dataframe_to_table


MODEL_OUT_PATH = DIR_ARTIFACTS / "semantic_lsa_model.joblib"
TRACEABILITY_REPORT_PATH = DIR_ARTIFACTS / "semantic_model_traceability.json"


def extract_sentences_from_corpus(stories: List[str]) -> List[str]:
    """Segmenta las historias en unidades textuales delimitadas por pausas acústicas >= 0.65s."""
    all_sentences = []
    
    for story in stories:
        tg_path = DIR_TEXTGRIDS / f"{story}.TextGrid"
        if not tg_path.exists():
            continue
            
        try:
            textgrid = tgt.io.read_textgrid(str(tg_path), include_empty_intervals=True)
            word_tier = next(t for t in textgrid.get_tier_names() if 'word' in t.lower())
            
            current_sentence = []
            for interval in textgrid.get_tier_by_name(word_tier).intervals:
                token = interval.text.strip()
                is_empty = (token == "")
                is_noise = bool(LSA_NOISE_PATTERN.search(token))
                
                if is_empty or is_noise:
                    duration = interval.end_time - interval.start_time
                    if duration >= THRESHOLD_PERIOD_SEC and current_sentence:
                        all_sentences.append(" ".join(current_sentence))
                        current_sentence = []
                    continue
                
                clean_word = re.sub(r'[^a-zA-Z\']', '', token).lower()
                if clean_word:
                    current_sentence.append(clean_word)
                    
            if current_sentence:
                all_sentences.append(" ".join(current_sentence))
                
        except Exception as e:
            print(f"Error procesando historia {story} para extracción semántica: {e}")
            
    return all_sentences


def train_and_save_semantic_model() -> None:
    """Ejecuta el entrenamiento del modelo LSA con control estricto de fuga de datos."""
    DIR_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    
    df_stories = load_table_to_dataframe('audit_stories')
    if df_stories is None or df_stories.empty:
        print("Error: No se encontró la tabla 'audit_stories'. Ejecute 01a primero.")
        return
        
    all_stories = df_stories['story'].tolist()
    
    # Decisión metodológica estricta: Entrenar ÚNICAMENTE con historias de entrenamiento
    train_stories = [
        s for s in all_stories 
        if s != TEST_STORY and s not in EXCLUDED_STORIES
    ]
    
    print(f"--- Entrenamiento Semántico LSA (Prevención de Fuga de Datos) ---")
    print(f"Total historias en corpus: {len(all_stories)}")
    print(f"Historias para entrenamiento: {len(train_stories)}")
    print(f"Historia reservada excluida del ajuste: '{TEST_STORY}'")
    
    print("\n1. Extrayendo unidades textuales de entrenamiento...")
    raw_sentences = extract_sentences_from_corpus(train_stories)
    print(f"   Unidades textuales extraídas: {len(raw_sentences)}")
    
    print(f"2. Cargando modelo {SPACY_MODEL_NAME} para lematización...")
    nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "ner"])
    
    print("3. Lematizando y filtrando stop words (CPU n_process=1 para evitar deadlocks)...")
    clean_sentences = []
    for doc in nlp.pipe(raw_sentences, batch_size=NLP_BATCH_SIZE, n_process=1):
        lemmas = [
            t.lemma_.lower() for t in doc 
            if not t.is_stop and not t.is_punct and not t.like_num and t.lemma_.strip()
        ]
        filtered_lemmas = [w for w in lemmas if w not in LSA_CUSTOM_STOP_WORDS]
        if filtered_lemmas:
            clean_sentences.append(" ".join(filtered_lemmas))
            
    print(f"   Unidades textuales válidas para ajuste: {len(clean_sentences)}")
    
    print("4. Ajustando Pipeline (TF-IDF + TruncatedSVD a 10 componentes)...")
    tfidf = TfidfVectorizer(min_df=LSA_MIN_DF, max_df=LSA_MAX_DF)
    svd = TruncatedSVD(n_components=LATENT_SEMANTIC_COMPONENTS, random_state=RANDOM_SEED)
    
    semantic_pipeline = Pipeline([
        ('tfidf', tfidf),
        ('svd', svd)
    ])
    
    semantic_pipeline.fit(clean_sentences)
    
    # Serializar el modelo entrenado
    joblib.dump(semantic_pipeline, MODEL_OUT_PATH)
    print(f"Modelo semántico serializado en: {MODEL_OUT_PATH.name}")
    
    # 5. Auditoría de Trazabilidad Léxica y Cobertura sobre la Historia de Prueba
    print("\n5. Calculando cobertura léxica sobre la historia reservada...")
    vocab_learned = tfidf.vocabulary_
    test_sentences_raw = extract_sentences_from_corpus([TEST_STORY])
    
    test_words = []
    for doc in nlp.pipe(test_sentences_raw, batch_size=NLP_BATCH_SIZE, n_process=1):
        lemmas = [
            t.lemma_.lower() for t in doc 
            if not t.is_stop and not t.is_punct and not t.like_num and t.lemma_.strip()
        ]
        test_words.extend([w for w in lemmas if w not in LSA_CUSTOM_STOP_WORDS])
        
    total_test_tokens = len(test_words)
    covered_test_tokens = sum(1 for w in test_words if w in vocab_learned)
    oov_tokens = total_test_tokens - covered_test_tokens
    coverage_pct = (covered_test_tokens / total_test_tokens * 100) if total_test_tokens > 0 else 0.0
    
    traceability_data = {
        'training_stories_count': len(train_stories),
        'test_story_excluded': TEST_STORY,
        'latent_components': LATENT_SEMANTIC_COMPONENTS,
        'min_df': LSA_MIN_DF,
        'max_df': LSA_MAX_DF,
        'vocabulary_size': len(vocab_learned),
        'explained_variance_ratio_sum': float(svd.explained_variance_ratio_.sum()),
        'test_total_tokens': total_test_tokens,
        'test_covered_tokens': covered_test_tokens,
        'test_oov_tokens': oov_tokens,
        'test_lexical_coverage_pct': round(coverage_pct, 2)
    }
    
    with open(TRACEABILITY_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(traceability_data, f, indent=4)
        
    df_traceability = pd.DataFrame([traceability_data])
    save_dataframe_to_table(df_traceability, 'audit_semantic_traceability')
    
    print(f"Reporte de trazabilidad persistido en: {TRACEABILITY_REPORT_PATH.name}")
    print(f"Vocabulario aprendido: {len(vocab_learned)} términos.")
    print(f"Cobertura léxica de '{TEST_STORY}': {coverage_pct:.2f}% ({covered_test_tokens}/{total_test_tokens} tokens).")


if __name__ == "__main__":
    start_time = time.time()
    train_and_save_semantic_model()
    log_execution_time("01b_train_semantic_model", time.time() - start_time)
