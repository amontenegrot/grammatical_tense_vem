# scripts/01b_train_semantic_model.py
#*
"""Orquestador de Entrenamiento del Modelo Semántico Latente.

Extrae el texto continuo de todo el corpus validado, lo lematiza eliminando 
muletillas conversacionales y entrena un pipeline global (TF-IDF + TruncatedSVD) 
para capturar la dinámica temática abstracta.
"""

import re
import spacy
import joblib
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import tgt

from src.config import (
    DIR_TEXTGRIDS, 
    DIR_ARTIFACTS, 
    SPACY_MODEL_NAME, 
    THRESHOLD_PERIOD_SEC, 
    LATENT_SEMANTIC_COMPONENTS,
    LSA_CUSTOM_STOP_WORDS,
    LSA_NOISE_PATTERN,
    LSA_MIN_DF,
    LSA_MAX_DF,
    NLP_BATCH_SIZE,
    RANDOM_SEED,
)
import time
from src.db_manager import save_dataframe_to_table, log_execution_time, load_table_to_dataframe


MODEL_OUT_PATH = DIR_ARTIFACTS / "semantic_lsa_model.joblib"


def extract_sentences_from_corpus(stories: List[str]) -> List[str]:
    """Segmenta las historias en oraciones utilizando pausas acústicas.
    
    Procesa iterativamente los archivos TextGrid, ignorando intervalos vacíos 
    y ruido de fondo (como respiraciones o risas), agrupando las palabras en 
    oraciones lógicas separadas por pausas mayores al umbral definido.

    Args:
        stories (List[str]): Lista con los nombres de las historias a procesar.
        
    Returns:
        List[str]: Lista global de oraciones (texto crudo) de todo el corpus.
    """
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
    """Ejecuta el flujo completo de entrenamiento y serialización del modelo LSA.
    
    Se encarga de:
    1. Extraer el texto de la base de datos de auditoría.
    2. Lematizar y limpiar usando spaCy (controlando stop-words).
    3. Entrenar el pipeline de TF-IDF acoplado a la reducción dimensional.
    4. Serializar el artefacto entrenado para su inferencia posterior.
    """
    DIR_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    
    df_stories = load_table_to_dataframe('audit_stories')
    if df_stories is None or df_stories.empty:
        print("Error: No se encontró la tabla 'audit_stories'.")
        return
        
    stories = df_stories['story'].tolist()
    
    print("1. Extrayendo oraciones basadas en pausas articulatorias...")
    raw_sentences = extract_sentences_from_corpus(stories)
    print(f"   Oraciones extraídas: {len(raw_sentences)}")
    
    print(f"2. Cargando modelo {SPACY_MODEL_NAME} para lematización...")
    nlp = spacy.load(SPACY_MODEL_NAME, disable=["parser", "ner"])
    
    print("3. Lematizando y filtrando stop words (Optimizando uso de tensores CPU)...")
    clean_sentences = []
    
    # Procesamiento secuencial optimizado (n_process=1 previene deadlocks en PyTorch)
    for doc in nlp.pipe(raw_sentences, batch_size=NLP_BATCH_SIZE, n_process=1):
        lemmas = [
            t.lemma_.lower() for t in doc 
            if not t.is_stop and not t.is_punct and not t.like_num and t.lemma_.strip()
        ]
        filtered_lemmas = [w for w in lemmas if w not in LSA_CUSTOM_STOP_WORDS]
        
        if filtered_lemmas:
            clean_sentences.append(" ".join(filtered_lemmas))
            
    print(f"   Oraciones válidas para entrenamiento: {len(clean_sentences)}")
    
    print("4. Entrenando Pipeline (TF-IDF + TruncatedSVD)...")
    semantic_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=LSA_MIN_DF, max_df=LSA_MAX_DF)),
        ('svd', TruncatedSVD(n_components=LATENT_SEMANTIC_COMPONENTS, random_state=RANDOM_SEED))
    ])
    
    semantic_pipeline.fit(clean_sentences)
    
    joblib.dump(semantic_pipeline, MODEL_OUT_PATH)
    print(f"Entrenamiento completado. Modelo persistido en: {MODEL_OUT_PATH.name}")


if __name__ == "__main__":
    start_time = time.time()
    train_and_save_semantic_model()
    log_execution_time("01b_train_semantic_model", time.time() - start_time)
