# src/features/text_parser.py
#*
"""Módulo de reconstrucción y alineación de texto.

Provee utilidades para convertir transcripciones fragmentadas de TextGrid 
en texto continuo con puntuación simulada (comas y puntos), preservando 
estrictamente la alineación temporal original de cada caracter.
"""

import re
from typing import Dict, List, Tuple

import pandas as pd
import tgt

from src.config import (
    LSA_NOISE_PATTERN, 
    THRESHOLD_COMMA_SEC, 
    THRESHOLD_PERIOD_SEC,
)


def reconstruct_and_map_text(textgrid_path: str) -> Tuple[str, pd.DataFrame]:
    """Reconstruye el texto continuo simulando sintaxis e indexando caracteres.
    
    Lee un archivo TextGrid, utiliza las duraciones de las pausas (silencios) 
    para inferir lógicamente las comas y los puntos, e ignora marcadores de ruido. 
    Genera un mapeo que relaciona el índice espacial de cada caracter con su 
    tiempo de inicio y fin en el audio original.
    
    Args:
        textgrid_path (str): Ruta absoluta al archivo TextGrid.
        
    Returns:
        Tuple[str, pd.DataFrame]: Una tupla que contiene:
            - str: Cadena de texto continuo con puntuación ortográfica.
            - pd.DataFrame: Matriz con las columnas 'original_word', 'start_time', 
              'end_time', 'char_start', 'char_end'.
              
    Raises:
        ValueError: Si la capa (tier) 'words' no es encontrada dentro del TextGrid.
    """
    textgrid = tgt.io.read_textgrid(textgrid_path, include_empty_intervals=True)
    
    word_tier = None
    for name in textgrid.get_tier_names():
        if 'word' in name.lower():
            word_tier = textgrid.get_tier_by_name(name)
            break
            
    if word_tier is None:
        raise ValueError(f"Capa 'words' ausente en {textgrid_path}")
        
    raw_tokens: List[str] = []
    word_mapping: List[Dict[str, float]] = []
    
    for interval in word_tier.intervals:
        token = interval.text.strip()
        is_empty = (token == "")
        # Decisión técnica: Se aplica el patrón centralizado para descartar ruido
        is_noise = bool(LSA_NOISE_PATTERN.search(token))
        
        if is_empty or is_noise:
            duration = interval.end_time - interval.start_time
            
            # Inferencia de puntuación basada en umbrales biológicos/acústicos
            if duration >= THRESHOLD_PERIOD_SEC:
                if raw_tokens and raw_tokens[-1] not in ['.', ',']:
                    raw_tokens.append('.')
                elif raw_tokens and raw_tokens[-1] == ',':
                    raw_tokens[-1] = '.'
            elif duration >= THRESHOLD_COMMA_SEC:
                if raw_tokens and raw_tokens[-1] not in ['.', ',']:
                    raw_tokens.append(',')
            continue
            
        clean_word = re.sub(r'[^a-zA-Z\']', '', token).lower()
        if not clean_word:
            continue
            
        raw_tokens.append(clean_word)
        word_mapping.append({
            'original_word': clean_word,
            'start_time': interval.start_time,
            'end_time': interval.end_time
        })
        
    reconstructed_text = " ".join(raw_tokens)
    
    # Limpieza final para adherir los signos de puntuación a la palabra anterior
    reconstructed_text = re.sub(r'\s+([.,])', r'\1', reconstructed_text)
    
    current_search_idx = 0
    final_mapping: List[Dict[str, float]] = []
    
    # Indexación de los límites de caracteres para alinear con modelos de NLP (ej. spaCy)
    for word_info in word_mapping:
        word = str(word_info['original_word'])
        match = re.search(r'\b' + re.escape(word) + r'\b', reconstructed_text[current_search_idx:])
        
        if match:
            char_start = current_search_idx + match.start()
            char_end = current_search_idx + match.end()
            
            final_mapping.append({
                'original_word': word,
                'start_time': word_info['start_time'],
                'end_time': word_info['end_time'],
                'char_start': char_start,
                'char_end': char_end
            })
            current_search_idx = char_end
            
    df_alignment = pd.DataFrame(final_mapping)
    return reconstructed_text.strip(), df_alignment
