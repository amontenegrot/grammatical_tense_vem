# src/features/extractors.py
"""Módulo de extractores de características.

Implementa las reglas de transformación para proyectar propiedades acústicas 
y lingüísticas en matrices temporales continuas a alta resolución (100 Hz).
Garantiza una arquitectura de exactamente 43 columnas (39 control + 4 de interés).
"""

import re
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import spacy
from sklearn.pipeline import Pipeline
from wordfreq import zipf_frequency

from src.config import (
    HIGH_RES_FS,
    LATENT_SEMANTIC_COMPONENTS,
    LSA_CUSTOM_STOP_WORDS,
    LSA_NOISE_PATTERN,
    THRESHOLD_PERIOD_SEC,
)


class PhonologicalExtractor:
    """Extractor del espacio de características fonético-fonológicas (14 rasgos)."""
    
    FEATURE_NAMES: List[str] = [
        'vocalic', 'consonantal', 'voiceless', 'voiced', 'bilabial', 
        'labiodental', 'dental_alveolar', 'palatal_velar', 'stop', 
        'fricative', 'affricate', 'nasal', 'liquid', 'approximant'
    ]
    
    CONSONANT_MAP: Dict[str, List[int]] = {
        'P':  [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'B':  [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'T':  [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        'D':  [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        'K':  [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        'G':  [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        'F':  [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'V':  [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        'TH': [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'DH': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'S':  [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'Z':  [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        'SH': [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        'ZH': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        'HH': [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        'CH': [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        'JH': [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        'M':  [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'N':  [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        'NG': [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        'L':  [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        'R':  [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        'Y':  [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        'W':  [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    }
    
    VOWEL_SET: Set[str] = {
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 
        'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW', 'A', 'E', 'I', 'O', 'U'
    }

    def _process_token(self, raw_token: str) -> List[int]:
        """Convierte un fonema ARPABET crudo en un vector de rasgos fonológicos."""
        clean_token = re.sub(r'[^A-Z]', '', str(raw_token).strip().upper())
        if not clean_token:
            return [0] * len(self.FEATURE_NAMES)
        if clean_token in self.CONSONANT_MAP:
            return self.CONSONANT_MAP[clean_token]
        if clean_token in self.VOWEL_SET:
            return [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return [0] * len(self.FEATURE_NAMES)

    def extract(self, intervals: List, total_duration: float) -> pd.DataFrame:
        """Proyecta los fonemas en una matriz temporal a 100 Hz."""
        total_samples = int(np.ceil(total_duration * HIGH_RES_FS))
        matrix = np.zeros((total_samples, len(self.FEATURE_NAMES)), dtype=np.int8)
        
        for interval in intervals:
            features = self._process_token(interval.text)
            if sum(features) > 0:
                start_idx = int(np.floor(interval.start_time * HIGH_RES_FS))
                end_idx = min(int(np.ceil(interval.end_time * HIGH_RES_FS)), total_samples)
                matrix[start_idx:end_idx, :] = features

        time_axis = np.arange(total_samples) / HIGH_RES_FS
        df = pd.DataFrame(matrix, columns=self.FEATURE_NAMES, index=time_axis)
        df.index.name = 'time_seconds'
        return df


class LexicalStatsExtractor:
    """Extractor del espacio de características léxico-estadísticas (4 rasgos)."""
    
    FEATURE_NAMES: List[str] = [
        'word_presence', 'lexical_frequency', 'word_length_chars', 'word_duration_secs'
    ]

    def extract(self, df_alignment: pd.DataFrame, total_duration: float) -> pd.DataFrame:
        """Proyecta métricas de uso y forma léxica en una matriz temporal."""
        total_samples = int(np.ceil(total_duration * HIGH_RES_FS))
        matrix = np.zeros((total_samples, len(self.FEATURE_NAMES)), dtype=np.float32)
        
        for _, row in df_alignment.iterrows():
            word = str(row['original_word'])
            freq_zipf = zipf_frequency(word, 'en')
            duration = row['end_time'] - row['start_time']
            
            features = [1.0, freq_zipf, float(len(word)), duration]
            
            start_idx = int(np.floor(row['start_time'] * HIGH_RES_FS))
            end_idx = min(int(np.ceil(row['end_time'] * HIGH_RES_FS)), total_samples)
            matrix[start_idx:end_idx, :] = features
            
        time_axis = np.arange(total_samples) / HIGH_RES_FS
        df = pd.DataFrame(matrix, columns=self.FEATURE_NAMES, index=time_axis)
        df.index.name = 'time_seconds'
        return df


class LexicalCategoricalExtractor:
    """Extractor del espacio léxico-categorial (8 rasgos de control de finitud)."""
    
    FEATURE_NAMES: List[str] = [
        'noun', 'adjective', 'adverb', 'pronoun', 'preposition', 
        'conjunction', 'determiner', 'non_finite_verb'
    ]

    def _classify(self, token: spacy.tokens.Token) -> List[int]:
        """Clasifica un token separando formas no finitas del espacio de interés."""
        features = [0] * len(self.FEATURE_NAMES)
        pos, tag = token.pos_, token.tag_
        
        if pos in ['NOUN', 'PROPN']: features[0] = 1
        elif pos == 'ADJ': features[1] = 1
        elif pos == 'ADV': features[2] = 1
        elif pos == 'PRON': features[3] = 1
        elif pos == 'ADP': features[4] = 1
        elif pos in ['CCONJ', 'SCONJ']: features[5] = 1
        elif pos == 'DET': features[6] = 1
        elif pos in ['VERB', 'AUX'] and tag in ['VB', 'VBG', 'VBN']: 
            features[7] = 1
            
        return features

    def extract(self, df_alignment: pd.DataFrame, doc: spacy.tokens.Doc, total_duration: float) -> pd.DataFrame:
        """Proyecta etiquetas gramaticales en la matriz temporal continua."""
        total_samples = int(np.ceil(total_duration * HIGH_RES_FS))
        matrix = np.zeros((total_samples, len(self.FEATURE_NAMES)), dtype=np.int8)
        
        char_to_token = {i: token for token in doc for i in range(token.idx, token.idx + len(token.text))}
        
        for _, row in df_alignment.iterrows():
            mid_char = (row['char_start'] + row['char_end']) // 2
            token = char_to_token.get(mid_char)
            
            if token:
                features = self._classify(token)
                if sum(features) > 0:
                    start_idx = int(np.floor(row['start_time'] * HIGH_RES_FS))
                    end_idx = min(int(np.ceil(row['end_time'] * HIGH_RES_FS)), total_samples)
                    matrix[start_idx:end_idx, :] = features
                    
        time_axis = np.arange(total_samples) / HIGH_RES_FS
        df = pd.DataFrame(matrix, columns=self.FEATURE_NAMES, index=time_axis)
        df.index.name = 'time_seconds'
        return df


class SyntacticExtractor:
    """Extractor del espacio sintáctico (3 rasgos: distancia, profundidad y raíz)."""
    
    FEATURE_NAMES: List[str] = ['dependency_distance', 'syntactic_depth', 'is_root']

    def _classify(self, token: spacy.tokens.Token) -> List[int]:
        """Calcula métricas de complejidad estructural sintáctica para un token."""
        is_root = 1 if token.dep_ == "ROOT" else 0
        dep_distance = int(abs(token.i - token.head.i))
        depth = int(sum(1 for _ in token.ancestors))
        return [dep_distance, depth, is_root]

    def extract(self, df_alignment: pd.DataFrame, doc: spacy.tokens.Doc, total_duration: float) -> pd.DataFrame:
        """Proyecta la complejidad clausal estructural en el tiempo."""
        total_samples = int(np.ceil(total_duration * HIGH_RES_FS))
        matrix = np.zeros((total_samples, len(self.FEATURE_NAMES)), dtype=np.int16)
        
        char_to_token = {i: token for token in doc for i in range(token.idx, token.idx + len(token.text))}
        
        for _, row in df_alignment.iterrows():
            mid_char = (row['char_start'] + row['char_end']) // 2
            token = char_to_token.get(mid_char)
            
            if token:
                features = self._classify(token)
                start_idx = int(np.floor(row['start_time'] * HIGH_RES_FS))
                end_idx = min(int(np.ceil(row['end_time'] * HIGH_RES_FS)), total_samples)
                matrix[start_idx:end_idx, :] = features
                
        time_axis = np.arange(total_samples) / HIGH_RES_FS
        df = pd.DataFrame(matrix, columns=self.FEATURE_NAMES, index=time_axis)
        df.index.name = 'time_seconds'
        return df


class FiniteTenseExtractor:
    """Extractor del Espacio de Interés: Tiempo Gramatical Finito (4 rasgos exactos)."""
    
    FEATURE_NAMES: List[str] = [
        'past_regular', 'past_irregular', 'non_past_marked_3sg', 'non_past_unmarked'
    ]
    AMBIGUOUS_CONTRACTIONS: Set[str] = {"'d"}

    def _classify(self, token: spacy.tokens.Token) -> List[int]:
        """Clasifica formas verbales finitas en sus 4 realizaciones morfológicas.
        
        Excluye explícitamente modales (MD) y la contracción ambigua 'd para
        preservar la pureza conceptual del contraste temporal finito.
        """
        features = [0, 0, 0, 0]
        pos, tag, text = token.pos_, token.tag_, token.text.lower()
        
        if tag == 'MD' or text in self.AMBIGUOUS_CONTRACTIONS or pos not in ['VERB', 'AUX']:
            return features
            
        if tag == 'VBD':
            if text.endswith('ed'): 
                features[0] = 1  # past_regular
            else: 
                features[1] = 1  # past_irregular
        elif tag == 'VBZ':
            features[2] = 1      # non_past_marked_3sg
        elif tag == 'VBP':
            features[3] = 1      # non_past_unmarked
            
        return features

    def extract(self, df_alignment: pd.DataFrame, doc: spacy.tokens.Doc, total_duration: float) -> pd.DataFrame:
        """Proyecta el espacio de interés en una matriz de exactamente 4 columnas.
        
        Decisión metodológica: La matriz predictora contiene ÚNICAMENTE las 4
        características elementales. Las categorías agregadas (past_total y 
        non_past_total) se generan exclusivamente para reportes de auditoría.
        """
        total_samples = int(np.ceil(total_duration * HIGH_RES_FS))
        matrix = np.zeros((total_samples, len(self.FEATURE_NAMES)), dtype=np.int8)
        
        char_to_token = {i: token for token in doc for i in range(token.idx, token.idx + len(token.text))}
        
        for _, row in df_alignment.iterrows():
            mid_char = (row['char_start'] + row['char_end']) // 2
            token = char_to_token.get(mid_char)
            
            if token:
                features = self._classify(token)
                if sum(features) > 0:
                    start_idx = int(np.floor(row['start_time'] * HIGH_RES_FS))
                    end_idx = min(int(np.ceil(row['end_time'] * HIGH_RES_FS)), total_samples)
                    matrix[start_idx:end_idx, :] = features
                    
        time_axis = np.arange(total_samples) / HIGH_RES_FS
        df = pd.DataFrame(matrix, columns=self.FEATURE_NAMES, index=time_axis)
        df.index.name = 'time_seconds'
        return df


class SemanticLSAExtractor:
    """Extractor del espacio léxico-semántico latente (10 componentes)."""

    def __init__(self, semantic_model: Pipeline, nlp_model: spacy.language.Language):
        """Inicializa el extractor con el modelo LSA entrenado en Train."""
        self.semantic_model = semantic_model
        self.nlp_model = nlp_model
        self.feature_names = [f"semantic_dim_{i}" for i in range(LATENT_SEMANTIC_COMPONENTS)]
        self.stop_words = LSA_CUSTOM_STOP_WORDS
        self.noise_pattern = LSA_NOISE_PATTERN

    def extract(self, intervals: List, total_duration: float) -> pd.DataFrame:
        """Proyecta las dimensiones semánticas continuas sobre las cláusulas."""
        total_samples = int(np.ceil(total_duration * HIGH_RES_FS))
        matrix = np.zeros((total_samples, len(self.feature_names)), dtype=np.float32)
        
        current_sentence_words = []
        sentence_start_time = None
        sentence_end_time = None
        
        for interval in intervals:
            token = interval.text.strip()
            is_empty = (token == "")
            is_noise = bool(self.noise_pattern.search(token))
            
            if is_empty or is_noise:
                duration = interval.end_time - interval.start_time
                if duration >= THRESHOLD_PERIOD_SEC and current_sentence_words:
                    self._process_and_project_chunk(
                        current_sentence_words, sentence_start_time, sentence_end_time,
                        matrix, total_samples
                    )
                    current_sentence_words = []
                    sentence_start_time = None
                continue
                
            clean_word = re.sub(r'[^a-zA-Z\']', '', token).lower()
            if clean_word:
                if not current_sentence_words:
                    sentence_start_time = interval.start_time
                sentence_end_time = interval.end_time
                current_sentence_words.append(clean_word)
                
        # Procesar el último fragmento restante
        if current_sentence_words and sentence_start_time is not None:
            self._process_and_project_chunk(
                current_sentence_words, sentence_start_time, sentence_end_time,
                matrix, total_samples
            )
            
        time_axis = np.arange(total_samples) / HIGH_RES_FS
        df = pd.DataFrame(matrix, columns=self.feature_names, index=time_axis)
        df.index.name = 'time_seconds'
        return df

    def _process_and_project_chunk(
        self, words: List[str], start_time: float, end_time: float, 
        matrix: np.ndarray, total_samples: int
    ) -> None:
        """Lematiza una cláusula y proyecta su vector LSA en la matriz (in-place)."""
        raw_sentence = " ".join(words)
        doc = self.nlp_model(raw_sentence)
        lemmas = [
            t.lemma_.lower() for t in doc 
            if not t.is_stop and not t.is_punct and not t.like_num and t.lemma_.strip()
        ]
        
        final_lemmas = [w for w in lemmas if w not in self.stop_words]
        lemmatized_sentence = " ".join(final_lemmas)
        
        if lemmatized_sentence.strip():
            vector_nd = self.semantic_model.transform([lemmatized_sentence])[0]
            start_idx = int(np.floor(start_time * HIGH_RES_FS))
            end_idx = min(int(np.ceil(end_time * HIGH_RES_FS)), total_samples)
            matrix[start_idx:end_idx, :] = vector_nd
