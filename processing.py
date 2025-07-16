import re
import numpy as np  # Добавлено: для работы с массивами
from difflib import SequenceMatcher
from typing import List, Dict, Any
from indexing import IndexBuilder
from utils import clean_keyword
from rank_bm25 import BM25Okapi  # Добавлено: если используется BM25

# Предполагается, что эти константы определены где-то в вашем проекте
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Имитация Streamlit для примера
import streamlit as st  # Добавлено: если используете Streamlit


class DataProcessor:
    def __init__(self, index_builder: IndexBuilder):
        self.index_builder = index_builder

    def process_text(self, text: str) -> List[str]:
        """Разделение текста на чанки с перекрытием"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def extract_keywords(self, text: str, bm25: BM25Okapi) -> List[str]:
        """Извлечение ключевых слов с учетом BM25"""
        try:
            words = re.findall(r'\b[а-яё]+\b', text.lower())
            stop_words = {"на", "под", "в", "среди", "перед", "затем", "после", "до", "сразу"}

            filtered = [
                word for word in words
                if len(word) >= 5
                and word not in stop_words
                and not re.search(r'\d', word)
            ]

            scores = bm25.get_scores(filtered)
            scored_words = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)

            unique_words = []
            seen = set()
            for word, _ in scored_words:
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)
                    if len(unique_words) == 20:
                        break

            return [clean_keyword(word) for word in unique_words]

        except Exception as e:
            st.error(f"Ошибка извлечения ключевых слов: {str(e)}")
            return []

    def search_relevant_chunks(self, bm25: BM25Okapi, original_chunks: List[str], keywords: List[str]) -> List[str]:
        """Поиск релевантных фрагментов"""
        try:
            query_weights = {term: 2 for term in keywords}
            weighted_query = []
            for term, weight in query_weights.items():
                weighted_query.extend([term] * weight)

            doc_scores = np.array(bm25.get_scores(weighted_query))
            sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
            return [original_chunks[i] for i in sorted_indices if doc_scores[i] > 0.0][:5]

        except Exception as e:
            st.error(f"Ошибка поиска: {str(e)}")
            return []

    def get_unique_chunks(self, main_chunks: List[str], new_chunks: List[str]) -> List[str]:
        """Фильтрация дублирующихся фрагментов с порогом схожести"""
        unique_chunks = []
        for new_chunk in new_chunks:
            is_duplicate = False
            for main_chunk in main_chunks:
                # Считаем дубликатом если >80% содержимого совпадает
                if SequenceMatcher(None, main_chunk[:1000], new_chunk[:1000]).ratio() > 0.8:
                    is_duplicate = True
                    break
            # ✅ Отступ исправлен: был неправильный, теперь корректный
            if not is_duplicate and new_chunk not in unique_chunks:
                unique_chunks.append(new_chunk)
        return unique_chunks

    def build_context(self, *chunk_sources: List[str]) -> str:
        """Построение контекста из различных источников"""
        context_parts = []
        for i, chunks in enumerate(chunk_sources):
            if chunks:
                context_parts.append(f"Контекст {i+1}:\n" + "\n\n".join(chunks[:3]))
        return "\n\n".join(context_parts)
