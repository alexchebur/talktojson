import re
import os
import numpy as np
from difflib import SequenceMatcher
from typing import List
from utils import clean_keyword
from rank_bm25 import BM25Okapi


class DataProcessor:
    def __init__(self, index_builder):
        self.index_builder = index_builder

    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Извлечение ключевых слов из текста"""
        try:
            # Удаляем специальные символы и приводим к нижнему регистру
            text_clean = re.sub(r'[^\w\s]', '', text.lower())
        
            # Разбиваем на слова и удаляем стоп-слова
            words = [word for word in text_clean.split() 
                    if len(word) > 3 and word not in self._get_stopwords()]
        
            # Подсчитываем частоту слов
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
            # Сортируем по частоте и выбираем топ-N
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in sorted_words[:max_keywords]]
    
        except Exception as e:
            print(f"Ошибка извлечения ключевых слов: {str(e)}")
            return []

    def _get_stopwords(self) -> set:
        """Возвращает набор стоп-слов для русского языка"""
        return {
            'это', 'как', 'так', 'и', 'в', 'над', 'к', 'до', 'не', 'на', 'но', 'за', 
            'то', 'с', 'ли', 'а', 'во', 'от', 'со', 'для', 'о', 'же', 'ну', 'вы', 
            'бы', 'что', 'кто', 'он', 'она'
        }
    
    def enhance_with_qdrant_search(
        self,
        query: str,
        top_k: int = 10,
        keyword_weight: float = 0.5  # Добавляем новый параметр
    ) -> List[str]:
        try:
            # Передаем параметры в index_builder
            qdrant_results = self.index_builder.semantic_search_in_qdrant(
                query, 
                top_k=top_k,
                keyword_weight=keyword_weight
            )
            return [result["text"] for result in qdrant_results]
        except Exception as e:
            print(f"Ошибка поиска в Qdrant: {str(e)}")
            return []
    
    def build_context(self, *chunk_sources: List[str]) -> str:
        context_parts = []
        for i, chunks in enumerate(chunk_sources):
            if chunks:
                context_parts.append(f"Контекст {i+1}:\n" + "\n\n".join(chunks[:3]))
        return "\n\n".join(context_parts)
