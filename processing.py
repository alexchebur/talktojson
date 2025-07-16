import re
from difflib import SequenceMatcher
from typing import List
from .indexing import IndexBuilder
from .utils import clean_keyword

class DataProcessor:
    def __init__(self, index_builder: IndexBuilder):
        self.index_builder = index_builder
    
    def process_text(self, text: str) -> List[str]:
        """Разделение текста на чанки"""
        # [Реализация как в оригинале]
    
    def extract_keywords(self, text: str, bm25: BM25Okapi) -> List[str]:
        """Извлечение ключевых слов"""
        # [Реализация как в оригинале]
    
    def search_relevant_chunks(self, bm25: BM25Okapi, original_chunks: List[str], keywords: List[str]) -> List[str]:
        """Поиск релевантных фрагментов"""
        # [Реализация как в оригинале]
    
    def get_unique_chunks(self, main_chunks: List[str], new_chunks: List[str]) -> List[str]:
        """Фильтрация дублирующихся фрагментов"""
        # [Реализация как в оригинале]
    
    def build_context(self, *chunk_sources: List[str]) -> str:
        """Построение контекста из различных источников"""
        context_parts = []
        for i, chunks in enumerate(chunk_sources):
            if chunks:
                context_parts.append(f"Контекст {i+1}:\n" + "\n\n".join(chunks[:3]))
        return "\n\n".join(context_parts)
