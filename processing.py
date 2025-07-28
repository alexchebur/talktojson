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
    
    def enhance_with_qdrant_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[str]:
        try:
            qdrant_results = self.index_builder.semantic_search_in_qdrant(query, top_k)
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
