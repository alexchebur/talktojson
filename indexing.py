import os
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Optional
from .config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.bm25_index = None
        self.original_chunks = []
        
    def create_bm25_index():
        """Создание BM25 индекса на основе документов в папке"""
        all_chunks = []
        original_texts = []
    
        try:
            if not os.path.exists("documents"):
                os.makedirs("documents")

            txt_files = [f for f in os.listdir("documents") if f.endswith(".txt")]
            if not txt_files:
                return None, None

            for filename in txt_files:
                file_path = os.path.join("documents", filename)
                try:
                    encoding = detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        text = f.read()
                    chunks = process_text(text)
                    all_chunks.extend(chunks)
                    original_texts.extend(chunks)
                except Exception as e:
                    st.error(f"Ошибка чтения {filename}: {str(e)}")
                    continue

            if not all_chunks:
                return None, None

            tokenized_chunks = [doc.split() for doc in all_chunks]
            return BM25Okapi(tokenized_chunks, k1=1.8, b=0.75), original_texts

        except Exception as e:
            st.error(f"Ошибка создания индекса: {str(e)}")
            return None, None

    
    def create_vector_index(self, embeddings: List[List[float]]) -> Any:
        """Создание векторного индекса (заглушка для будущей реализации)"""
        # TODO: Реализация с FAISS или аналогичной библиотекой
        pass
    
    def update_index(self, new_documents: List[str]):
        """Обновление индекса новыми документами"""
        # TODO: Реализация инкрементального обновления
        pass
