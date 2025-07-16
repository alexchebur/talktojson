import os
import re
import chardet
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Tuple, Optional, List

class IndexBuilder:
    def __init__(self, chunk_size: int = 10000, chunk_overlap: int = 1000):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_index = None
        self.original_chunks = []

    def _process_text(self, text: str) -> List[str]:
        """Разделение текста на чанки с перекрытием"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _detect_file_encoding(self, file_path: str) -> str:
        """Определение кодировки файла"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
        return chardet.detect(raw_data)['encoding']

    def create_bm25_index(self, documents_dir: str = "documents") -> Tuple[Optional[BM25Okapi], List[str]]:
        """Создание BM25 индекса из документов в указанной директории"""
        try:
            if not os.path.exists(documents_dir):
                os.makedirs(documents_dir)
                return None, []

            txt_files = [f for f in os.listdir(documents_dir) if f.endswith(".txt")]
            if not txt_files:
                return None, []

            all_chunks = []
            original_texts = []

            for filename in txt_files:
                file_path = os.path.join(documents_dir, filename)
                try:
                    encoding = self._detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        text = f.read()
                    chunks = self._process_text(text)
                    all_chunks.extend(chunks)
                    original_texts.extend(chunks)
                except Exception as e:
                    print(f"Ошибка чтения {filename}: {str(e)}")
                    continue

            if not all_chunks:
                return None, []

            tokenized_chunks = [doc.split() for doc in all_chunks]
            self.bm25_index = BM25Okapi(tokenized_chunks, k1=1.8, b=0.75)
            self.original_chunks = original_texts

            return self.bm25_index, self.original_chunks

        except Exception as e:
            print(f"Ошибка создания индекса: {str(e)}")
            return None, []
