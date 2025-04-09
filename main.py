# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import pickle
import io
from docx import Document
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
import requests
from typing import List, Dict, Any
from pymorphy2 import MorphAnalyzer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from config import API_KEY, API_URL
except ImportError:
    st.error("Ошибка: Создайте файл config.py с переменными API_KEY и API_URL")
    API_KEY = ""
    API_URL = "https://api.vsegpt.ru/v1/chat/completions"

# Конфигурация
DATA_DIR = "data"
MAX_CONTEXT_LENGTH = 15000
MAX_ANSWER_LENGTH = 15000
TEMPERATURE = 0.2
os.makedirs(DATA_DIR, exist_ok=True)

class TextPreprocessor:
    def __init__(self):
        self.regex = re.compile(r'[^\w\s]')

    def preprocess(self, text: str) -> List[str]:
        text = self.regex.sub(' ', text.lower())
        return text.split()

class BM25SearchEngine:
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.bm25 = None
        self.chunks_info = []
        self.doc_index = defaultdict(list)
        self.is_index_loaded = False
        self.index_path = os.path.join(DATA_DIR, "bm25_index.pkl")

    def load_index(self) -> bool:
        """Загружает индекс из pickle файла"""
        try:
            if not os.path.exists(self.index_path):
                st.error(f"Файл индекса {self.index_path} не найден")
                return False

            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                
                if isinstance(data, tuple) and len(data) == 3:
                    self.bm25, self.chunks_info, self.doc_index = data
                elif isinstance(data, dict):
                    self.bm25 = data.get('bm25')
                    self.chunks_info = data.get('chunks_info', [])
                    self.doc_index = data.get('doc_index', defaultdict(list))
                else:
                    st.error("Неверный формат данных в файле индекса")
                    return False
                
                if self.bm25 is None:
                    st.error("Не удалось загрузить индекс BM25 из файла")
                    return False
                
                self.is_index_loaded = True
                return True
                
        except Exception as e:
            st.error(f"Ошибка загрузки индекса: {e}")
            return False

    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        """Выполняет поиск по индексу BM25"""
        if not self.is_index_loaded:
            if not self.load_index():
                st.error("Не удалось загрузить индекс для поиска")
                return []

        if self.bm25 is None:
            st.error("Индекс BM25 не инициализирован")
            return []

        tokens = self.preprocessor.preprocess(query)
        if not tokens:
            st.warning("Не удалось извлечь ключевые слова из запроса")
            return []

        try:
            scores = self.bm25.get_scores(tokens)
            best_indices = np.argsort(scores)[-top_n:][::-1]

            results = []
            for idx in best_indices:
                if idx < len(self.chunks_info):
                    result = {
                        'doc_id': self.chunks_info[idx].get('doc_id', f"doc_{idx}"),
                        'doc_name': self.chunks_info[idx].get('doc_name', "Без названия"),
                        'chunk_text': self.chunks_info[idx].get('chunk_text', ""),
                        'score': float(scores[idx])
                    }
                    results.append(result)
            return results
        except Exception as e:
            st.error(f"Ошибка при выполнении поиска: {e}")
            return []

# Остальной код (LLMClient, DocumentAnalyzer, main) остается без изменений
