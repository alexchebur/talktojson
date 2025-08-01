import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Tuple
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import fasttext  # Добавляем FastText
import numpy as np
import re

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self.dense_model = None  # Для плотных векторов (sentence-transformers)
        self.sparse_model = None  # Для разреженных векторов (FastText)
        
    def _init_qdrant_client(self):
        """Инициализация клиента Qdrant с обработкой ошибок"""
        try:
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True,
                timeout=30
            )
            # Проверка соединения
            client.get_collections()
            return client
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {str(e)}")
            raise

    def _load_models(self):
        """Загрузка моделей для плотных и разреженных векторов"""
        # Загрузка модели для плотных векторов
        if self.dense_model is None:
            self.dense_model = SentenceTransformer(
                "cointegrated/rubert-tiny2",
                device="cpu",
                trust_remote_code=True
            )
        
        # Загрузка модели для разреженных векторов (FastText)
        if self.sparse_model is None:
            try:
                # Пытаемся загрузить предобученную модель
                self.sparse_model = fasttext.load_model("cc.ru.300.bin")
                logger.info("Загружена предобученная модель FastText")
            except:
                # Если не удалось, создаем простую модель на лету
                logger.warning("Не найдена предобученная модель FastText, создаем временную")
                self.sparse_model = fasttext.train_unsupervised(
                    input="",  # Пустой ввод для простой модели
                    model='skipgram',
                    dim=100
                )

    def _generate_sparse_vector(self, text: str) -> Tuple[List[int], List[float]]:
        """Генерация разреженного вектора с помощью FastText"""
        self._load_models()
        
        # Очистка текста и токенизация
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Фильтрация коротких слов
        words = [word for word in words if len(word) > 2]
        
        # Создание разреженного вектора
        indices = []
        values = []
        total_words = len(words)
        
        for word in set(words):  # Уникальные слова
            try:
                # Получаем эмбеддинг слова
                word_emb = self.sparse_model.get_word_vector(word)
                # Нормализуем и используем первый элемент как вес
                norm = np.linalg.norm(word_emb)
                weight = norm if norm > 0 else 1.0
                
                # Хешируем слово для получения индекса
                index = hash(word) % 100000  # Фиксированный размер вектора
                indices.append(index)
                values.append(weight)
            except:
                continue
        
        return indices, values

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Семантический поиск по плотным векторам"""
        try:
            self._load_models()
            embedding = self.dense_model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).tolist()
            
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding,
                limit=top_k,
                with_payload=True
            )
            
            return [{
                "id": res.id,
                "score": res.score,
                "payload": res.payload,
                "text": res.payload.get("text", "")
            } for res in results]
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {str(e)}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[dict]:
        """Поиск по разреженным векторам FastText"""
        try:
            self._load_models()
            # Объединяем ключевые слова в один запрос
            query_text = " ".join(keywords)
            
            # Генерируем разреженный вектор
            indices, values = self._generate_sparse_vector(query_text)
            
            if not indices:
                return []
                
            # Выполняем поиск в Qdrant
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=models.NamedVector(
                    name="sparse",  # Имя поля с разреженными векторами
                    vector=models.SparseVector(
                        indices=indices,
                        values=values
                    )
                ),
                limit=top_k,
                with_payload=True
            )
            
            return [{
                "id": res.id,
                "score": res.score,
                "payload": res.payload,
                "text": res.payload.get("text", "")
            } for res in results]
        except Exception as e:
            logger.error(f"Ошибка поиска по разреженным векторам: {str(e)}")
            return []

    # Удаляем гибридный поиск и расширение запроса, так как они больше не нужны
    # (Оригинальные методы hybrid_search и expand_query удалены)
