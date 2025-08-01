import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding  # Новый импорт
import numpy as np

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self.dense_model = None  # Для плотных векторов
        self.sparse_model = None  # Для разреженных векторов
        
    def _init_qdrant_client(self):
        """Инициализация клиента Qdrant"""
        try:
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True,
                timeout=30
            )
            client.get_collections()  # Проверка соединения
            return client
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {str(e)}")
            raise

    def _load_models(self):
        """Загрузка моделей для плотных и разреженных векторов"""
        # Плотные векторы
        if self.dense_model is None:
            self.dense_model = SentenceTransformer(
                "cointegrated/rubert-tiny2",
                device="cpu",
                trust_remote_code=True
            )
        
        # Разреженные векторы (используем fastembed)
        if self.sparse_model is None:
            try:
                self.sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")
                logger.info("Модель для разреженных векторов загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели для разреженных векторов: {str(e)}")
                raise

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
                query_vector=("dense", embedding),
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

    def sparse_vector_search(self, keywords: List[str], top_k: int = 5) -> List[dict]:
        """Поиск по разреженным векторам с использованием fastembed"""
        try:
            self._load_models()
            query_text = " ".join(keywords)
            
            # Генерация разреженного вектора запроса
            sparse_embedding = list(self.sparse_model.embed(query_text))[0]
            
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=("sparse", models.SparseVector(
                    indices=sparse_embedding.indices.tolist(),
                    values=sparse_embedding.values.tolist()
                )),
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
