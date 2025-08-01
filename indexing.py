import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import numpy as np

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self.dense_model = None
        self.sparse_model = None
        
    def _init_qdrant_client(self):
        """Инициализация клиента Qdrant"""
        try:
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True,
                timeout=30
            )
            client.get_collections()
            return client
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {str(e)}")
            raise

    def _load_models(self):
        """Загрузка моделей для векторизации"""
        # Плотные векторы
        if self.dense_model is None:
            self.dense_model = SentenceTransformer(
                "cointegrated/rubert-tiny2",
                device="cpu",
                trust_remote_code=True
            )
        
        # Разреженные векторы
        if self.sparse_model is None:
            try:
                self.sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")
                logger.info("Модель для разреженных векторов успешно загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки sparse модели: {str(e)}")
                raise

    def _generate_sparse_vector(self, text: str) -> Optional[models.SparseVector]:
        """Генерация разреженного вектора с защитой от пустых результатов"""
        try:
            if not text.strip():
                return None
                
            # Получаем эмбеддинги (может вернуть несколько результатов для больших текстов)
            embeddings = list(self.sparse_model.embed(text))
            
            if not embeddings:
                logger.warning(f"Не удалось сгенерировать эмбеддинг для текста: {text[:50]}...")
                return None
                
            # Берем первый эмбеддинг (для коротких запросов будет только один)
            sparse_embedding = embeddings[0]
            
            # Преобразуем в формат Qdrant
            return models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
            
        except Exception as e:
            logger.error(f"Ошибка генерации sparse вектора: {str(e)}")
            return None

    def sparse_vector_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Поиск по разреженным векторам с защитой от ошибок"""
        try:
            self._load_models()
            
            # Генерируем разреженный вектор
            sparse_vector = self._generate_sparse_vector(query)
            if sparse_vector is None:
                logger.warning("Не удалось сгенерировать sparse вектор для поиска")
                return []
            
            # Выполняем поиск
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=("sparse", sparse_vector),
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
            logger.error(f"Ошибка sparse поиска: {str(e)}", exc_info=True)
            return []

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

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[dict]:
        """Гибридный поиск с защитой от ошибок"""
        try:
            self._load_models()
            
            # Плотный вектор
            dense_embedding = self.dense_model.encode(
                query,
                convert_to_numpy=True
            ).tolist()
            
            # Разреженный вектор
            sparse_vector = self._generate_sparse_vector(query)
            if sparse_vector is None:
                logger.warning("Используем только плотные векторы (не удалось сгенерировать sparse)")
                return self.semantic_search(query, top_k)
            
            # Выполняем гибридный поиск
            results = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    models.Prefetch(
                        query=dense_embedding,
                        using="dense",
                        limit=top_k*2
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=top_k*2
                    )
                ],
                search_params=models.SearchParams(
                    fusion=models.Fusion.DBSF,
                    alpha=alpha
                ),
                limit=top_k,
                with_payload=True
            )
            
            return [{
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
                "text": point.payload.get("text", "")
            } for point in results.points]
            
        except Exception as e:
            logger.error(f"Ошибка гибридного поиска: {str(e)}", exc_info=True)
            return []
