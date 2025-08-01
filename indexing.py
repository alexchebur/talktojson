import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional, Union
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
        """Генерация разреженного вектора"""
        try:
            if not text.strip():
                return None
                
            embeddings = list(self.sparse_model.embed(text))
            if not embeddings:
                return None
                
            sparse_embedding = embeddings[0]
            return models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
        except Exception as e:
            logger.error(f"Ошибка генерации sparse вектора: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Семантический поиск через search (не через query_points)"""
        try:
            self._load_models()
            dense_query = self.dense_model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).tolist()
            
            # Используем обычный search вместо query_points
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=("dense", dense_query),
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

    def sparse_vector_search(self, query: Union[str, List[str]], top_k: int = 5) -> List[dict]:
        """Альтернативная реализация через query_points"""
        try:
            self._load_models()
            query_text = " ".join(query) if isinstance(query, list) else query
        
            embeddings = list(self.sparse_model.embed(query_text))
            if not embeddings:
                return []
            
            sparse_embedding = embeddings[0]
            sparse_vector = {
                "indices": sparse_embedding.indices.tolist(),
                "values": [float(v) for v in sparse_embedding.values.tolist()]
            }
        
            # Используем query_points вместо search
            results = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=models.SearchRequest(
                    vector=models.NamedVector(
                        name="sparse",
                        vector=models.SparseVector(**sparse_vector)
                    ),
                    limit=top_k,
                    with_payload=True
                )
            )
        
            return [{
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
                "text": point.payload.get("text", "")
            } for point in results]
        
        except Exception as e:
            logger.error(f"Ошибка sparse поиска (alt): {str(e)}")
            return []

    def hybrid_search(self, query: Union[str, List[str]], top_k: int = 5, alpha: float = 0.5) -> List[dict]:
        """Гибридный поиск через query_points с использованием search_queries"""
        try:
            self._load_models()
            query_text = " ".join(query) if isinstance(query, list) else query
            
            # Плотный вектор
            dense_query = self.dense_model.encode(
                query_text,
                convert_to_numpy=True
            ).tolist()
            
            # Разреженный вектор
            sparse_vector = self._generate_sparse_vector(query_text)
            if sparse_vector is None:
                return self.semantic_search(query_text, top_k)
            
            # Формируем запросы для prefetch
            dense_query_request = models.SearchRequest(
                vector=models.NamedVector(
                    name="dense",
                    vector=dense_query
                ),
                limit=top_k*2,
                with_payload=True
            )
            
            sparse_query_request = models.SearchRequest(
                vector=models.NamedVector(
                    name="sparse",
                    vector=sparse_vector
                ),
                limit=top_k*2,
                with_payload=True
            )
            
            # Выполняем гибридный поиск через query_points
            results = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                queries=[
                    dense_query_request,
                    sparse_query_request
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
            } for point in results]
        except Exception as e:
            logger.error(f"Ошибка гибридного поиска: {str(e)}")
            return []
