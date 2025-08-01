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
from fastembed import SparseTextEmbedding
import numpy as np

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self.dense_model = None  # Для плотных векторов (sentence-transformers)
        self.sparse_model = None  # Для разреженных векторов (fastembed)
        self.SPARSE_MODEL_TYPE = "fastembed"  # Тип модели для разреженных векторов
        
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
        if self.sparse_model is None and self.SPARSE_MODEL_TYPE == "fastembed":
            try:
                self.sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")
                logger.info("Модель для разреженных векторов fastembed загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки sparse модели: {str(e)}")
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

    def sparse_vector_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Поиск по разреженным векторам (аналог из Colab)"""
        try:
            self._load_models()
            
            # Генерация разреженного вектора
            if self.SPARSE_MODEL_TYPE == "fastembed":
                sparse_query = list(self.sparse_model.embed(query))[0]
                indices = sparse_query.indices.tolist()
                values = sparse_query.values.tolist()
            else:
                # Fallback для FastText (если потребуется)
                words = query.lower().split()
                word_counts = {}
                
                for word in words:
                    if len(word) > 2:
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                indices = []
                values = []
                total_words = len(words)
                for word, count in word_counts.items():
                    try:
                        tf = count / total_words
                        index = hash(word) % 100000
                        indices.append(index)
                        values.append(tf)
                    except:
                        pass
            
            # Формируем запрос аналогично Colab-скрипту
            results = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    models.Prefetch(
                        query=models.NamedVector(
                            name="sparse",
                            vector=models.SparseVector(
                                indices=indices,
                                values=values
                            )
                        ),
                        using="sparse",
                        limit=top_k*2
                    )
                ],
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
            logger.error(f"Ошибка sparse поиска: {str(e)}", exc_info=True)
            return []

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[dict]:
        """Гибридный поиск как в Colab"""
        try:
            self._load_models()
            
            # Плотный вектор
            dense_query = self.dense_model.encode(
                query,
                convert_to_numpy=True
            ).tolist()
            
            # Разреженный вектор
            if self.SPARSE_MODEL_TYPE == "fastembed":
                sparse_query = list(self.sparse_model.embed(query))[0]
                indices = sparse_query.indices.tolist()
                values = sparse_query.values.tolist()
            else:
                indices, values = [], []
            
            # Формируем запрос аналогично Colab
            results = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using="dense",
                        limit=top_k*2
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=indices,
                            values=values
                        ),
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
