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
    def semantic_search(self, queries: List[str], top_k: int = 5) -> List[Dict]:
        """Семантический поиск по списку запросов (исходный + сгенерированные)"""
        try:
            self._load_models()
            all_results = []
        
            for query in queries:
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
            
                for res in results:
                    all_results.append({
                        "id": res.id,
                        "score": res.score,
                        "content": res.payload.get("content", ""),
                        "query": query,  # Сохраняем запрос, который дал этот результат
                        "payload": res.payload
                    })
        
            # Удаляем дубликаты и сортируем по score
            unique_results = {res['id']: res for res in all_results}.values()
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {str(e)}")
            return []

    def sparse_vector_search(self, queries: List[str], top_k: int = 5) -> List[dict]:
        """Оптимизированный поиск по разреженным векторам"""
        try:
            self._load_models()
            all_results = []
        
            # Ограничиваем количество параллельных запросов
            max_parallel_queries = 3
            query_batches = [queries[i:i+max_parallel_queries] for i in range(0, len(queries), max_parallel_queries)]
        
            for batch in query_batches:
                batch_vectors = []
            
                # Генерация векторов для батча
                for query in batch:
                    try:
                        embedding = list(self.sparse_model.embed(query))[0]
                        batch_vectors.append({
                            "query": query,
                            "vector": {
                                "indices": embedding.indices.tolist(),
                                "values": [float(v) for v in embedding.values.tolist()]
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Ошибка генерации вектора для запроса '{query[:50]}...': {str(e)}")
                        continue
            
                # Пакетный поиск
                try:
                    search_requests = [
                        models.SearchRequest(
                            vector=models.NamedVector(
                                name="sparse",
                                vector=v["vector"]
                            ),
                            limit=top_k,
                            with_payload=True
                        ) for v in batch_vectors
                    ]
                
                    batch_results = self.qdrant_client.search_batch(
                        collection_name=QDRANT_COLLECTION,
                        requests=search_requests
                    )
                
                    # Обработка результатов
                    for query, results in zip(batch_vectors, batch_results):
                        for res in results:
                            all_results.append({
                                "id": res.id,
                                "score": float(res.score),
                                "content": res.payload.get("content", ""),
                                "query": query["query"],
                                "payload": res.payload
                            })
                        
                except Exception as e:
                    logger.error(f"Ошибка пакетного поиска: {str(e)}")
                    continue
        
            # Дедупликация и сортировка
            unique_results = {}
            for res in all_results:
                if res['id'] not in unique_results or res['score'] > unique_results[res['id']]['score']:
                    unique_results[res['id']] = res
                
            return sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
        
        except Exception as e:
            logger.error(f"Критическая ошибка sparse поиска: {str(e)}", exc_info=True)
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
                "content": point.payload.get("content", "")
            } for point in results]
        except Exception as e:
            logger.error(f"Ошибка гибридного поиска: {str(e)}")
            return []
