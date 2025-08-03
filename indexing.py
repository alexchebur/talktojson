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
        """Поиск по разреженным векторам по списку запросов"""
        try:
            self._load_models()
            all_results = []
        
            for query in queries:
                sparse_embedding = list(self.sparse_model.embed(query))[0]
                sparse_vector = {
                    "indices": sparse_embedding.indices.tolist(),
                    "values": sparse_embedding.values.tolist()
               }
            
                results = self.qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=("sparse", sparse_vector),
                    limit=top_k,
                    with_payload=True
                )
            
                for res in results:
                    all_results.append({
                        "id": res.id,
                        "score": res.score,
                        "content": res.payload.get("content", ""),
                        "query": query,  # Сохраняем исходный запрос
                        "payload": res.payload
                    })
        
            # Удаляем дубликаты и сортируем
            unique_results = {res['id']: res for res in all_results}.values()
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        except Exception as e:
            logger.error(f"Ошибка sparse поиска: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def hybrid_search(self, queries: Union[str, List[str]], top_k: int = 5) -> List[dict]:
        """Гибридный поиск для списка запросов"""
        try:
            self._load_models()
        
            # Если пришел единичный запрос, преобразуем в список
            if isinstance(queries, str):
                queries = [queries]
            
            all_results = []
        
            for query in queries:
                # Плотный вектор
                dense_query = self.dense_model.encode(
                    query,
                    convert_to_numpy=True
                ).tolist()
            
                # Разреженный вектор
                sparse_vector = self._generate_sparse_vector(query)
            
                # Если sparse_vector не сгенерирован, используем только dense
                if sparse_vector is None:
                    results = self.qdrant_client.search(
                        collection_name=QDRANT_COLLECTION,
                        query_vector=("dense", dense_query),
                        limit=top_k,
                        with_payload=True
                    )
                
                    for res in results:
                        all_results.append({
                            "id": res.id,
                            "score": res.score,
                            "content": res.payload.get("content", ""),
                            "query": query,
                            "payload": res.payload
                        })
                    continue
            
                # Формируем запросы для гибридного поиска
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
            
                # Выполняем гибридный поиск
                results = self.qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    queries=[dense_query_request, sparse_query_request],
                    search_params=models.SearchParams(
                        fusion=models.Fusion.DBSF,
                        alpha=0.5  # Значение по умолчанию
                    ),
                    limit=top_k,
                    with_payload=True
                )
            
                for point in results:
                    all_results.append({
                        "id": point.id,
                        "score": point.score,
                        "query": query,
                        "payload": point.payload,
                        "content": point.payload.get("content", "")
                    })
        
            # Дедупликация результатов
            seen_ids = set()
            unique_results = []
            for res in sorted(all_results, key=lambda x: x['score'], reverse=True):
                if res['id'] not in seen_ids:
                    seen_ids.add(res['id'])
                    unique_results.append(res)
                    if len(unique_results) >= top_k:
                        break
                    
            return unique_results
        
    except Exception as e:
        logger.error(f"Ошибка гибридного поиска: {str(e)}")
        return []
