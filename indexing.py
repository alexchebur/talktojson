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
        """Гибридный поиск без использования torch.no_grad()"""
        try:
            self._load_models()
        
            if isinstance(queries, str):
                queries = [queries]
            
            all_results = []
        
            for query in queries:
                # Плотный вектор (без torch контекста)
                dense_embedding = self.dense_model.encode(
                    query,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).tolist()
            
                # Разреженный вектор
                sparse_vector = self._generate_sparse_vector(query)
            
                search_params = {
                    "collection_name": QDRANT_COLLECTION,
                    "query_vector": ("dense", dense_embedding),
                    "limit": top_k,
                    "with_payload": True
                }
            
                if sparse_vector:
                    search_params["query_sparse_vector"] = ("sparse", sparse_vector)
            
                results = self.qdrant_client.search(**search_params)
            
                for res in results:
                    all_results.append({
                        "id": res.id,
                        "score": res.score,
                        "content": res.payload.get("content", ""),
                        "query": query,
                        "payload": res.payload
                    })
        
            # Дедупликация
            seen_ids = set()
            return [res for res in sorted(all_results, key=lambda x: -x['score']) 
                    if res['id'] not in seen_ids and not seen_ids.add(res['id'])][:top_k]
        
        except Exception as e:
            logger.error(f"Ошибка поиска: {str(e)}")
            return []
