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
        """Инициализация клиента Qdrant для версии 1.15+"""
        try:
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True,
                timeout=30
            )
            # Проверка подключения через легкий метод
            client.get_collections()
            logger.info("Успешное подключение к Qdrant 1.15+")
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
            logger.info("Dense модель успешно загружена")
        
        # Разреженные векторы
        if self.sparse_model is None:
            try:
                self.sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")
                logger.info("Sparse модель успешно загружена")
            except Exception as e:
                logger.error(f"Ошибка загрузки sparse модели: {str(e)}")
                self.sparse_model = None

    def _generate_sparse_vector(self, text: str) -> Optional[models.SparseVector]:
        """Генерация разреженного вектора для Qdrant 1.15+"""
        try:
            if not text.strip() or not self.sparse_model:
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
    def hybrid_search(self, queries: Union[str, List[str]], top_k: int = 5) -> List[dict]:
        """Гибридный поиск для Qdrant 1.15+"""
        try:
            self._load_models()
            
            if isinstance(queries, str):
                queries = [queries]
                
            all_results = []
            
            for query in queries:
                # Генерация векторов
                dense_embedding = self.dense_model.encode(
                    query,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).tolist()
                
                sparse_vector = self._generate_sparse_vector(query)
                
                # Формируем запрос для гибридного поиска
                search_params = {
                    "collection_name": QDRANT_COLLECTION,
                    "query_vector": models.NamedVector(
                        name="dense",
                        vector=dense_embedding
                    ),
                    "limit": top_k,
                    "with_payload": True
                }
                
                # Добавляем sparse вектор если доступен
                if sparse_vector:
                    search_params["sparse_vector"] = models.NamedSparseVector(
                        name="sparse",
                        vector=sparse_vector
                    )
                
                # Выполняем поиск
                results = self.qdrant_client.search(**search_params)
                
                for res in results:
                    all_results.append({
                        "id": res.id,
                        "score": res.score,
                        "content": res.payload.get("content", ""),
                        "query": query,
                        "payload": res.payload
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
