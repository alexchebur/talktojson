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
import torch

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self.dense_model = None
        self.sparse_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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
                device=self.device,
                trust_remote_code=True
            )
            self.dense_model.to(self.device)
        
        # Разреженные векторы
        if self.sparse_model is None:
            try:
                self.sparse_model = SparseTextEmbedding(
                    "Qdrant/bm42-all-minilm-l6-v2-attentions",
                    device=self.device
                )
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
                indices=sparse_embedding.indices.cpu().numpy().tolist(),
                values=sparse_embedding.values.cpu().numpy().tolist()
            )
        except Exception as e:
            logger.error(f"Ошибка генерации sparse вектора: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def hybrid_search(self, queries: Union[str, List[str]], top_k: int = 5) -> List[dict]:
        """Унифицированный гибридный поиск (плотные + разреженные векторы)"""
        if isinstance(queries, str):
            queries = [queries]
            
        try:
            self._load_models()
            all_results = []
            
            for query in queries:
                # Плотный вектор
                with torch.no_grad():
                    dense_embedding = self.dense_model.encode(
                        query,
                        normalize_embeddings=True,
                        convert_to_tensor=True
                    ).cpu().numpy().tolist()
                
                # Разреженный вектор
                sparse_vector = self._generate_sparse_vector(query)
                
                # Формируем запрос
                search_params = {
                    "collection_name": QDRANT_COLLECTION,
                    "query_vector": ("dense", dense_embedding),
                    "limit": top_k,
                    "with_payload": True
                }
                
                if sparse_vector:
                    search_params["query_sparse_vector"] = ("sparse", sparse_vector)
                
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
            
            # Удаляем дубликаты и сортируем
            unique_results = {res['id']: res for res in all_results}.values()
            return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
            
        except Exception as e:
            logger.error(f"Ошибка гибридного поиска: {str(e)}")
            return []
