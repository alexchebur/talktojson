import os
import torch
from typing import List, Dict, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self.device = self._get_device()
        self.dense_model = self._init_dense_model()
        self.sparse_model = self._init_sparse_model()

    def _get_device(self):
        """Определяем доступное устройство"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _init_dense_model(self):
        """Инициализация модели для плотных векторов"""
        try:
            model = SentenceTransformer(
                "cointegrated/rubert-tiny2",
                device=self.device,
                trust_remote_code=True
            )
            # Явная инициализация весов
            if next(model.parameters()).is_meta:
                model.to_empty(device=self.device)
            else:
                model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки dense модели: {str(e)}")
            raise

    def _init_sparse_model(self):
        """Инициализация модели для разреженных векторов"""
        try:
            model = SparseTextEmbedding(
                model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
                device=self.device
            )
            return model
        except Exception as e:
            logger.warning(f"Ошибка загрузки sparse модели: {str(e)}")
            return None

    def _init_qdrant_client(self):
        """Инициализация клиента Qdrant"""
        try:
            return QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                prefer_grpc=True,
                timeout=30
            )
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {str(e)}")
            raise

    def _generate_dense_embedding(self, text: str) -> List[float]:
        """Генерация плотного вектора"""
        with torch.no_grad():
            return self.dense_model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True
            ).tolist()

    def _generate_sparse_vector(self, text: str) -> Optional[models.SparseVector]:
        """Генерация разреженного вектора"""
        if not self.sparse_model or not text.strip():
            return None
            
        try:
            embeddings = next(self.sparse_model.embed(text))
            return models.SparseVector(
                indices=embeddings.indices.cpu().numpy().tolist(),
                values=embeddings.values.cpu().numpy().tolist()
            )
        except Exception as e:
            logger.warning(f"Ошибка генерации sparse вектора: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def hybrid_search(self, queries: Union[str, List[str]], top_k: int = 5) -> List[Dict]:
        """Гибридный поиск с обработкой мета-тензоров"""
        if isinstance(queries, str):
            queries = [queries]
            
        all_results = []
        
        for query in queries:
            try:
                # Генерация векторов
                dense_vec = self._generate_dense_embedding(query)
                sparse_vec = self._generate_sparse_vector(query)
                
                # Параметры поиска
                search_params = {
                    "collection_name": os.getenv("QDRANT_COLLECTION"),
                    "query_vector": ("dense", dense_vec),
                    "limit": top_k,
                    "with_payload": True
                }
                
                if sparse_vec:
                    search_params["query_sparse_vector"] = ("sparse", sparse_vec)
                
                # Выполнение поиска
                results = self.qdrant_client.search(**search_params)
                
                for res in results:
                    all_results.append({
                        "id": res.id,
                        "score": res.score,
                        "content": res.payload.get("content", ""),
                        "query": query,
                        "payload": res.payload
                    })
                    
            except Exception as e:
                logger.error(f"Ошибка обработки запроса '{query}': {str(e)}")
                continue
        
        # Дедупликация и сортировка результатов
        unique_results = {}
        for res in all_results:
            if res['id'] not in unique_results or res['score'] > unique_results[res['id']]['score']:
                unique_results[res['id']] = res
                
        return sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
