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
        
    def _init_qdrant_client(self) -> QdrantClient:
        """Надежная инициализация клиента Qdrant"""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")
    
        # Логирование для отладки
        logger.info(f"Подключение к Qdrant по адресу: {qdrant_url}")
    
        # Проверка и корректировка URL
        if "://" not in qdrant_url:
            qdrant_url = "http://" + qdrant_url
            logger.warning(f"Добавлен протокол по умолчанию: {qdrant_url}")
    
        try:
            # Пробуем разные варианты подключения
            if qdrant_url.startswith("https://"):
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=api_key,
                    port=443,  # Стандартный порт для HTTPS
                    prefer_grpc=False,
                    timeout=15
                )
            else:
                # Для HTTP используем порт 6333
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=api_key,
                    port=6333,
                    prefer_grpc=False,
                    timeout=15
                )
        
            # Упрощенная проверка подключения
            client._client.openapi_client.models_api.models_get()
            logger.info("Подключение к Qdrant установлено")
            return client
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {str(e)}")
            # Попытка fallback на HTTP
            try:
                logger.warning("Пробуем подключение по HTTP...")
                return QdrantClient(
                    url=qdrant_url.replace("grpc://", "http://"),
                    api_key=api_key,
                    prefer_grpc=False,
                    timeout=15
                )
            except Exception as fallback_e:
                logger.critical(f"Фолбэк подключение не удалось: {str(fallback_e)}")
                raise ConnectionError("Не удалось подключиться к Qdrant") from fallback_e

    def _init_dense_model(self) -> SentenceTransformer:
        """Безопасная инициализация модели для плотных векторов"""
        try:
            # Загрузка без автоматического перемещения на устройство
            model = SentenceTransformer(
                "cointegrated/rubert-tiny2",
                device=None,  # Не перемещаем автоматически
                trust_remote_code=True
            )
            
            # Явная инициализация на CPU
            model = model.to(self.device)
            
            # Проверка весов
            for param in model.parameters():
                if param.is_meta:
                    raise RuntimeError("Обнаружены мета-тензоры")
            
            logger.info("Dense модель успешно загружена")
            return model
        except Exception as e:
            logger.error(f"Ошибка инициализации dense модели: {str(e)}")
            raise

    def _init_sparse_model(self) -> Optional[SparseTextEmbedding]:
        """Инициализация модели для разреженных векторов"""
        try:
            model = SparseTextEmbedding(
                model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
                device=self.device
            )
            logger.info("Sparse модель успешно загружена")
            return model
        except Exception as e:
            logger.warning(f"Не удалось загрузить sparse модель: {str(e)}")
            return None

    def _init_qdrant_client(self) -> QdrantClient:
        """Инициализация клиента Qdrant"""
        try:
            client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                prefer_grpc=True,
                timeout=30
            )
            client.get_collections()  # Проверка подключения
            return client
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {str(e)}")
            raise

    def _generate_dense_embedding(self, text: str) -> List[float]:
        """Генерация плотного вектора с обработкой ошибок"""
        if not text.strip():
            return []
            
        try:
            with torch.no_grad():
                embedding = self.dense_model.encode(
                    text,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                return embedding.tolist()
        except Exception as e:
            logger.error(f"Ошибка генерации dense вектора: {str(e)}")
            return []

    def _generate_sparse_vector(self, text: str) -> Optional[models.SparseVector]:
        """Генерация разреженного вектора"""
        if not self.sparse_model or not text.strip():
            return None
            
        try:
            embeddings = next(self.sparse_model.embed(text))
            return models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
        except Exception as e:
            logger.warning(f"Ошибка генерации sparse вектора: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def hybrid_search(self, queries: Union[str, List[str]], top_k: int = 5) -> List[Dict]:
        """Устойчивый гибридный поиск"""
        if isinstance(queries, str):
            queries = [queries]
            
        all_results = []
        
        for query in queries:
            try:
                # Генерация векторов
                dense_vec = self._generate_dense_embedding(query)
                sparse_vec = self._generate_sparse_vector(query)
                
                if not dense_vec:
                    continue
                
                # Формирование запроса
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
