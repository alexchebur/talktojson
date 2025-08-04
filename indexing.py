import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
from pathlib import Path
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from qdrant_client import QdrantClient, models
from qdrant_client.http import models
from typing import Tuple, List, Dict, Optional, Union
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

    def _load_models(self) -> Tuple[bool, str]:
        """Загрузка моделей с явным указанием возвращаемого типа"""
        try:
            # Dense модель
            if self.dense_model is None:
                self.dense_model = SentenceTransformer(
                    "cointegrated/rubert-tiny2",
                    device="cpu",
                    cache_folder=os.path.join(os.getcwd(), "models")
                )
                logger.info("Dense модель загружена")

            # Sparse модель
            if self.sparse_model is None:
                try:
                    self.sparse_model = SparseTextEmbedding(
                        "Qdrant/bm42-all-minilm-l6-v2-attentions",
                        cache_dir=os.path.join(os.getcwd(), "models")
                    )
                    logger.info("Sparse модель загружена")
                    return True, "Все модели успешно загружены"
                except Exception as e:
                    logger.error(f"Ошибка загрузки sparse модели: {str(e)}")
                    return False, f"Ошибка sparse модели: {str(e)}"

            return True, "Модели уже были загружены"
    
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки моделей: {str(e)}")
            return False, f"Критическая ошибка: {str(e)}"

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
    def _is_valid_sparse_vector(self, sparse_vector: models.SparseVector) -> bool:
        """Проверка корректности формата sparse-вектора"""
        try:
            # Проверка наличия обязательных атрибутов
            if not hasattr(sparse_vector, 'indices') or not hasattr(sparse_vector, 'values'):
                logger.error("Sparse-вектор не содержит indices или values")
                return False
            
            # Проверка типов
            if not isinstance(sparse_vector.indices, list) or not isinstance(sparse_vector.values, list):
                logger.error("Sparse-вектор содержит неверные типы данных")
                return False
            
            # Проверка длины массивов
            if len(sparse_vector.indices) != len(sparse_vector.values):
                logger.error("Длина indices и values не совпадает")
                return False
            
            # Проверка на пустые данные
            if len(sparse_vector.indices) == 0:
                logger.warning("Пустой sparse-вектор")
                return False
            
            # Проверка на допустимость значений индексов
            if any(i < 0 for i in sparse_vector.indices):
                logger.error("Отрицательные индексы в sparse-векторе")
                return False
            
            # Проверка на допустимость значений
            if any(not isinstance(v, float) for v in sparse_vector.values):
                logger.error("Некорректные типы значений в sparse-векторе")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Ошибка проверки sparse-вектора: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def hybrid_search(self, queries: Union[str, List[str]], top_k: int = 5) -> List[dict]:
        """Гибридный поиск с плотными и разреженными векторами"""
        try:
            self._load_models()
        
            # Проверка существования коллекции
            try:
                collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
                logger.info(f"Коллекция {QDRANT_COLLECTION} найдена")
            except Exception as e:
                logger.error(f"Коллекция {QDRANT_COLLECTION} не найдена: {str(e)}")
                return []

            if isinstance(queries, str):
                queries = [queries]

            all_results = []
        
            for query_idx, query in enumerate(queries):
                if not query.strip():
                    continue

                # Генерация векторов
                dense_embedding = self.dense_model.encode(
                    query,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).tolist()
            
                sparse_vector = self._generate_sparse_vector(query)
            
                # Формируем запросы
                requests = []
            
                # Запрос для dense
                requests.append(models.SearchRequest(
                    vector=models.NamedVector(
                        name="dense",
                        vector=dense_embedding
                    ),
                    limit=top_k * 2,
                    with_payload=True
                ))
            
                # Запрос для sparse (если доступен)
                if sparse_vector and self._is_valid_sparse_vector(sparse_vector):
                    requests.append(models.SearchRequest(
                        vector=models.NamedSparseVector(
                            name="sparse",
                            vector=sparse_vector
                        ),
                        limit=top_k * 2,
                        with_payload=True
                    ))

                # Выполняем поиск
                try:
                    batch_results = self.qdrant_client.search_batch(
                        collection_name=QDRANT_COLLECTION,
                        requests=requests
                    )
                except Exception as e:
                    logger.error(f"Ошибка поиска для запроса '{query}': {str(e)}")
                    continue

                # Обрабатываем результаты с учетом типа вектора
                for req_idx, result_set in enumerate(batch_results):
                    vector_type = "dense" if req_idx == 0 else "sparse"
                
                    for res in result_set:
                        all_results.append({
                            "id": res.id,
                            "score": res.score,
                            "content": res.payload.get("content", ""),
                            "query": query,
                            "vector_name": vector_type,
                            "payload": res.payload
                        })

            # Дедупликация и сортировка
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
            logger.error(f"Критическая ошибка гибридного поиска: {str(e)}", exc_info=True)
            return []
