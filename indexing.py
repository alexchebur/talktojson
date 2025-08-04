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
            # Загрузка моделей при первом вызове
            self._load_models()
        
            # Проверка существования коллекции
            try:
                collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
                logger.info(f"Коллекция {QDRANT_COLLECTION} найдена, размер: {collection_info.points_count} точек")
            except Exception as e:
                logger.error(f"Коллекция {QDRANT_COLLECTION} не найдена: {str(e)}")
                return []
        
            # Нормализация запросов
            if isinstance(queries, str):
                queries = [queries]
        
            all_results = []
            logger.info(f"Начало гибридного поиска по {len(queries)} запросам")
        
            for query in queries:
                # Пропускаем пустые запросы
                if not query.strip():
                    continue
                
                logger.debug(f"Обработка запроса: '{query}'")
            
                # Генерация плотного вектора
                dense_embedding = self.dense_model.encode(
                    query,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                ).tolist()
            
                # Генерация sparse-вектора
                sparse_vector = self._generate_sparse_vector(query)
            
                # Формируем запросы для Qdrant
                search_requests = []
            
                # Запрос для плотного вектора
                search_requests.append(models.SearchRequest(
                    vector=models.NamedVector(
                        name="dense",
                        vector=dense_embedding
                    ),
                    limit=top_k * 2,
                    with_payload=True
                ))
            
                # Запрос для разреженного вектора (если доступен)
                if sparse_vector:
                    # Проверка формата sparse-вектора
                    if self._is_valid_sparse_vector(sparse_vector):
                        search_requests.append(models.SearchRequest(
                            vector=models.NamedSparseVector(
                                name="sparse",
                                vector=sparse_vector
                            ),
                            limit=top_k * 2,
                            with_payload=True
                        ))
                    else:
                        logger.warning(f"Некорректный sparse-вектор для запроса: '{query}'")
            
                # Выполняем поиск
                try:
                    batch_results = self.qdrant_client.search_batch(
                        collection_name=QDRANT_COLLECTION,
                        requests=search_requests
                    )
                    logger.debug(f"Получено {len(batch_results)} наборов результатов")
                except Exception as e:
                    logger.error(f"Ошибка поиска в Qdrant: {str(e)}")
                    continue
            
                # Обработка результатов
                for result_set in batch_results:
                    for res in result_set:
                        result_data = {
                            "id": res.id,
                            "score": res.score,
                            "content": res.payload.get("content", ""),
                            "query": query,
                            "payload": res.payload
                        }
                    
                        # Определяем тип вектора
                        if "dense" in str(res.vector_name):
                            result_data["vector_name"] = "dense"
                        elif "sparse" in str(res.vector_name):
                            result_data["vector_name"] = "sparse"
                    
                        all_results.append(result_data)
        
            # Дедупликация и сортировка результатов
            seen_ids = set()
            unique_results = []
        
            # Сначала сортируем по релевантности
            sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        
            for res in sorted_results:
                if res['id'] not in seen_ids:
                    seen_ids.add(res['id'])
                    unique_results.append(res)
                    if len(unique_results) >= top_k:
                        break
        
            logger.info(f"Найдено {len(unique_results)} уникальных результатов")
            return unique_results
        
        except Exception as e:
            logger.error(f"Критическая ошибка гибридного поиска: {str(e)}", exc_info=True)
            return []
