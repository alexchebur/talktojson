#indexing.py
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

Path("models").mkdir(exist_ok=True)
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
                        "Qdrant/bm25",
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

    def get_models_status(self) -> dict:
        return {
            'dense_loaded': self.dense_model is not None,
            'sparse_loaded': self.sparse_model is not None,
            'dense_dim': self.dense_model.get_sentence_embedding_dimension() 
                         if self.dense_model else 0
        }

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
            if not hasattr(sparse_vector, 'indices') or not hasattr(sparse_vector, 'values'):
                return False
            if not isinstance(sparse_vector.indices, list) or not isinstance(sparse_vector.values, list):
                return False
            if len(sparse_vector.indices) != len(sparse_vector.values):
                return False
            if len(sparse_vector.indices) == 0:
                return False
            if any(i < 0 for i in sparse_vector.indices):
                return False
            if any(not isinstance(v, float) for v in sparse_vector.values):
                return False
            return True
        except Exception as e:
            logger.error(f"Ошибка проверки sparse-вектора: {str(e)}")
            return False

    def _get_context_around_chunk(self, result: dict, max_chars: int = 10000) -> str:
        """
        Возвращает сбалансированный контекст вокруг найденного чанка, 
        включая соседние чанки из того же документа
        """
        payload = result.get('payload', {})
        #file_path = payload.get('file_path')
        document_id = payload.get('document_id')
        current_chunk_id = payload.get('chunk_id')
        
        if not document_id or current_chunk_id is None:
            return result.get('content', '')[:max_chars]
            
        try:
            # Получаем все чанки этого документа
            file_filter = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
            all_chunks = self.qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=file_filter,
                limit=100,  # Максимальное количество чанков для документа
                with_payload=True,
                with_vectors=False
            )[0]
            
            # Сортируем чанки по chunk_id
            all_chunks_sorted = sorted(all_chunks, key=lambda x: x.payload.get('chunk_id', 0))
            
            # Находим текущий чанк
            current_index = None
            for idx, chunk in enumerate(all_chunks_sorted):
                if chunk.payload.get('chunk_id') == current_chunk_id:
                    current_index = idx
                    break
                    
            if current_index is None:
                return result.get('content', '')[:max_chars]
                
            # Собираем контекст
            context = ""
            current_length = 0
            
            # Начинаем с текущего чанка
            current_chunk_content = all_chunks_sorted[current_index].payload.get('content', '')
            context = current_chunk_content
            current_length = len(context)
            
            # Индексы для движения влево и вправо
            left_index = current_index - 1
            right_index = current_index + 1
            has_left = left_index >= 0
            has_right = right_index < len(all_chunks_sorted)
            
            # Пока не достигнем лимита и есть чанки
            while current_length < max_chars and (has_left or has_right):
                # Сначала идем влево
                if has_left and current_length < max_chars:
                    left_chunk = all_chunks_sorted[left_index]
                    left_content = left_chunk.payload.get('content', '')
                    if current_length + len(left_content) <= max_chars:
                        context = left_content + context
                        current_length += len(left_content)
                    else:
                        # Добавляем только часть
                        remaining_chars = max_chars - current_length
                        context = left_content[-remaining_chars:] + context
                        current_length = max_chars
                        break
                    left_index -= 1
                    has_left = left_index >= 0
                    
                # Затем вправо
                if has_right and current_length < max_chars:
                    right_chunk = all_chunks_sorted[right_index]
                    right_content = right_chunk.payload.get('content', '')
                    if current_length + len(right_content) <= max_chars:
                        context += right_content
                        current_length += len(right_content)
                    else:
                        remaining_chars = max_chars - current_length
                        context += right_content[:remaining_chars]
                        current_length = max_chars
                        break
                    right_index += 1
                    has_right = right_index < len(all_chunks_sorted)
                    
            return context
        except Exception as e:
            logger.error(f"Ошибка получения контекста: {str(e)}")
            return result.get('content', '')[:max_chars]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def hybrid_search(self, queries: Union[str, List[str]], top_k: int = 5) -> List[dict]:
        """Гибридный поиск с плотными и разреженными векторами и расширением контекста"""
        try:
            self._load_models()
        
            # Проверка существования коллекции
            try:
                collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
            except Exception as e:
                logger.error(f"Коллекция {QDRANT_COLLECTION} не найдена: {str(e)}")
                return [], str(e) 

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

                # Обработка результатов
                for req_idx, result_set in enumerate(batch_results):
                    vector_type = "dense" if req_idx % 2 == 0 else "sparse"
                
                    for res in result_set:
                        all_results.append({
                            "id": res.id,
                            "score": res.score,
                            "content": res.payload.get("content", ""),
                            "query": query,
                            "vector_type": vector_type,
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

            # Добавляем расширенный контекст для каждого результата
            for res in unique_results:
                expanded_context = self._get_context_around_chunk(res, max_chars=10000)
                res['expanded_context'] = expanded_context

            return unique_results, None

        except Exception as e:
            logger.error(f"Критическая ошибка гибридного поиска: {str(e)}", exc_info=True)
            return [], str(e)
