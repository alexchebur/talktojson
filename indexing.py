import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer  # Только этот импорт нужен

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self.model = None  # Будет загружен при первом использовании
        
    def _init_qdrant_client(self):
        """Инициализация клиента Qdrant с обработкой ошибок"""
        try:
            client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True,
                timeout=30
            )
            # Проверка соединения
            client.get_collections()
            return client
        except Exception as e:
            logger.error(f"Ошибка подключения к Qdrant: {str(e)}")
            raise

    def _load_model(self):
        """Загрузка модели sentence-transformers"""
        if self.model is None:
            self.model = SentenceTransformer(
                "cointegrated/rubert-tiny2",
                device="cpu",  # Используйте "cuda" если есть GPU
                trust_remote_code=True  # Критично для rubert-tiny2
            )


            #self.model = SentenceTransformer(
                #"cointegrated/rubert-tiny2",
                #device="cpu",
                #use_onnx=True,  # АКТИВИРУЕМ ONNX
                #onnx_providers=["CPUExecutionProvider"]  # Используем CPU
            #)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Семантический поиск с использованием sentence-transformers"""
        try:
            self._load_model()
            
            # Генерация эмбеддинга (sentence-transformers делает mean pooling + L2 нормализацию автоматически)
            embedding = self.model.encode(
                query,
                normalize_embeddings=True,  # Обязательно для совместимости с Qdrant
                convert_to_numpy=True
            ).tolist()
            
            # Используем современный API Qdrant
            results = self.qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=embedding,
                limit=top_k,
                with_payload=True
            ).points

            return [{
                "id": res.id,
                "score": res.score,
                "payload": res.payload,
                "text": res.payload.get("text", "")
            } for res in results]
            
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {str(e)}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 5) -> List[dict]:
        """Полнотекстовый поиск по ключевым словам"""
        should_conditions = [
            models.FieldCondition(
                key="text",
                match=models.MatchText(text=keyword)  
            ) for keyword in keywords
        ]
        
        results = self.qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=models.Filter(should=should_conditions),
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )[0]
        
        return [{
            "id": hit.id,
            "payload": hit.payload,
            "text": hit.payload.get("text", "")
        } for hit in results]
    
    # Словарь синонимов (оставлен без изменений)
    SYNONYMS = {
        "теплосетевая": ["сетевая", "теплосеть", "теплоснабжающая"],
        "электросетевая": ["сетевая", "электросеть", "электроснабжающая"],
        "предельный": ["максимальный", "верхний предел", "лимит"],
        "убытки": ["ущерб", "потери", "финансовые потери", "компенсация"],
        "расторжение": ["прекращение", "закрытие", "отмена"],
        "заявитель": ["потребитель", "пользователь", "клиент", "заказчик"],
        "технологическое присоединение": ["подключение", "присоединение", "техническое подключение"]
    }

    def expand_query(self, query: str) -> str:
        """Расширяет запрос синонимами"""
        # Упрощенная обработка (удалена зависимость от mystem)
        clean_query = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in query)
        words = clean_query.split()
        
        expanded = []
        for word in words:
            expanded.append(word)
            if word in self.SYNONYMS:
                expanded.extend(self.SYNONYMS[word])
        return " ".join(expanded)

    def hybrid_search(self, query: str, top_k: int = 5, keyword_weight: float = 0.5) -> List[dict]:
        """Гибридный поиск с расширением запроса"""
        expanded_query = self.expand_query(query)
        
        # Семантический поиск через sentence-transformers
        self._load_model()
        query_embedding = self.model.encode(
            expanded_query,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist()
        
        semantic_results = self.qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_embedding,
            limit=top_k * 2,
            with_payload=True
        ).points
        
        # Полнотекстовый поиск
        text_results = self.qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=expanded_query[:100],  # Используем текст как запрос
            using="text_index",  # Убедитесь, что у вас есть текстовый индекс
            limit=top_k * 2,
            with_payload=True
        ).points
        
        # Объединение результатов
        combined = {}
        for res in semantic_results:
            combined[res.id] = {
                "payload": res.payload,
                "semantic_score": res.score,
                "keyword_score": 0.0
            }
        
        for res in text_results:
            if res.id in combined:
                combined[res.id]["keyword_score"] = res.score
            else:
                combined[res.id] = {
                    "payload": res.payload,
                    "semantic_score": 0.0,
                    "keyword_score": res.score
                }
        
        # Ранжирование
        final_results = []
        for point_id, data in combined.items():
            if data["keyword_score"] > 0:
                score = (keyword_weight * data["keyword_score"] + 
                       (1 - keyword_weight) * data["semantic_score"])
            else:
                score = 0.9 * data["semantic_score"]
            
            final_results.append({
                "id": point_id,
                "payload": data["payload"],
                "score": score,
                "semantic_score": data["semantic_score"],
                "keyword_score": data["keyword_score"]
            })
        
        return sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]
