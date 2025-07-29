import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from typing import Dict, List, Tuple, Optional
import uuid
from sentence_transformers import SentenceTransformer
from pymystem3 import Mystem
from qdrant_client.models import TextIndexParams, TokenizerType, KeywordIndexParams

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            timeout=30
        )
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.mystem = Mystem() if self._check_mystem() else None
        self._ensure_qdrant_collection()
    
    def _check_mystem(self):
        try:
            from pymystem3 import Mystem
            return True
        except ImportError:
            print("ℹ️ Лемматизатор Mystem недоступен. Установите pymystem3 для лучшего поиска")
            return False
    
    def _ensure_qdrant_collection(self):
        try:
            collection_info = self.qdrant_client.get_collection(QDRANT_COLLECTION)
            # Проверяем наличие текстового индекса с русским стеммингом
            if "text" not in collection_info.payload_schema:
                self._create_text_index()
        except Exception:
            self._create_collection_with_indexes()
    
    def _create_collection_with_indexes(self):
        self.qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=768,
                distance=models.Distance.COSINE
            )
        )
        self._create_text_index()
    
    def _create_text_index(self):
        self.qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True
            )
        )
    
    # Добавить словарь синонимов
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
        """Расширяет запрос синонимами и стеммингом"""
        if self.mystem:
            lemmas = self.mystem.lemmatize(query)
            clean_query = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in ' '.join(lemmas))
        else:
            clean_query = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in query)
    
        words = clean_query.split()
        expanded = []
        for word in words:
            expanded.append(word)
            if word in self.SYNONYMS:
                expanded.extend(self.SYNONYMS[word])
        return " ".join(expanded)

    def hybrid_search(self, query: str, top_k: int = 5, keyword_weight: float = 0.5) -> List[dict]:
        """Обновленный гибридный поиск с расширением запроса"""
        expanded_query = self.expand_query(query)
    
        # Семантический поиск
        query_embedding = self.model.encode(expanded_query).tolist()
        semantic_results = self.qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_embedding,
            limit=top_k * 2,
            with_payload=True
        )
    
        # Полнотекстовый поиск
        text_results = self.qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_filter=models.Filter(
                must=[models.FieldCondition(
                    key="text",
                    match=models.MatchText(text=expanded_query[:100])  # Ограничиваем длину запроса
                )]
            ),
            limit=top_k * 2,
            with_payload=True
        )
    
        # Объединение и ранжирование (как в примере)
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
