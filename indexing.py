import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from typing import Dict, List, Tuple, Optional
import uuid

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            timeout=30
        )
        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self):
        try:
            self.qdrant_client.get_collection(QDRANT_COLLECTION)
        except Exception:
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=768,  # Для all-mpnet-base-v2
                    distance=models.Distance.COSINE
                )
            )
    
    def semantic_search_in_qdrant(
        self, 
        query: str, 
        top_k: int = 10,
        keyword_weight: float = 0.5
    ) -> List[dict]:
        try:
            # Используем гибридный поиск с учетом веса
            results = self.hybrid_search(
                query, 
                top_k=top_k,
                keyword_weight=keyword_weight
            )
            return [{"text": result['payload']['text']} for result in results]
        except Exception as e:
            print(f"Ошибка поиска в Qdrant: {str(e)}")
            return []

    def hybrid_search(
        self, 
        query: str, 
        top_k: int = 5, 
        keyword_weight: float = 0.5
    ) -> List[dict]:
        """Гибридный поиск: семантика + ключевые слова"""
        try:
            # 1. Получаем эмбеддинг для запроса
            query_embedding = self._get_embeddings_batch([query])[0]
            if not query_embedding:
                return []

            # 2. Семантический поиск
            semantic_results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
        
            # 3. Полнотекстовый поиск
            text_results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=query),
                    )]
                ),
                limit=top_k,
                with_payload=True
            )
        
            # 4. Комбинирование результатов
            combined = {}
        
            # Обрабатываем семантические результаты
            for res in semantic_results:
                combined[res.id] = {
                    "payload": res.payload,
                    "semantic_score": res.score,
                    "keyword_score": 0.0
                }
        
            # Добавляем текстовые результаты
            for res in text_results:
                if res.id in combined:
                    combined[res.id]["keyword_score"] = res.score
                else:
                    combined[res.id] = {
                        "payload": res.payload,
                        "semantic_score": 0.0,
                        "keyword_score": res.score
                    }
        
            # 5. Расчет комбинированной оценки
            final_results = []
            for point_id, data in combined.items():
                combined_score = (keyword_weight * data["keyword_score"] + 
                               (1 - keyword_weight) * data["semantic_score"])
                final_results.append({
                    "id": point_id,
                    "payload": data["payload"],
                    "score": combined_score
                })
        
            return sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_k]
    
        except Exception as e:
            print(f"Ошибка гибридного поиска: {str(e)}")
            return []
