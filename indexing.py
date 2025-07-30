from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self):
        self.qdrant_client = self._init_qdrant_client()
        self._tokenizer = None
        self._model = None
        
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

    def _load_models(self):
        """Загрузка моделей для эмбеддингов"""
        if self._tokenizer is None or self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
            self._model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
            self._model.eval()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Семантический поиск с повторами при ошибках"""
        try:
            self._load_models()
            
            with torch.no_grad():
                inputs = self._tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = self._model(**inputs)
                embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embedding = embeddings[0].numpy().tolist()
                
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding,
                limit=top_k,
                with_payload=True
            )
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
        """Полнотекстовый поиск по ключевым словам (нестрогое соответствие)"""
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
