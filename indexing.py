def sparse_vector_search(self, queries: List[str], top_k: int = 5) -> List[dict]:
    """Корректная реализация поиска по разреженным векторам"""
    try:
        self._load_models()
        all_results = []
        
        for query in queries:
            # Генерация разреженного вектора
            embeddings = list(self.sparse_model.embed(query))
            if not embeddings:
                continue
                
            sparse_embedding = embeddings[0]
            
            # Создаем словарь с явным указанием типов
            sparse_dict = {
                "indices": [int(i) for i in sparse_embedding.indices.tolist()],
                "values": [float(v) for v in sparse_embedding.values.tolist()]
            }
            
            # Вариант 1: Используем NamedSparseVector (для новых версий Qdrant)
            try:
                from qdrant_client.http.models import NamedSparseVector
                results = self.qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=NamedSparseVector(
                        name="sparse",
                        vector=sparse_dict
                    ),
                    limit=top_k,
                    with_payload=True
                )
            except:
                # Вариант 2: Старый формат для совместимости
                results = self.qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector={
                        "sparse": {
                            "indices": sparse_dict["indices"],
                            "values": sparse_dict["values"]
                        }
                    },
                    limit=top_k,
                    with_payload=True
                )
            
            # Форматируем результаты
            for res in results:
                all_results.append({
                    "id": res.id,
                    "score": float(res.score),
                    "content": res.payload.get("content", ""),
                    "query": query,
                    "payload": res.payload
                })
        
        # Удаляем дубликаты и сортируем
        unique_results = {res['id']: res for res in all_results}.values()
        return sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
    except Exception as e:
        logger.error(f"Ошибка sparse поиска: {str(e)}", exc_info=True)
        return []
