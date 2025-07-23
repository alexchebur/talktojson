import os
import re
import json
import time
import chardet
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from config import GEMINI_API_KEY, API_URL, API_TIMEOUT
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
import uuid

class IndexBuilder:
    def __init__(self, chunk_size: int = 10000, chunk_overlap: int = 1000):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.bm25_index = None
        self.original_chunks = []
        #self.embeddings_index: Dict[str, List[float]] = {}
        #self.document_graph: Dict[str, List[str]] = {}
        #self.EMBEDDINGS_CACHE_DIR = "data/embeddings"
        self.embeddings_index = {}
        self.document_graph = {}
        #self.EMBEDDINGS_CACHE_DIR = os.path.abspath("data/embeddings")
        #self.EMBEDDINGS_CACHE_DIR = os.path.abspath("/tmp/embeddings")
        #self.EMBEDDINGS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings")
        self.EMBEDDING_MODEL = "models/embedding-001"
        self.MAX_BATCH_SIZE = 5
        self.RATE_LIMIT_DELAY = 60
        #os.makedirs(self.EMBEDDINGS_CACHE_DIR, exist_ok=True)
        #os.makedirs(self.EMBEDDINGS_CACHE_DIR, exist_ok=True)
        #print(f"Папка эмбеддингов: {os.path.abspath(self.EMBEDDINGS_CACHE_DIR)}")
        # Замените существующую строку на:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.EMBEDDINGS_CACHE_DIR = os.path.join(REPO_ROOT, "data", "embeddings")
        os.makedirs(self.EMBEDDINGS_CACHE_DIR, exist_ok=True)
        print(f"Папка эмбеддингов: {self.EMBEDDINGS_CACHE_DIR}")
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            prefer_grpc=True,
            timeout=30
        )
        self._ensure_qdrant_collection()
    
    def _ensure_qdrant_collection(self):
        try:
            # Проверяем наличие коллекции
            self.qdrant_client.get_collection(QDRANT_COLLECTION)
        except Exception:
            # Создаем коллекцию с векторами
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
    
        # Создаем полнотекстовый индекс для поля "text"
        self.qdrant_client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="text",  # Поле в payload для индексации
            field_schema=models.PayloadSchemaType.KEYWORD,  # Используем KEYWORD для полнотекстового поиска
            text_index_params=models.TextIndexParams(
                tokenizer="word",  # Токенизатор
                min_token_len=2,                         # Минимальная длина токена
                max_token_len=15,                        # Максимальная длина токена
                lowercase=True                           # Приведение к нижнему регистру
            )
        )
    def _upload_to_qdrant(self, text: str, embedding: List[float]):
        chunks = self._process_text(text)
        
        points = []
        for idx, chunk in enumerate(chunks):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        #"filename": filename,
                        "text": chunk,
                        "full_text": text[:10000],  # Первые 10k символов для контекста
                        "chunk_num": idx,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            )
        
        self.qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points,
            wait=True
        )
    
    def semantic_search_in_qdrant(self, query: str, keywords: List[str], top_k: int = 10) -> List[dict]:
        try:
            # Генерируем эмбеддинг для векторного поиска
            query_embedding = self._get_embeddings_batch([query])[0]
            if not query_embedding:
                return []

            # Создаем фильтр для полнотекстового поиска
            filter_conditions = [
                models.FieldCondition(
                    key="text",  # Поле в payload для поиска
                    match=models.MatchText(text=keyword)  # Поиск совпадений по ключевому слову
                ) for keyword in keywords
            ]

            # Выполняем поиск с фильтром
            results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding,
                query_filter=models.Filter(must=filter_conditions),  # Комбинируем условия
                limit=top_k,
                with_payload=True
            )

            # Возвращаем результаты
            return [{
                "text": hit.payload["text"],
                "filename": hit.payload["filename"],
                "score": hit.score
            } for hit in results]
    
        except Exception as e:
            print(f"Ошибка поиска в Qdrant: {str(e)}")
            return []
    

    def _process_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _detect_file_encoding(self, file_path: str) -> str:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
        return chardet.detect(raw_data)['encoding']

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            contents = [{"parts": [{"text": text[:10000]}]} for text in texts]
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/{self.EMBEDDING_MODEL}:batchEmbedContents",
                params={"key": GEMINI_API_KEY},
                json={
                    "requests": [
                        {"model": self.EMBEDDING_MODEL, "content": content}
                        for content in contents
                    ]
                },
                timeout=30
            )
            
            if response.status_code == 429:
                print(f"Достигнут лимит запросов. Ожидание {self.RATE_LIMIT_DELAY} секунд...")
                time.sleep(self.RATE_LIMIT_DELAY)
                return self._get_embeddings_batch(texts)
                
            response.raise_for_status()
            response_data = response.json()
            return [embedding["values"] for embedding in response_data.get("embeddings", [])]
            
        except Exception as e:
            print(f"Ошибка получения эмбеддингов: {str(e)}")
            return []

    def _get_cached_embedding(self, filename: str) -> Tuple[bool, List[float]]:
        cache_path = os.path.join(self.EMBEDDINGS_CACHE_DIR, f"{filename}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    if "embedding" in data:
                        print(f"Использован кэш для {filename}")
                        return True, data["embedding"]
            except:
                pass
        return False, []

    def _save_embedding_to_cache(self, filename: str, embedding: List[float]):
        # Нормализуем имя файла для безопасности
        safe_name = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
        cache_path = os.path.join(self.EMBEDDINGS_CACHE_DIR, f"{safe_name}.json")
    
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "filename": filename,
                    "embedding": embedding,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
            print(f"Эмбеддинг сохранен в репозитории: {cache_path}")
        except Exception as e:
            print(f"Ошибка сохранения в репозиторий: {str(e)}")
            raise

    def _build_document_graph(self, filename: str, text: str):
        # Всегда добавляем документ в граф
        if filename not in self.document_graph:
            self.document_graph[filename] = []
    
        # Поиск ссылок
        for doc_name in self.embeddings_index.keys():
            if doc_name != filename and re.search(r"\b" + re.escape(doc_name) + r"\b", text):
                self.document_graph[filename].append(doc_name)

    def _save_full_index(self):
        index_path = os.path.join(self.EMBEDDINGS_CACHE_DIR, "full_index.json")
        with open(index_path, "w") as f:
            json.dump({
                "embeddings_index": self.embeddings_index,
                "document_graph": self.document_graph,
                "timestamp": datetime.now().isoformat()
            }, f)

    def build_embeddings_index(self, documents_dir: str):
        self.embeddings_index = {}
        self.document_graph = {}
        
        txt_files = [f for f in os.listdir(documents_dir) if f.endswith(".txt")]
        if not txt_files:
            return
            
        cached_embeddings = {}
        files_to_process = []
        file_contents = {}
        
        for filename in txt_files:
            in_cache, embedding = self._get_cached_embedding(filename)
            if in_cache:
                cached_embeddings[filename] = embedding
            else:
                files_to_process.append(filename)
        
        for i in range(0, len(files_to_process), self.MAX_BATCH_SIZE):
            batch_files = files_to_process[i:i + self.MAX_BATCH_SIZE]
            batch_texts = []
            
            for filename in batch_files:
                file_path = os.path.join(documents_dir, filename)
                try:
                    encoding = self._detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        text = f.read()
                    file_contents[filename] = text
                    batch_texts.append(text[:10000])
                except Exception as e:
                    print(f"Ошибка чтения {filename}: {str(e)}")
                    continue
            
            embeddings = self._get_embeddings_batch(batch_texts)
            
        for j, filename in enumerate(batch_files):
            if j < len(embeddings):
                self.embeddings_index[filename] = embeddings[j]
                self._save_embedding_to_cache(filename, embeddings[j])
                print(f"Сохранен эмбеддинг для {filename}")
            else:
                print(f"Не удалось получить эмбеддинг для {filename}")
        
        self.embeddings_index.update(cached_embeddings)


        for j, filename in enumerate(batch_files):
            if j < len(embeddings):
                self.embeddings_index[filename] = embeddings[j]
                self._save_embedding_to_cache(filename, embeddings[j])
                # Добавлено: загрузка в Qdrant
                self._upload_to_qdrant(filename, batch_texts[j], embeddings[j])
                print(f"Сохранен эмбеддинг для {filename} в Qdrant")
        
        for filename in txt_files:
            if filename in file_contents:
                text = file_contents[filename]
            else:
                file_path = os.path.join(documents_dir, filename)
                try:
                    encoding = self._detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        text = f.read()
                except:
                    continue
            self._build_document_graph(filename, text)
        
        self._save_full_index()
        print(f"Индекс эмбеддингов построен. Документов: {len(self.embeddings_index)}")

    def load_full_index(self):
        index_path = os.path.join(self.EMBEDDINGS_CACHE_DIR, "full_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                    self.embeddings_index = data.get("embeddings_index", {})
                    self.document_graph = data.get("document_graph", {})
                    print(f"Загружен индекс из кэша. Документов: {len(self.embeddings_index)}")
                    return True
            except:
                pass
        return False

    def semantic_search(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self._get_embeddings_batch([query])[0]
        if not query_embedding:
            return []
        
        results = []
        for doc_name, doc_embedding in self.embeddings_index.items():
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            results.append((doc_name, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc_name for doc_name, _ in results[:top_k]]

    def get_related_documents(self, doc_name: str) -> List[str]:
        return self.document_graph.get(doc_name, [])

    def create_bm25_index(self, documents_dir: str = "documents") -> Tuple[Optional[BM25Okapi], List[str]]:
        try:
            if not os.path.exists(documents_dir):
                os.makedirs(documents_dir)
                return None, []

            txt_files = [f for f in os.listdir(documents_dir) if f.endswith(".txt")]
            if not txt_files:
                return None, []

            all_chunks = []
            original_texts = []

            for filename in txt_files:
                file_path = os.path.join(documents_dir, filename)
                try:
                    encoding = self._detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        text = f.read()
                    chunks = self._process_text(text)
                    all_chunks.extend(chunks)
                    original_texts.extend(chunks)
                except Exception as e:
                    print(f"Ошибка чтения {filename}: {str(e)}")
                    continue

            if not all_chunks:
                return None, []

            tokenized_chunks = [doc.split() for doc in all_chunks]
            self.bm25_index = BM25Okapi(tokenized_chunks, k1=1.8, b=0.75)
            self.original_chunks = original_texts
            print(f"индекс bm25 создан")

            return self.bm25_index, self.original_chunks

        except Exception as e:
            print(f"Ошибка создания индекса: {str(e)}")
            return None, []
