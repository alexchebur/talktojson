import re
import os
import numpy as np
from difflib import SequenceMatcher
from typing import List
from utils import clean_keyword

class DataProcessor:
    def __init__(self, index_builder):
        self.index_builder = index_builder
    
    def process_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + 10000
            chunks.append(text[start:end])
            start += 10000 - 1000
        return chunks
    
    def extract_keywords(self, text: str, bm25) -> List[str]:
        try:
            words = re.findall(r'\b[а-яё]+\b', text.lower())
            stop_words = {"на", "под", "в", "среди", "перед", "затем", "после", "до", "сразу"}
            
            filtered = [
                word for word in words
                if len(word) >= 5 
                and word not in stop_words
                and not re.search(r'\d', word)
            ]

            scores = bm25.get_scores(filtered)
            scored_words = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
            
            unique_words = []
            seen = set()
            for word, _ in scored_words:
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)
                    if len(unique_words) == 20:
                        break

            return [clean_keyword(word) for word in unique_words]

        except Exception as e:
            print(f"Ошибка извлечения ключевых слов: {str(e)}")
            return []
    
    def search_relevant_chunks(self, bm25, original_chunks: List[str], keywords: List[str]) -> List[str]:
        try:
            query_weights = {term: 2 for term in keywords}
            weighted_query = []
            for term, weight in query_weights.items():
                weighted_query.extend([term] * weight)
            
            doc_scores = np.array(bm25.get_scores(weighted_query))
            sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
            return [original_chunks[i] for i in sorted_indices if doc_scores[i] > 0.0][:5]
        
        except Exception as e:
            print(f"Ошибка поиска: {str(e)}")
            return []
    
    def get_unique_chunks(self, main_chunks: List[str], new_chunks: List[str]) -> List[str]:
        unique_chunks = []
        for new_chunk in new_chunks:
            is_duplicate = False
            for main_chunk in main_chunks:
                if SequenceMatcher(None, main_chunk[:1000], new_chunk[:1000]).ratio() > 0.8:
                    is_duplicate = True
                    break
            if not is_duplicate and new_chunk not in unique_chunks:
                unique_chunks.append(new_chunk)
        return unique_chunks
    
    def enhance_with_semantic_search(
        self, 
        query: str, 
        original_chunks: List[str],
        top_k: int = 3
    ) -> List[str]:
        related_docs = self.index_builder.semantic_search(query, top_k)
        enhanced_chunks = []
        
        for doc_name in related_docs:
            doc_path = os.path.join("documents", doc_name)
            try:
                encoding = self.index_builder._detect_file_encoding(doc_path)
                with open(doc_path, 'r', encoding=encoding, errors='replace') as f:
                    text = f.read()
                chunks = self.index_builder._process_text(text)
                enhanced_chunks.extend(chunks[:2])
            except:
                continue
        
        return self.get_unique_chunks(original_chunks, enhanced_chunks)

    def enhance_with_graph_context(
        self, 
        main_doc_name: str,
        original_chunks: List[str],
        depth: int = 1
    ) -> List[str]:
        if not main_doc_name:
            return original_chunks
            
        related_docs = []
        to_explore = [main_doc_name]
        
        for _ in range(depth + 1):
            next_level = []
            for doc in to_explore:
                if doc not in related_docs:
                    related_docs.append(doc)
                    next_level.extend(self.index_builder.get_related_documents(doc))
            to_explore = next_level
        
        enhanced_chunks = []
        for doc_name in related_docs[1:]:
            doc_path = os.path.join("documents", doc_name)
            try:
                encoding = self.index_builder._detect_file_encoding(doc_path)
                with open(doc_path, 'r', encoding=encoding, errors='replace') as f:
                    text = f.read()
                chunks = self.index_builder._process_text(text)
                enhanced_chunks.extend(chunks[:1])
            except:
                continue
        
        return self.get_unique_chunks(original_chunks, enhanced_chunks)

    
    def enhance_with_qdrant_search(
        self,
        query: str,
        keywords: List[str],
        original_chunks: List[str],
        top_k: int = 10
    ) -> List[str]:
        try:
            qdrant_results = self.index_builder.semantic_search_in_qdrant(query, keywords, top_k)
            return [result["text"] for result in qdrant_results]
        except Exception as e:
            print(f"Ошибка поиска в Qdrant: {str(e)}")
            return []
    
    def build_context(self, *chunk_sources: List[str]) -> str:
        context_parts = []
        for i, chunks in enumerate(chunk_sources):
            if chunks:
                context_parts.append(f"Контекст {i+1}:\n" + "\n\n".join(chunks[:3]))
        return "\n\n".join(context_parts)
