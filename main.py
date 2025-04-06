# -*- coding: utf-8 -*-
import streamlit as st
import json
import os
import re
import pickle
from docx import Document
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
import requests
from typing import List, Dict, Any
from rake_nltk import Rake
from pymorphy2 import MorphAnalyzer

try:
    from config import API_KEY, API_URL
except ImportError:
    st.error("Ошибка: Создайте файл config.py с переменными API_KEY и API_URL")
    API_KEY = ""
    API_URL = "https://api.vsegpt.ru/v1/chat/completions"

#конфигурация
DATA_DIR = "data"
MAX_CONTEXT_LENGTH = 8000
MAX_ANSWER_LENGTH = 15000
TEMPERATURE = 0.4
os.makedirs(DATA_DIR, exist_ok=True)

# Системные промпты
SYSTEM_PROMPT = "Ты - профессиональный опытный юрист-литигатор из энергетической компании. Ты можешь критически оценивать процессуальные документы, комментировать позиции и прогнозировать развитие споров."

BUTTON_PROMPTS = {
    "quality": """Ты юрист-литигатор в энергетической компании Т Плюс. Оцени сильные и слабые стороны процессуального документа, в том числе: 
1. Полноту, последовательность и структуру аргументации, 
2. относимость и достаточность доказательств, 
3. правильность, полноту указания применимых норм права. 
4. суммарное количество недостатков в формате оценки от 1 до 10 ["Я бы поставил итоговую оценку документу ... из 10] 
5.[ВАЖНО] ОБЯЗАТЕЛЬНО критично оцени документ на предмет соответствия методическим рекомендациям Т Плюс по подготовке процессуальных документов, при выявлении нарушений - процитируй нарушенные требования рекомендаций Т Плюс.""",

    "strategy": """Ты - юрист-литигатор в энергетической компании. Оцени правовую ситуацию с точки зрения слабых и сильных мест в позиции оппонента и компании, неопределенности каких-либо фактических обстоятельств, недостатков доказательств. Напиши пошагово стратегию процессуального поведения, например проведения экспертиз, истребования доказательств, смены акцентов в обосновании позиции энергетической компании. Предложи, на чем необходимо сосредоточиться: на сборе доказательств, повышении качества представления интересов, дополнения позиции аргументами или сменой позиции.""",

    "prediction": """Ты - юрист-литигатор в энергетической компании. Оцени правовую ситуацию с точки зрения слабых и сильных мест в позиции энергетической компании, неопределенности каких-либо фактических обстоятельств, недостатков доказательств. Предположи три варианта ответных действий второй стороны после получения процессуального документа энергетической компании, например: [оппонент может пытаться доказывать…], [оппонент может усилить аргументацию в части…], [оппонент попытается опровергать…], с описанием существа действий. Предположи, каков может быть исход дела при неблагоприятном для энергетической компании развитии ситуации."""
}

class TextPreprocessor:
    def __init__(self):
        self.regex = re.compile(r'[^\w\s]')

    def preprocess(self, text: str) -> List[str]:
        text = self.regex.sub(' ', text.lower())
        return text.split()

class BM25SearchEngine:
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.bm25 = None
        self.chunks_info = []
        self.doc_index = defaultdict(list)
        self.is_index_loaded = False

    def build_index(self, documents: List[Dict]) -> None:
        corpus = []
        self.chunks_info = []

        for doc_idx, doc in enumerate(documents):
            self.chunks_info.append({
                'doc_id': doc.get("name", f"doc_{doc_idx}"),
                'doc_name': doc.get("name", "Без названия"),
                'chunk_text': doc.get("content", "")
            })
            corpus.append(doc.get("content", ""))

        tokenized_corpus = [self.preprocessor.preprocess(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_index_loaded = True

    def load_from_cache(self, cache_path: str) -> bool:
        try:
            if not os.path.exists(cache_path):
                return False

            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25, self.chunks_info, self.doc_index = data
                self.is_index_loaded = True
                return True
        except:
            return False

    def save_to_cache(self, cache_path: str) -> None:
        with open(cache_path, 'wb') as f:
            pickle.dump((self.bm25, self.chunks_info, self.doc_index), f)

    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        if not self.is_index_loaded:
            return []

        tokens = self.preprocessor.preprocess(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        best_indices = np.argsort(scores)[-top_n:][::-1]

        results = []
        for idx in best_indices:
            result = {**self.chunks_info[idx], 'score': float(scores[idx])}
            results.append(result)

        return results
class LLMClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.is_initialized = True

    def query(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        try:
            payload = {
                "model": "qwen/qwq-32b",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", error_detail)
                except:
                    pass
                
                raise Exception(f"API Error {response.status_code}: {error_detail}")

            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка соединения: {str(e)}")
        except Exception as e:
            raise Exception(f"Ошибка обработки ответа: {str(e)}")


class DocumentAnalyzer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.search_engine = BM25SearchEngine(self.preprocessor)
        self.llm_client = None
        self.documents = []
        self.llm_initialized = False

    def initialize_llm(self, api_url: str, api_key: str):
        if not api_key:
            raise ValueError("API Key не найден.")
        
        self.llm_client = LLMClient(api_url, api_key)
        self.llm_initialized = True
        return True

    def load_documents(self, uploaded_files):
        self.documents = []
        for file in uploaded_files:
            try:
                doc = Document(file)
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                
                if len(text) > MAX_CONTEXT_LENGTH:
                    truncate_ratio = MAX_CONTEXT_LENGTH / len(text)
                    cutoff = int(len(text) * truncate_ratio)
                    text = text[:cutoff]
                    st.warning(f"Документ {file.name} был обрезан до {cutoff} символов")
                
                self.documents.append({
                    "name": file.name,
                    "content": text
                })
            except Exception as e:
                st.error(f"Ошибка обработки файла {file.name}: {str(e)}")
        
        if self.documents:
            self.search_engine.build_index(self.documents)
            with open(os.path.join(DATA_DIR, "documents.json"), "w") as f:
                json.dump(self.documents, f)
            self.search_engine.save_to_cache(os.path.join(DATA_DIR, "bm25_index.pkl"))

    def analyze_document(self, prompt_type: str) -> str:
        if not self.documents:
            return "Пожалуйста, загрузите документы для анализа"
        
        # Собираем текст всех документов
        full_text = " ".join([doc["content"] for doc in self.documents])
        
        # Определяем веса для разных частей запроса
        SEARCH_WEIGHTS = {
            "base_query": 0.7,    # Вес стандартных ключевых слов
            "doc_content": 1.0    # Вес контента документа
        }
        
        # Формируем комбинированный запрос
        base_queries = {
            "quality": "оценка качества документа структура аргументации доказательства нормы права",
            "strategy": "стратегия спора доказательства процессуальное поведение",
            "prediction": "позиция второй стороны прогнозирование аргументы оппонента"
        }
        
        # Усиливаем важные термины повторами
        boosted_query = (
            f"{base_queries[prompt_type]} " * int(SEARCH_WEIGHTS["base_query"] * 10) +
            f"{full_text[:1000]} " * int(SEARCH_WEIGHTS["doc_content"] * 10)
        )
        
        # Поиск с комбинированным запросом
        chunks = self.search_engine.search(boosted_query)
        context = self._build_context(chunks)
        
        # Формирование промпта для LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BUTTON_PROMPTS[prompt_type] + "\n\nКонтекст:\n" + context}
        ]
        
        return self.llm_client.query(messages, TEMPERATURE, MAX_ANSWER_LENGTH)
        #except Exception as e:
            #return f"Ошибка при анализе документа: {str(e)}"

    def _build_context(self, chunks: List[Dict]) -> str:
        context_parts = ["Релевантные фрагменты из документов:"]
        
        for chunk in sorted(chunks, key=lambda x: x.get('score', 0), reverse=True)[:5]:
            context_parts.extend([
                f"\nДокумент: {chunk.get('doc_name', 'Без названия')}",
                f"Содержание: {chunk.get('chunk_text', '')[:1000]}"
            ])
        
        return '\n'.join(context_parts)[:MAX_CONTEXT_LENGTH]

def main():
    st.set_page_config(page_title="El Documente", layout="wide")
    st.title("El Documente: проверьте свой процессуальный документ")
    
    # Инициализация анализатора
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DocumentAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Автоматическая инициализация LLM при запуске
    if not hasattr(analyzer, 'llm_initialized') or not analyzer.llm_initialized:
        try:
            if analyzer.initialize_llm(API_URL, API_KEY):
                st.session_state.llm_initialized = True
                st.sidebar.success("LLM успешно инициализирован")
            else:
                st.sidebar.error("Не удалось инициализировать LLM")
        except Exception as e:
            st.sidebar.error(f"Ошибка инициализации: {str(e)}")
    
    # Загрузка документов
        weights = st.sidebar.slider(
            "Вес контента документа в поиске",
            0.1, 2.0, 0.7, 0.1
        )
    st.header("Загрузка документов")
    uploaded_files = st.file_uploader(
        "Выберите документы в формате DOCX", 
        type=["docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Обработка документов..."):
            analyzer.load_documents(uploaded_files)
        st.success(f"Загружено документов: {len(uploaded_files)}")
    
    # Кнопки анализа
    st.header("Анализ документов")
    col1, col2, col3 = st.columns(3)

    # Проверка инициализации LLM и загрузки документов
    buttons_disabled = not (uploaded_files and hasattr(analyzer, 'llm_initialized') and analyzer.llm_initialized)
    
    with col1:
        if st.button("Оценить качество документа", disabled=buttons_disabled):
            with st.spinner("Анализ документа..."):
                result = analyzer.analyze_document("quality")
                st.markdown("### Результат оценки")
                st.markdown(result)  # Автоматически рендерит Markdown
    
    with col2:
        if st.button("Дать рекомендации по стратегии спора", disabled=buttons_disabled):
            with st.spinner("Формирование рекомендаций..."):
                result = analyzer.analyze_document("strategy")
                st.markdown("### Рекомендации по стратегии")
                st.markdown(result)
    
    with col3:
        if st.button("Спрогнозировать позицию второй стороны", disabled=buttons_disabled):
            with st.spinner("Прогнозирование позиции..."):
                result = analyzer.analyze_document("prediction")
                st.markdown("### Прогноз позиции оппонента")
                st.markdown(result)

if __name__ == "__main__":
    main()
