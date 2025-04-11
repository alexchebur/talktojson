# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import json
import io
from docx import Document
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
import requests
from typing import List, Dict, Any
import shutil
from pathlib import Path
import glob

try:
    from config import API_KEY, API_URL
except ImportError:
    st.error("Ошибка: Создайте файл config.py с переменными API_KEY и API_URL")
    API_KEY = ""
    API_URL = "https://api.vsegpt.ru/v1/chat/completions"

# Конфигурация
DATA_DIR = "data"
MAX_CONTEXT_LENGTH = 20000
MAX_ANSWER_LENGTH = 15000
TEMPERATURE = 0.2
os.makedirs(DATA_DIR, exist_ok=True)

# Промпт для генерации ключевых слов (упрощенный)
KEYWORDS_PROMPT = """
Извлеки из текста ключевые юридические термины и слова, которые точно отражают суть документа.
Сформируй список из 15-20 наиболее значимых слов и их синонимов, которые помогут найти релевантные документы в юридической базе.
Используй только конкретные термины, избегая общих слов.
"""

# Системные промпты
SYSTEM_PROMPT = """Ты - профессиональный опытный юрист-литигатор из энергетической компании. Ты можешь критически оценивать
процессуальные документы, комментировать позиции и прогнозировать развитие споров. правила цитирования: ЗАПРЕЩЕНО цитировать и ссылаться на конкретные нормы, пункты, правила и конкретные законы,
отсутствующие в предоставленных документах. Все выводы и цитаты должны основываться на тексте предоставленных документов."""

BUTTON_PROMPTS = {
    "quality": """Ты юрист-литигатор в энергетической компании. Оцени сильные и слабые стороны процессуального документа НА ОСНОВЕ ЭТИХ ФРАГМЕНТОВ:\n{context}\n\n. Правила анализа: 
    1. Полноту, последовательность и структуру аргументации, 
    2. относимость и достаточность доказательств, 
    3. полноту указания в подкрепление аргументов применимых норм права. 
    4. суммарное количество недостатков в формате оценки от 1 до 10 ["Я бы поставил итоговую оценку документу ... из 10]
    5. правила цитирования: ЗАПРЕЩЕНО цитировать и ссылаться на конкретные нормы и конкретные законы, отсутствующие в предоставленных документах. Все выводы и цитаты должны основываться на тексте предоставленных документов.
    6.[ВАЖНО] критично оцени документ на предмет соответствия методическим указаниям по процессуальным документам из фрагментов  \n{context}\n\n  (если рекомендации представлены в документе), при выявлении нарушений - процитируй нарушенные требования методических указаний, если требования указаны в предоставленных документах.""",

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
        self.is_index_loaded = False
        self.cache_path = os.path.join("data", "bm25_index.json")
        self._load_index()
        self.llm_keywords = []
        
    def _normalize_processed(self, processed_data: Any) -> List[str]:
        """Нормализует поле processed в единый формат списка токенов"""
        if processed_data is None:
            return []
    
        if isinstance(processed_data, str):
            return [token for token in processed_data.split() if token]
        elif isinstance(processed_data, list):
            result = []
            for item in processed_data:
                if isinstance(item, str):
                    result.extend(item.split())
                elif isinstance(item, (int, float)):
                    result.append(str(item))
            return result
        else:
            return [str(processed_data)]

    def _load_index(self) -> bool:
        """Загрузка индекса с проверкой"""
        if not os.path.exists(self.cache_path):
            st.sidebar.warning(f"Файл индекса не найден: {self.cache_path}")
            return False

        try:
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, dict) or 'metadata' not in data:
                st.sidebar.error("Неверный формат индекса")
                return False
                
            processed_texts = []
            valid_metadata = []
            
            for item in data.get('metadata', []):
                if not isinstance(item, dict):
                    continue
                    
                original_text = item.get('original', '')
                processed = self._normalize_processed(item.get('processed', []))
                
                if not processed and original_text:
                    processed = self.preprocessor.preprocess(original_text)
                
                if processed:
                    processed_texts.append(processed)
                    valid_metadata.append(item)
            
            if not processed_texts:
                st.sidebar.error("Индекс не содержит валидных документов")
                return False
            
            self.bm25 = BM25Okapi(processed_texts)
            self.chunks_info = valid_metadata
            self.is_index_loaded = True
            
            st.sidebar.success(f"Загружен индекс с {len(processed_texts)} фрагментами данных")
            return True
            
        except Exception as e:
            st.sidebar.error(f"Ошибка загрузки индекса: {str(e)}")
            return False

    def search(self, keywords: List[str], top_n: int = 5) -> List[Dict]:
        """Поиск только по ключевым словам от LLM"""
        if not self.is_index_loaded or not keywords:
            return []

        try:
            tokens = self.preprocessor.preprocess(' '.join(keywords))
            
            if not tokens:
                return []

            scores = self.bm25.get_scores(tokens)
            best_indices = np.argsort(scores)[-top_n:][::-1]

            return [
                {
                    'doc_id': self.chunks_info[idx].get('file_id', ''),
                    'doc_name': self.chunks_info[idx].get('doc_name', 'Документ'),
                    'chunk_text': self.chunks_info[idx].get('original', '')[:2000],
                    'score': round(float(scores[idx]), 4)
                }
                for idx in best_indices
                if idx < len(self.chunks_info)
            ]
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

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
                "model": "google/gemini-2.0-flash-lite-001",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
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
    def __init__(self, api_url: str = None, api_key: str = None):
        self.preprocessor = TextPreprocessor()
        self.search_engine = BM25SearchEngine(self.preprocessor)
        self.current_docx = None
        self.llm_client = None
        self.llm_initialized = False
        self._initialize_llm(api_url, api_key)

    def _initialize_llm(self, api_url: str, api_key: str) -> None:
        """Инициализирует клиент LLM"""
        if not api_url or not api_key:
            self.llm_initialized = False
            return
            
        try:
            self.llm_client = LLMClient(api_url, api_key)
            self.llm_initialized = True
        except Exception as e:
            print(f"Ошибка инициализации LLM: {e}")
            self.llm_initialized = False

    def _generate_keywords_from_text(self, text: str) -> List[str]:
        """Генерирует ключевые слова из текста с помощью LLM"""
        if not self.llm_initialized or not text:
            return []
            
        try:
            messages = [
                {"role": "system", "content": "Ты - эксперт по юридической терминологии. Извлеки ключевые слова из текста."},
                {"role": "user", "content": f"{KEYWORDS_PROMPT}\n\nТекст для анализа:\n{text[:10000]}"}
            ]
            
            response = self.llm_client.query(messages, TEMPERATURE, MAX_ANSWER_LENGTH)
            
            # Обработка ответа - извлекаем только ключевые слова
            keywords = []
            for line in response.split('\n'):
                # Удаляем маркеры списка и лишние символы
                clean_line = re.sub(r'^[\d\-•>*]+', '', line).strip()
                if clean_line:
                    # Разбиваем на отдельные слова/фразы
                    words = re.findall(r'[\w\-]+(?:\s+[\w\-]+)*', clean_line.lower())
                    keywords.extend(words)
            
            # Удаляем дубликаты и слишком короткие слова
            keywords = list(set(k for k in keywords if len(k) > 2))
            return keywords[:20]  # Ограничиваем количество ключевых слов
            
        except Exception as e:
            print(f"Ошибка генерации ключевых слов: {e}")
            return []

    def analyze_document(self, prompt_type: str) -> str:
        """Анализирует документ с использованием LLM"""
        if not self.current_docx:
            return "Пожалуйста, загрузите DOCX файл"
            
        docx_text = self.current_docx["content"]
        
        # Генерация ключевых слов из текста документа
        if not self.search_engine.llm_keywords:
            with st.spinner("Генерация ключевых слов..."):
                self.search_engine.llm_keywords = self._generate_keywords_from_text(docx_text)
                st.sidebar.info(f"Сгенерированные ключевые слова: {', '.join(self.search_engine.llm_keywords)}")
        
        # Поиск ТОЛЬКО по ключевым словам
        chunks = self.search_engine.search(self.search_engine.llm_keywords)
        context = self._build_context(docx_text, chunks)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BUTTON_PROMPTS[prompt_type] + f"\n\nКОНТЕКСТ:\n" + context}
        ]

        return self.llm_client.query(messages, TEMPERATURE, MAX_ANSWER_LENGTH)

    def load_documents(self, uploaded_files) -> None:
        """Загружает DOCX файлы"""
        if not uploaded_files:
            return

        try:
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.size == 0:
                        st.warning(f"Файл {uploaded_file.name} пуст")
                        continue

                    file_bytes = io.BytesIO(uploaded_file.read())
                    doc = Document(file_bytes)
                    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

                    if not text:
                        st.warning(f"Файл {uploaded_file.name} не содержит текста")
                        continue

                    if len(text) > MAX_CONTEXT_LENGTH:
                        text = text[:MAX_CONTEXT_LENGTH]
                        st.warning(f"Документ {uploaded_file.name} обрезан до {MAX_CONTEXT_LENGTH} символов")

                    self.current_docx = {
                        "name": uploaded_file.name,
                        "content": text
                    }

                    st.success(f"Документ {uploaded_file.name} загружен для анализа")

                except Exception as e:
                    st.error(f"Ошибка обработки файла {uploaded_file.name}: {str(e)}")
                    continue

        except Exception as e:
            st.error(f"Общая ошибка при загрузке документов: {str(e)}")

    def _build_context(self, docx_text: str, chunks: List[Dict]) -> str:
        """Строит контекст для LLM из DOCX и найденных фрагментов"""
        context_parts = [
            "=== ЗАГРУЖЕННЫЙ ДОКУМЕНТ ===",
            docx_text,
            "\n=== РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ ИЗ БАЗЫ ЗНАНИЙ ===",
            chunks,
        ]
        
        for chunk in chunks:
            context_parts.append(f"\n📄 {chunk['doc_name']} (релевантность: {chunk['score']:.2f}):")
            context_parts.append(chunk['chunk_text'][:3000])
        
        return "\n".join(context_parts)

def main():
    st.set_page_config(page_title="El Documente", layout="wide", initial_sidebar_state="collapsed")
    st.title("El Documente: проверьте свой процессуальный документ")
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DocumentAnalyzer(API_URL, API_KEY)
    
    analyzer = st.session_state.analyzer
    
    if not analyzer.llm_initialized:
        st.sidebar.error("LLM не инициализирован. Проверьте API ключ и URL")
    
    st.header("Загрузка документа")
    uploaded_files = st.file_uploader(
        "Выберите документ в формате DOCX", 
        type=["docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Обработка документов..."):
            analyzer.load_documents(uploaded_files)
        st.success(f"Загружено документов: {len(uploaded_files)}")

    st.header("Анализ документа")
    col1, col2, col3 = st.columns(3)

    buttons_disabled = not (uploaded_files and analyzer.llm_initialized)
    
    with col1:
        if st.button("Оценить качество документа", disabled=buttons_disabled):
            with st.spinner("Анализ документа..."):
                result = analyzer.analyze_document("quality")
                st.markdown("### Результат оценки")
                st.markdown(result)
    
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
