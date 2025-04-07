# -*- coding: utf-8 -*-
import streamlit as st
import json
import os
import re
import pickle
import io  # Добавлен отсутствующий импорт
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
MAX_CONTEXT_LENGTH = 10000
MAX_ANSWER_LENGTH = 15000
TEMPERATURE = 0.2
os.makedirs(DATA_DIR, exist_ok=True)

# Системные промпты
SYSTEM_PROMPT = """Ты - профессиональный опытный юрист-литигатор из энергетической компании. Ты можешь критически оценивать
процессуальные документы, комментировать позиции и прогнозировать развитие споров. правила цитирования: ЗАПРЕЩЕНО цитировать и ссылаться на конкретные нормы, пункты, правила и конкретные законы,
отсутствующие в предоставленных документах. Все выводы и цитаты должны основываться на тексте предоставленных документов."""

BUTTON_PROMPTS = {
    "quality": """Ты юрист-литигатор в энергетической компании Т Плюс. Оцени сильные и слабые стороны процессуального документа НА ОСНОВЕ ЭТИХ ФРАГМЕНТОВ:\n{context}\n\n. Правила анализа: 
1. Полноту, последовательность и структуру аргументации, 
2. относимость и достаточность доказательств, 
3. полноту указания в подкрепление аргументов применимых норм права. 
4. суммарное количество недостатков в формате оценки от 1 до 10 ["Я бы поставил итоговую оценку документу ... из 10]
5. правила цитирования: ЗАПРЕЩЕНО цитировать и ссылаться на конкретные нормы и конкретные законы, отсутствующие в предоставленных документах. Все выводы и цитаты должны основываться на тексте предоставленных документов.
6.[ВАЖНО] критично оцени документ на предмет соответствия методическим рекомендациям Т Плюс из фрагментов  \n{context}\n\n по подготовке процессуальных документов (если рекомендации представлены в документе), при выявлении нарушений - процитируй нарушенные требования рекомендаций Т Плюс, если требования указаны в предоставленных документах.""",

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
        self.cache_path = os.path.join(DATA_DIR, "bm25_index.pkl")  # Добавим путь к кешу

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
        print(f"Индексация документов: {self.chunks_info}")  # Отладочное сообщение
        tokenized_corpus = [self.preprocessor.preprocess(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_index_loaded = True
        self.save_to_cache()  # Автоматически сохраняем индекс после построения

    def load_from_cache(self) -> bool:
        try:
            if not os.path.exists(self.cache_path):
                return False

            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25, self.chunks_info, self.doc_index = data
                self.is_index_loaded = True
                return True
        except Exception as e:
            print(f"Ошибка загрузки кеша: {e}")
            return False

    def save_to_cache(self) -> None:
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.bm25, self.chunks_info, self.doc_index), f)
        except Exception as e:
            print(f"Ошибка сохранения кеша: {e}")

    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        if not self.is_index_loaded:
            if not self.load_from_cache():  # Попробуем загрузить из кеша
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
                "model": "google/gemini-2.0-flash-lite-001", #"qwen/qwq-32b",
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
        self.knowledge_base = []  # Документы из knowledge_base.json для поиска BM25
        self.current_docx = None  # Текущий загруженный DOCX-документ
        self.llm_client = None
        self.llm_initialized = False
        
        # Загружаем базу знаний при инициализации
        self._load_knowledge_base()
        
        # Инициализируем LLM
        self._initialize_llm(api_url, api_key)  # Убрали условие, так как проверка внутри метода

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

    def analyze_document(self, prompt_type: str) -> str:
        """Анализирует документ с использованием LLM"""
        if not self.current_docx:
            return "Пожалуйста, загрузите DOCX файл"
            
        if not self.knowledge_base:
            return "База знаний пуста (добавьте документы в knowledge_base.json)"
        
        # 1. Получаем текст из загруженного DOCX
        docx_text = self.current_docx["content"]
        
        # 2. Формируем запрос для BM25 на основе текста DOCX
        query = self._generate_search_query(prompt_type, docx_text)
        
        # 3. Ищем релевантные фрагменты в базе знаний
        chunks = self.search_engine.search(query)
        
        # 4. Формируем контекст для LLM
        context = self._build_context(docx_text, chunks)
        
        # 5. Формируем и выполняем запрос к LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BUTTON_PROMPTS[prompt_type] + "\n\nКОНТЕКСТ:\n" + context}
        ]

        # Выводим итоговый запрос в сайдбар
        st.sidebar.header("Итоговый запрос к LLM")
        st.sidebar.markdown("### Запрос пользователя:")
        st.sidebar.markdown(BUTTON_PROMPTS[prompt_type] + "\n\nКОНТЕКСТ:\n" + context)
        
        return self.llm_client.query(messages, TEMPERATURE, MAX_ANSWER_LENGTH)

    def _load_knowledge_base(self) -> None:
        """Загружает базу знаний из JSON файла для BM25 поиска"""
        json_path = os.path.join(DATA_DIR, "knowledge_base.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Загруженные данные: {data}")  # Отладочное сообщение

                # Убедитесь, что data является словарем
                if isinstance(data, dict):
                    # Извлекаем документы
                    documents = data.get("object", {}).get("documents", [])
                    print(f"Найденные документы: {documents}")  # Отладочное сообщение

                    if isinstance(documents, list) and documents:
                        # Преобразуем документы в нужный формат для BM25
                        formatted_documents = []
                        for doc in documents:
                            for chunk in doc.get("chunks", []):
                                formatted_documents.append({
                                    "name": doc.get("source_file", "Без названия"),
                                    "content": chunk.get("chunk_text", "")
                                })
                        if formatted_documents:
                            self.search_engine.build_index(formatted_documents)
                            print(f"Загружено {len(formatted_documents)} фрагментов из базы знаний")
                        else:
                            st.error("Ошибка: Не удалось извлечь фрагменты из документов.")
                    else:
                        st.error("Ошибка: База знаний должна содержать список документов.")
                        print(f"Тип данных documents: {type(documents)}")  # Проверка типа
                        print(f"Содержимое documents: {documents}")  # Проверка содержимого
                else:
                    st.error("Ошибка: Загруженные данные не являются словарем.")
            except Exception as e:
                print(f"Ошибка загрузки базы знаний: {e}")

    def load_documents(self, uploaded_files) -> None:
        """Загружает и обрабатывает DOCX файл (без сохранения в базу знаний)"""
        if not uploaded_files:
            return

        try:
            documents = []  # Список для хранения загруженных документов
            for uploaded_file in uploaded_files:  # Обработка каждого загруженного файла
                if uploaded_file.size == 0:  # Проверка на пустой файл
                    st.warning(f"Файл {uploaded_file.name} пуст")
                    continue

                file_bytes = io.BytesIO(uploaded_file.read())  # Используем read() вместо getvalue()
                doc = Document(file_bytes)
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

                if not text:
                    st.warning(f"Файл {uploaded_file.name} не содержит текста")
                    continue

                if len(text) > MAX_CONTEXT_LENGTH:
                    text = text[:MAX_CONTEXT_LENGTH]
                    st.warning(f"Документ {uploaded_file.name} обрезан до {MAX_CONTEXT_LENGTH} символов")

                # Добавляем документ в список
                documents.append({
                    "name": uploaded_file.name,  # Имя документа
                    "content": text              # Содержимое документа
                })

                # Обновляем current_docx для последнего загруженного файла
                self.current_docx = {
                    "name": uploaded_file.name,
                    "content": text
                }

                st.success(f"Документ {uploaded_file.name} загружен для анализа")

            # Индексируем загруженные документы
            if documents:
                self.search_engine.build_index(documents)  # Передаем список документов в индекс

        except Exception as e:
            st.error(f"Ошибка обработки файла: {str(e)}")
        
    def analyze_document(self, prompt_type: str) -> str:
        """Анализирует документ с использованием LLM"""
        if not self.current_docx:
            return "Пожалуйста, загрузите DOCX файл"
            
        if not self.knowledge_base:
            return "База знаний пуста (добавьте документы в knowledge_base.json)"
        
        # 1. Получаем текст из загруженного DOCX
        docx_text = self.current_docx["content"]
        
        # 2. Формируем запрос для BM25 на основе текста DOCX
        query = self._generate_search_query(prompt_type, docx_text)
        
        # 3. Ищем релевантные фрагменты в базе знаний
        chunks = self.search_engine.search(query)
        
        # 4. Формируем контекст для LLM
        context = self._build_context(docx_text, chunks)
        
        # 5. Формируем и выполняем запрос к LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BUTTON_PROMPTS[prompt_type] + "\n\nКОНТЕКСТ:\n" + context}
        ]

        # Выводим итоговый запрос в сайдбар
        st.sidebar.header("Итоговый запрос к LLM")
        st.sidebar.markdown("### Запрос пользователя:")
        st.sidebar.markdown(BUTTON_PROMPTS[prompt_type] + "\n\nКОНТЕКСТ:\n" + context)
        
        return self.llm_client.query(messages, TEMPERATURE, MAX_ANSWER_LENGTH)

    def _generate_search_query(self, prompt_type: str, docx_text: str) -> str:
        """Генерирует поисковый запрос для BM25"""
        base_queries = {
            "quality": "оценка качества документа структура аргументации доказательства нормы права",
            "strategy": "стратегия спора доказательства процессуальное поведение",
            "prediction": "позиция второй стороны прогнозирование аргументы оппонента"
        }
        
        # Комбинируем базовый запрос с текстом DOCX
        return f"{base_queries[prompt_type]} {docx_text[:1000]}"

    def _build_context(self, docx_text: str, chunks: List[Dict]) -> str:
        """Строит контекст для LLM из DOCX и найденных фрагментов"""
        context_parts = [
            "=== ЗАГРУЖЕННЫЙ ДОКУМЕНТ ===",
            docx_text,
            "\n=== РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ ИЗ БАЗЫ ЗНАНИЙ ==="
        ]
        
        for chunk in chunks:
            context_parts.append(f"\n📄 {chunk['doc_name']} (релевантность: {chunk['score']:.2f}):")
            context_parts.append(chunk['chunk_text'][:1000])
        
        return "\n".join(context_parts)

def main():
    st.set_page_config(page_title="El Documente", layout="wide")
    st.title("El Documente: проверьте свой процессуальный документ")
    
    # Инициализация анализатора
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DocumentAnalyzer(API_URL, API_KEY)  # Передаем параметры при создании
    
    analyzer = st.session_state.analyzer
    
    st.sidebar.header("Настройки поиска")
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        weight = st.slider(
            "Вес контента документа",
            0.1, 2.0, 0.7, 0.1,
            key="doc_weight_slider",
            help="Регулирует влияние текста документа на результаты поиска"
        )

    with col2:
        st.metric("Текущее значение", f"{weight:.1f}")

    # Для отладки (можно убрать в продакшене)
    st.sidebar.write(f"Выбрано значение: {weight}")
    
    # Проверка инициализации LLM
    if not analyzer.llm_initialized:
        st.sidebar.error("LLM не инициализирован. Проверьте API ключ и URL")
    
    # Загрузка документов
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
