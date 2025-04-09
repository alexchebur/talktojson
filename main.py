# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import pickle
import io
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

# Конфигурация
DATA_DIR = "data"
MAX_CONTEXT_LENGTH = 15000
MAX_ANSWER_LENGTH = 15000
TEMPERATURE = 0.2
os.makedirs(DATA_DIR, exist_ok=True)

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
        self.cache_path = os.path.join(DATA_DIR, "bm25_index.pkl")
        
        # Пытаемся загрузить индекс сразу при инициализации
        self._load_index()

    def _load_index(self) -> bool:
        """Загружает индекс из файла bm25_index.pkl"""
        try:
            if not os.path.exists(self.cache_path):
                st.error(f"Файл индекса {self.cache_path} не найден")
                return False

            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)
                
                if isinstance(data, dict):
                    self.bm25 = BM25Okapi([item['processed'] for item in data.get('metadata', [])])
                    self.chunks_info = data.get('metadata', [])
                    self.doc_index = defaultdict(list)  # Если у вас нет doc_index, просто инициализируйте его
                    self.is_index_loaded = True
                    st.success("Индекс успешно загружен")
                    return True
                else:
                    st.error("Неправильный формат файла индекса")
                    return False
                    
        except Exception as e:
            st.error(f"Ошибка загрузки индекса: {str(e)}")
            return False

    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        """Выполняет поиск по индексу"""
        if not self.is_index_loaded:
            st.error("Индекс не загружен")
            return []

        if not query or not isinstance(query, str):
            return []

        try:
            # Препроцессинг запроса
            tokens = self.preprocessor.preprocess(query)
            if not tokens:
                return []

            # Получаем оценки релевантности
            scores = self.bm25.get_scores(tokens)
            
            # Сортируем результаты по релевантности
            best_indices = np.argsort(scores)[-top_n:][::-1]

            # Формируем результаты
            results = []
            for idx in best_indices:
                if idx < len(self.chunks_info):
                    result = {
                        'doc_id': self.chunks_info[idx].get('file_id', ''),
                        'doc_name': self.chunks_info[idx].get('file_id', 'Без названия'),
                        'chunk_text': self.chunks_info[idx].get('original', '')[:1000],  # Измените на 'original'
                        'score': float(scores[idx])
                    }
                    results.append(result)
            
            return results

        except Exception as e:
            st.error(f"Ошибка при выполнении поиска: {str(e)}")
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
        self.current_docx = None  # Текущий загруженный DOCX-документ
        self.llm_client = None
        self.llm_initialized = False
        
        # Инициализируем LLM
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

    def analyze_document(self, prompt_type: str) -> str:
        """Анализирует документ с использованием LLM"""
        if not self.current_docx:
            return "Пожалуйста, загрузите DOCX файл"
            
        # 1. Получаем текст из загруженного DOCX
        docx_text = self.current_docx["content"]
        
        # 2. Формируем запрос для BM25 на основе текста DOCX
        query = self._generate_search_query(prompt_type, docx_text)
        
        # 3. Ищем релевантные фрагменты в индексе BM25
        chunks = self.search_engine.search(query)
        
        # 4. Формируем контекст для LLM
        context = self._build_context(docx_text, chunks)
        
        # 5. Формируем и выполняем запрос к LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BUTTON_PROMPTS[prompt_type] + f"\n\nВес контента: {st.session_state.doc_weight_slider:.1f}\n\nКОНТЕКСТ:\n" + context}
        ]

        # Выводим итоговый запрос в сайдбар
        st.sidebar.header("Итоговый запрос к LLM")
        st.sidebar.markdown("### Запрос пользователя:")
        st.sidebar.markdown(BUTTON_PROMPTS[prompt_type] + f"\n\nВес контента: {st.session_state.doc_weight_slider:.1f}\n\nКОНТЕКСТ:\n" + context)
        
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

                    # Сохраняем только последний загруженный документ
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
            context_parts.append(chunk['chunk_text'][:3000])
        
        return "\n".join(context_parts)

def main():
    st.set_page_config(page_title="El Documente", layout="wide")
    gif_path = "data/maracas-sombrero-hat.gif"
    st.image(gif_path, caption="Hola!", width=64)
    st.title("El Documente: проверьте свой процессуальный документ")
    
    # Инициализация анализатора
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DocumentAnalyzer(API_URL, API_KEY)
    
    analyzer = st.session_state.analyzer
    
    st.sidebar.header("Настройки поиска")
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        weight = st.slider(
            "Вес контента документа",
            0.1, 1.0, 0.7, 0.1,
            key="doc_weight_slider",
            help="Регулирует влияние текста документа на результаты поиска"
        )

    with col2:
        st.metric("Текущее значение", f"{weight:.1f}")

    st.sidebar.write(f"Выбрано значение: {weight}")
    
    # Проверка инициализации LLM
    if not analyzer.llm_initialized:
        st.sidebar.error("LLM не инициализирован. Проверьте API ключ и URL")
    
    # Загрузка документов
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

    # Настройки чата
    CHAT_TEMPERATURE = 0.6
    CHAT_SYSTEM_PROMPT = """Ты - опытный дружелюбный юрист энергетической компании, отвечающий на правовые вопросы. ЗАПРЕЩЕНО:  1. ссылаться на выдуманные законы и судебную практику. 2. указывать в ответе, что ты ознакомился с документом, просто поддерживай диалог. Ответы излагай в деловом стиле, без категорических мнений."""

    # Чат с наставником Карлосом
    st.header("Чат")

    user_input = st.text_area(
        "Обсудить с наставником Карлосом",
        max_chars=500,
        height=100
    )

    ask_button = st.button("Спросить", disabled=not (uploaded_files and user_input))

    if 'docx_added' not in st.session_state:
        st.session_state.docx_added = False

    if ask_button:
        conversation_log = []

        if not st.session_state.docx_added and analyzer.current_docx:
            conversation_log.append(analyzer.current_docx["content"])
            st.session_state.docx_added = True

        conversation_log.append(user_input)

        relevant_chunks = analyzer.search_engine.search(user_input)

        for chunk in relevant_chunks:
            conversation_log.append(chunk['chunk_text'])

        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(conversation_log)}
        ]

        response = analyzer.llm_client.query(messages, CHAT_TEMPERATURE, MAX_ANSWER_LENGTH)

        response_container = st.empty()
        response_container.text_area("Ответ от Карлоса", value=response, height=200, disabled=True)

    # Кнопки анализа
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
