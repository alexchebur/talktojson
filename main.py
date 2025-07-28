import os
import streamlit as st
import time
import requests
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import initialize_session, setup_logging, file_to_text, clean_text
from config import GEMINI_API_KEY, API_URL, API_TIMEOUT
from web_search import WebSearcher
from query_generator import QueryGenerator
from indexing import IndexBuilder
from processing import DataProcessor
from prompts import get_prompt
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

logger = setup_logging()
initialize_session()

if "context_parts" not in st.session_state:
    st.session_state.context_parts = []

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "data", "embeddings")
DOCUMENTS_DIR = os.path.join(REPO_ROOT, "documents")

# Создаем необходимые директории
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# Инициализация компонентов
index_builder = IndexBuilder()
data_processor = DataProcessor(index_builder)
query_generator = QueryGenerator()
web_searcher = WebSearcher()

if 'data_processor' not in st.session_state:
    st.session_state.index_builder = IndexBuilder()
    st.session_state.data_processor = DataProcessor(st.session_state.index_builder)
    st.session_state.query_generator = QueryGenerator()
    st.session_state.web_searcher = WebSearcher()

# Перед веб-поиском добавьте:
if 'generated_queries' not in st.session_state:
    st.session_state.generated_queries = []

# Генерация запросов
try:
    keywords = st.session_state.data_processor.extract_keywords(user_input)
    st.session_state.generated_queries = query_generator.generate(user_input, keywords)
    st.write(f"Сгенерировано запросов: {len(st.session_state.generated_queries)}")  # Для отладки
except Exception as e:
    st.error(f"Ошибка генерации запросов: {str(e)}")
    st.session_state.generated_queries = []


# Инициализация веб-поиска
if "web_searcher" not in st.session_state:
    st.session_state.web_searcher = web_searcher

# Загрузка полного индекса (если нужно)
#if not index_builder.load_full_index():
#    print("Полный индекс не найден, будет построен при обработке документа")

# Интерфейс приложения
st.title("ИИ-помощник по подготовке правовых заключений")
uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])


# Убрана инициализация BM25
if "index_builder" not in st.session_state:
    st.session_state.index_builder = IndexBuilder()
    st.session_state.data_processor = DataProcessor(st.session_state.index_builder)

# Обработка загруженного файла
if uploaded_file:
    with st.spinner("Анализ документа..."):
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        
        # Сохраняем текст документа для контекста
        st.session_state.document_text = clean_text(file_text)[:10000]  # Первые 10k символов



if st.button("Отправить", key="send_btn"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    
    st.session_state.last_query = user_input
    
    with st.spinner("Обработка запроса..."):
        # 1. Инициализация переменных
        if 'web_search_results' not in st.session_state:
            st.session_state.web_search_results = []
        if 'qdrant_chunks' not in st.session_state:
            st.session_state.qdrant_chunks = []
        if 'chat_log' not in st.session_state:
            st.session_state.chat_log = ""

        # 2. Веб-поиск
        try:
            queries = [user_input]
            if 'generated_queries' in st.session_state:
                queries.extend(st.session_state.generated_queries[:2])
            
            st.session_state.web_search_results = []
            for query in queries[:3]:  # Максимум 3 запроса
                results = web_searcher.perform_search(query, max_results=2)
                if results:
                    # Добавляем query к каждому результату
                    for res in results:
                        res['query'] = query  # Гарантируем наличие ключа 'query'
                    st.session_state.web_search_results.extend(results)
        except Exception as e:
            st.error(f"Ошибка веб-поиска: {str(e)}")
            st.session_state.web_search_results = []

        # 3. Поиск в Qdrant
        try:
            top_k = st.session_state.get('qdrant_top_k', 10)
            balance = st.session_state.get('search_balance', 0.5)
            
            st.session_state.qdrant_chunks = st.session_state.data_processor.enhance_with_qdrant_search(
                query=user_input,
                top_k=top_k,
                keyword_weight=balance
            ) or []  # Гарантируем список, даже если None
        except Exception as e:
            st.error(f"Ошибка поиска в Qdrant: {str(e)}")
            st.session_state.qdrant_chunks = []

        # 4. Формирование контекста
        context_parts = []
        
        # Веб-результаты
        if st.session_state.web_search_results:
            web_context = []
            for i, res in enumerate(st.session_state.web_search_results[:3]):
                web_context.append(
                    f"Источник {i+1} ({res.get('title', 'Без названия')}):\n"
                    f"URL: {res.get('url', '')}\n"
                    f"{res.get('snippet', 'Нет описания')}\n"
                    f"{res.get('full_content', '')[:2000]}"
                )
            context_parts.append("Веб-результаты:\n" + "\n\n".join(web_context))
        
        # Qdrant результаты
        if st.session_state.qdrant_chunks:
            context_parts.append("База знаний:\n" + "\n\n".join(
                chunk[:2000] for chunk in st.session_state.qdrant_chunks[:5]
            ))
        
        # Загруженный документ
        if 'document_text' in st.session_state and st.session_state.document_text:
            context_parts.append("Загруженный документ:\n" + st.session_state.document_text[:5000])
        
        full_context = "\n\n".join(context_parts)[:15000]

        # 5. Отправка в LLM
        try:
            prompt = get_prompt("system", {
                "user_query": user_input,
                "context": full_context
            })
            
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 5000
                    }
                },
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            
            answer = (
                response_data['candidates'][0]['content']['parts'][0]['text']
                if 'candidates' in response_data and response_data['candidates']
                else "Не удалось получить ответ от API"
            )
            
            st.session_state.llm_response = answer
            st.session_state.chat_log += f"\nПользователь: {user_input}\nАссистент: {answer}"
            
        except Exception as e:
            st.error(f"Ошибка API: {str(e)}")

# Основной ответ
if st.session_state.get('llm_response'):
    st.subheader("Ответ ассистента:")
    st.markdown(st.session_state.llm_response)

# Сайдбар с дополнительной информацией
with st.sidebar:
    st.subheader("Результаты поиска")
    
    # Блок веб-результатов
    if st.session_state.get('web_search_results'):
        st.subheader("🌐 Веб-результаты")
        
        # Разделяем результаты по источникам
        main_query_results = [
            r for r in st.session_state.web_search_results 
            if r.get('query', '') == st.session_state.get('last_query', '')
        ]
        
        if main_query_results:
            st.caption(f"По основному запросу: '{st.session_state.last_query}'")
            for i, res in enumerate(main_query_results[:3]):
                with st.expander(f"{i+1}. {res.get('title', 'Без названия')[:50]}...", expanded=False):
                    st.markdown(f"**URL:** [{res.get('url', '')[:30]}...]({res.get('url', '')})")
                    st.markdown("**Сниппет:**")
                    st.info(res.get('snippet', 'Нет описания')[:200] + "...")
                    if res.get('full_content'):
                        st.text_area(
                            "Полный текст", 
                            value=res['full_content'][:3000], 
                            height=200,
                            key=f"main_web_{i}"
                        )

with st.sidebar:
    # ... остальные блоки ...
    
    # Блок сгенерированных запросов
    if st.session_state.get('generated_queries'):
        st.subheader("🔍 Сгенерированные запросы")
        st.write("Эти запросы были автоматически созданы для уточнения поиска:")
        for i, query in enumerate(st.session_state.generated_queries[:3]):
            st.code(f"{i+1}. {query}")
            
        # Результаты по сгенерированным запросам
        gen_results = [
            r for r in st.session_state.get('web_search_results', [])
            if r.get('query_type') == 'generated'
        ]
        
        if gen_results:
            st.subheader("📌 Результаты по уточняющим запросам")
            for i, res in enumerate(gen_results[:4]):
                with st.expander(f"Запрос: '{res.get('query', '')}'", expanded=False):
                    st.markdown(f"**URL:** [{res.get('url', '')[:30]}...]({res.get('url', '')})")
                    st.markdown("**Сниппет:**")
                    st.info(res.get('snippet', 'Нет описания')[:200] + "...")
    
    # Блок результатов из Qdrant
    if st.session_state.get('qdrant_chunks'):
        st.subheader("📚 База знаний (Qdrant)")
        for i, chunk in enumerate(st.session_state.qdrant_chunks[:5]):
            st.text_area(
                f"Фрагмент {i+1}", 
                value=chunk[:2000], 
                height=150,
                key=f"qdrant_chunk_{i}"
            )
    else:
        st.info("Нет результатов из базы знаний")

    # Остальные блоки (документ, настройки, история)...
    

    # Блок загруженного документа
    if st.session_state.get('document_text'):
        st.subheader("📄 Загруженный документ")
        with st.expander("Показать текст документа", expanded=False):
            st.text_area(
                "Содержание документа",
                value=st.session_state.document_text[:5000],
                height=300,
                key="uploaded_doc_preview"
            )

    # Настройки поиска
    with st.expander("⚙️ Настройки поиска", expanded=False):
        st.slider(
            "Количество фрагментов из Qdrant",
            min_value=3,
            max_value=15,
            value=10,
            key="qdrant_top_k"
        )
        st.slider(
            "Баланс семантика/текст",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 - только семантика, 1 - только текст",
            key="search_balance"
        )

    # История диалога
    if st.session_state.get('chat_log'):
        st.subheader("История диалога")
        st.text_area(label="История", 
                    value=st.session_state.chat_log, 
                    height=300, 
                    key="chat_history_unique",
                    disabled=True)
