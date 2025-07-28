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

user_input = st.text_area("Введите ваш вопрос:", height=150, max_chars=600, key="user_input")

if st.button("Отправить", key="send_btn"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    
    st.session_state.last_query = user_input
    
    with st.spinner("Обработка запроса..."):
        # Инициализируем переменные для результатов
        web_search_results = []
        qdrant_chunks = []
        
        # 1. Веб-поиск
        try:
            web_queries = [user_input] + (st.session_state.get('generated_queries', [])[:2])
            for query in web_queries[:3]:  # Не более 3 запросов
                results = web_searcher.perform_search(query, max_results=2)
                if results:
                    web_search_results.extend(results)
            
            st.session_state.web_search_results = web_search_results
        except Exception as e:
            st.error(f"Ошибка веб-поиска: {str(e)}")
            st.session_state.web_search_results = []

        # 2. Поиск в Qdrant
        try:
            top_k = st.session_state.get('qdrant_top_k', 10)
            balance = st.session_state.get('search_balance', 0.5)
            
            qdrant_chunks = st.session_state.data_processor.enhance_with_qdrant_search(
                user_input,
                top_k=top_k,
                keyword_weight=balance
            )
            st.session_state.qdrant_chunks = qdrant_chunks
        except Exception as e:
            st.error(f"Ошибка поиска в Qdrant: {str(e)}")
            st.session_state.qdrant_chunks = []

        # 3. Формирование контекста для LLM
        context_parts = []
        
        if web_search_results:
            context_parts.append("Веб-результаты:\n" + "\n\n".join(
                [f"Источник {i+1} ({res['title']}):\n{res['snippet']}\n{res['full_content'][:2000]}"
                 for i, res in enumerate(web_search_results[:3])]
            ))
        
        if qdrant_chunks:
            context_parts.append("База знаний:\n" + "\n\n".join(qdrant_chunks[:5]))
        
        if st.session_state.get('document_text'):
            context_parts.append("Загруженный документ:\n" + st.session_state.document_text[:5000])
        
        full_context = "\n\n".join(context_parts)[:15000]  # Ограничение контекста

        # 4. Отправка запроса в LLM
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
            
            if 'candidates' in response_data and response_data['candidates']:
                answer = response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                answer = "Не удалось получить ответ от API"
            
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
        for i, result in enumerate(st.session_state.web_search_results[:3]):
            with st.expander(f"{i+1}. {result.get('title', 'Без названия')[:50]}...", expanded=False):
                st.markdown(f"**URL:** [{result.get('url', '#')[:30]}...]({result.get('url', '#')})")
                st.markdown("**Сниппет:**")
                st.info(result.get('snippet', 'Нет описания')[:200] + "...")
                
                if result.get('full_content'):
                    st.text_area(
                        "Полный текст", 
                        value=result['full_content'][:3000], 
                        height=200,
                        key=f"web_content_{i}"
                    )
    else:
        st.info("Нет результатов веб-поиска")

    # Блок результатов из Qdrant
    if st.session_state.get('qdrant_chunks'):
        st.subheader("🔍 Релевантные фрагменты")
        for i, chunk in enumerate(st.session_state.qdrant_chunks[:5]):
            st.text_area(
                f"Фрагмент {i+1}", 
                value=chunk[:2000], 
                height=150,
                key=f"qdrant_chunk_{i}"
            )
    else:
        st.info("Нет результатов из базы знаний")
    

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
