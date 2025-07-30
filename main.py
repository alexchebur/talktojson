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

# Фикс для конфликта Streamlit и PyTorch
try:
    from fix_torch import fix_torch_classes
    fix_torch_classes()
except ImportError:
    pass

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



# Инициализация веб-поиска
if "web_searcher" not in st.session_state:
    st.session_state.web_searcher = web_searcher

def parse_stage1_response(response: str) -> dict:
    """Парсинг структурированного текстового ответа"""
    result = {
        "problem_formulation": "",
        "expanded_queries": [],
        "expanded_keywords": []
    }
    
    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("Проблема:"):
            result["problem_formulation"] = line.replace("Проблема:", "").strip()
        elif line.startswith("Вопросы:"):
            continue
        elif re.match(r'^\d+\.', line):
            result["expanded_queries"].append(re.sub(r'^\d+\.\s*', '', line).strip())
        elif line.startswith("Ключевые слова:"):
            keywords = line.replace("Ключевые слова:", "").strip()
            result["expanded_keywords"] = [k.strip() for k in keywords.split(',')]
    
    return result

# Интерфейс приложения
#st.title("ИИ-помощник по подготовке правовых заключений")
#uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
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



# ... (импорты и инициализация остаются без изменений) ...

# Новая функция для вызова Gemini API
def call_gemini_api(prompt: str, temperature=0.3, max_output_tokens=5000) -> str:
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens
                }
            },
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        st.error(f"Ошибка API: {str(e)}")
        return ""



# Инициализация ввода пользователя
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Поле для ввода вопроса пользователя
st.session_state.user_input = st.text_area(
    "Введите ваш вопрос:", 
    value=st.session_state.user_input,
    height=150,
    max_chars=600,
    key="user_input_area"
)

if st.button("Отправить", key="send_btn"):
    # Получаем ввод пользователя напрямую из session_state
    user_input = st.session_state.user_input.strip()
    
    if not user_input:
        st.error("Введите текст вопроса")
        st.stop()
    st.session_state.last_query = user_input
    
    # Этап 1: Генерация структурированных данных для поиска
    stage1_prompt = get_prompt("stage1", {"user_query": user_input})
    stage1_response = call_gemini_api(stage1_prompt)
    try:
        search_data = parse_stage1_response(stage1_response)
        st.session_state.search_data = search_data
    except Exception as e:
        st.error(f"Ошибка разбора ответа: {str(e)}")
        st.text_area("Ответ модели для отладки:", value=stage1_response)
        st.stop()
    
    # Этап 2: Поиск информации
    with st.spinner("Поиск информации..."):
        # Веб-поиск по сгенерированным запросам
        web_results = []
        for query in search_data['expanded_queries']:
            web_results.extend(web_searcher.perform_search(query, max_results=2))
        
        # Поиск в Qdrant
        qdrant_results = []
        # Семантический поиск по основному запросу и расширенным
        for query in [user_input] + search_data['expanded_queries']:
            qdrant_results.extend(index_builder.semantic_search(query, top_k=3))
        # Полнотекстовый поиск по ключевым словам
        qdrant_results.extend(index_builder.keyword_search(search_data['expanded_keywords'], top_k=5))
        
        # Формирование контекста
        context_parts = [
            f"Проблема: {search_data['problem_formulation']}",
            "Веб-результаты:"
        ]
        
        for i, res in enumerate(web_results[:5]):
            context_parts.append(f"{i+1}. [{res['title']}]({res['url']}): {res['snippet']}")
        
        context_parts.append("Базовые знания:")
        for i, res in enumerate(qdrant_results[:10]):
            context_parts.append(f"{i+1}. {res['text'][:500]}...")
        
        full_context = "\n\n".join(context_parts)[:15000]
    
    # Этап 3: Генерация проекта заключения
    with st.spinner("Подготовка проекта заключения..."):
        stage2_prompt = get_prompt("stage2", {
            "user_query": user_input,
            "problem_formulation": search_data['problem_formulation'],
            "context": full_context
        })
        stage2_response = call_gemini_api(stage2_prompt, max_output_tokens=10000)
        
        try:
            opinion_data = json.loads(stage2_response)
        except:
            st.error("Ошибка разбора JSON на этапе 2")
            st.stop()
    
    # Этап 4: Финальная проверка и оформление
    with st.spinner("Финальная проверка..."):
        stage3_prompt = get_prompt("stage3", {
            "opinion_draft": opinion_data['opinion_draft']
        })
        final_opinion = call_gemini_api(stage3_prompt, max_output_tokens=10000)
        st.session_state.final_opinion = final_opinion

# Вывод результата
if st.session_state.get('final_opinion'):
    st.subheader("Правовое заключение")
    st.markdown(st.session_state.final_opinion)

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
