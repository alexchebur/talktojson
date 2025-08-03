import sys
import os
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHING"] = "false"
os.environ["STREAMLIT_DISABLE_WATCHDOG"] = "true"
from concurrent.futures import ThreadPoolExecutor
import re
import fasttext.util
import fasttext


import time
import requests
import json
import numpy as np
import networkx as nx

from utils import initialize_session, setup_logging, file_to_text, clean_text
from config import GEMINI_API_KEY, API_URL, API_TIMEOUT
from web_search import WebSearcher
from query_generator import QueryGenerator
from indexing import IndexBuilder
from processing import DataProcessor
from prompts import get_prompt
from typing import Dict, List, Tuple, Optional

from pathlib import Path
import nest_asyncio
nest_asyncio.apply()
import streamlit as st



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

def parse_stage2_response(response: str) -> dict:
    """Парсинг текстового ответа этапа 2"""
    result = {
        "reasoning": "",
        "opinion_draft": ""
    }
    
    # Разделяем анализ и заключение
    analysis_part = ""
    draft_part = ""
    
    if "Анализ:" in response:
        parts = response.split("Анализ:", 1)
        if "Проект заключения:" in parts[1]:
            analysis_part, draft_part = parts[1].split("Проект заключения:", 1)
        else:
            analysis_part = parts[1]
    elif "Проект заключения:" in response:
        draft_part = response.split("Проект заключения:", 1)[1]
    
    # Очистка текста
    result["reasoning"] = analysis_part.strip()
    result["opinion_draft"] = draft_part.strip()
    
    # Проверка заполненности
    if not result["opinion_draft"]:
        raise ValueError("Не удалось извлечь проект заключения")
    
    return result
    


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
        time.sleep(1)  # 1 секунда между запросами


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

# ... (предыдущий код)

if st.button("Отправить", key="send_btn"):
    user_input = st.session_state.user_input.strip()
    
    if not user_input:
        st.error("Введите текст вопроса")
        st.stop()
    
    st.session_state.last_query = user_input
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Этап 1: Генерация поисковых запросов
        status_text.text("Анализ вопроса...")
        progress_bar.progress(20)
        
        stage1_prompt = get_prompt("stage1", {"user_query": user_input})
        stage1_response = call_gemini_api(stage1_prompt, temperature=0.5)
        search_data = parse_stage1_response(stage1_response)
        
        # Фильтруем и ограничиваем запросы
        valid_queries = [
            q for q in search_data['expanded_queries'] 
            if q and len(q.split()) <= 10
        ][:3]
        
        st.session_state.search_data = search_data
        st.session_state.generated_queries = valid_queries
        st.session_state.generated_keywords = search_data['expanded_keywords'][:5]

        # Этап 2: Параллельный поиск
        status_text.text("Поиск информации...")
        progress_bar.progress(40)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Веб-поиск
            web_future = executor.submit(
                lambda: [res for q in [user_input] + valid_queries for res in web_searcher.perform_search(q, 2)]
            )
    
            # Гибридный поиск в Qdrant (ИСПРАВЛЕНО - добавляем user_input)
            qdrant_future = executor.submit(
                lambda: index_builder.hybrid_search(
                    queries=[user_input] + valid_queries,  # <- Главное исправление
                    top_k=3
                )
            )
            
            web_results = web_future.result()
            qdrant_results = qdrant_future.result()

        progress_bar.progress(80)
        
        # Сохраняем результаты
        st.session_state.web_search_results = web_results
        st.session_state.qdrant_results = qdrant_results

        # Этап 3: Формирование контекста
        status_text.text("Формирование ответа...")
        context_parts = [
            f"Проблема: {search_data['problem_formulation']}",
            "Веб-результаты:"
        ]
        
        for i, res in enumerate(web_results[:5]):
            context_parts.append(f"{i+1}. [{res['title']}]({res['url']}): {res['snippet']}")
        
        context_parts.append("Базовые знания:")
        for i, res in enumerate(qdrant_results[:10]):
            content = res.get('content') or res.get('text', '')
            context_parts.append(f"{i+1}. {content[:500]}...")
        
        full_context = "\n\n".join(context_parts)[:30000]
        
        progress_bar.progress(100)
        status_text.text("Готово!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Ошибка обработки запроса: {str(e)}")
        st.stop()
    # Этап 3: Генерация проекта заключения
    with st.spinner("Подготовка проекта заключения..."):
        try:
            stage2_prompt = get_prompt("stage2", {
                "user_query": user_input,
                "problem_formulation": search_data['problem_formulation'],
                "context": full_context
            })
        
            stage2_response = call_gemini_api(stage2_prompt, max_output_tokens=10000)
        
            try:
                opinion_data = parse_stage2_response(stage2_response)
                st.session_state.opinion_data = opinion_data
            
                # Для отладки
                with st.expander("Промежуточные результаты"):
                    st.write("**Анализ:**")
                    st.write(opinion_data["reasoning"])
                    st.write("**Проект заключения (фрагмент):**")
                    st.write(opinion_data["opinion_draft"][:1000] + "...")
            except Exception as e:
                st.error(f"Ошибка разбора ответа: {str(e)}")
                st.text_area("Полный ответ модели:", value=stage2_response, height=300)
                st.stop()
        except Exception as e:
            st.error(f"Ошибка при генерации заключения: {str(e)}")
            st.stop()
    
    # Этап 4: Финальная проверка и оформление
    with st.spinner("Финальная проверка..."):
        try:
            stage3_prompt = get_prompt("stage3", {
                "opinion_draft": opinion_data['opinion_draft']
            })
            final_opinion = call_gemini_api(stage3_prompt, max_output_tokens=10000)
            st.session_state.final_opinion = final_opinion
            
            # Отображение поисковых данных после успешной генерации
            if st.session_state.get('search_data'):
                st.subheader("Поисковые данные")
                with st.expander("Показать детали поиска", expanded=False):
                    st.write(f"**Проблема:** {st.session_state.search_data['problem_formulation']}")
                    st.write("**Сгенерированные запросы:**")
                    for i, query in enumerate(st.session_state.search_data['expanded_queries']):
                        st.code(f"{i+1}. {query}")
                    st.write(f"**Ключевые слова:** {', '.join(st.session_state.search_data['expanded_keywords'])}")
                    
        except Exception as e:
            st.error(f"Ошибка при финальной проверке: {str(e)}")
            st.stop()

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
        for i, res in enumerate(st.session_state.web_search_results[:3]):
            with st.expander(f"{i+1}. {res.get('title', 'Без названия')[:50]}...", expanded=False):
                st.markdown(f"**URL:** [{res.get('url', '')[:30]}...]({res.get('url', '')})")
                st.markdown("**Сниппет:**")
                st.info(res.get('snippet', 'Нет описания')[:200] + "...")
    else:
        st.subheader("🌐 Веб-результаты")
        st.info("Нет веб-результатов")

    # В разделе отображения результатов замените текущий код на:

    if st.session_state.get('hybrid_results'):
        st.subheader("🔍 Результаты гибридного поиска")
    
        # Группировка результатов по типу
        dense_results = []
        sparse_results = []
    
        for res in st.session_state.hybrid_results:
            if 'dense' in res.get('vector_name', ''):
                dense_results.append(res)
            elif 'sparse' in res.get('vector_name', ''):
                sparse_results.append(res)
    
        # Вкладки для разных типов результатов
        tab1, tab2 = st.tabs(["Плотные векторы", "Разреженные векторы"])
    
        with tab1:
            if dense_results:
                for i, res in enumerate(dense_results[:5]):
                    with st.expander(f"Результат {i+1} (сходство: {res['score']:.2f})", expanded=False):
                        st.write(f"**Запрос:** {res.get('query', '')}")
                        st.write(f"**Содержание:** {res.get('content', '')[:500]}...")
                        st.write(f"**ID:** {res['id']} | **Score:** {res['score']:.4f}")
            else:
                st.warning("Нет результатов по плотным векторам")
    
        with tab2:
            if sparse_results:
                for i, res in enumerate(sparse_results[:5]):
                    with st.expander(f"Результат {i+1} (сходство: {res['score']:.2f})", expanded=False):
                        st.write(f"**Запрос:** {res.get('query', '')}")
                        st.write(f"**Содержание:** {res.get('content', '')[:500]}...")
                        st.write(f"**ID:** {res['id']} | **Score:** {res['score']:.4f}")
            else:
                st.warning("Нет результатов по разреженным векторам")
    else:
        st.info("Результаты поиска не найдены")
    

    # Блок семантического поиска в Qdrant
    if st.session_state.get('qdrant_semantic_results'):
        st.subheader("🧠 Семантический поиск (Qdrant)")
        for i, res in enumerate(st.session_state.qdrant_semantic_results[:3]):
            with st.expander(f"Сем. результат {i+1} (сходство: {res['score']:.2f})", expanded=False):
                st.write(res['content'][:200] + "...")
    else:
        st.subheader("🧠 Семантический поиск (Qdrant)")
        st.info("Нет результатов семантического поиска")
    

    
    # Заменяем блок полнотекстового поиска
    if st.session_state.get('qdrant_sparse_results'):  # Новое название!
        st.subheader("🔤 Поиск по разреженным векторам (Qdrant)")
        for i, res in enumerate(st.session_state.qdrant_sparse_results[:3]):
            with st.expander(f"Разреженный результат {i+1} (сходство: {res['score']:.2f})", expanded=False):
                st.write(res['content'][:200] + "...")
    else:
        st.subheader("🔤 Поиск по разреженным векторам (Qdrant)")
        st.info("Нет результатов поиска по разреженным векторам")
    
    # Блок сгенерированных запросов
    if st.session_state.get('generated_queries'):
        st.subheader("🔍 Сгенерированные запросы")
        st.write("Эти запросы были автоматически созданы для уточнения поиска:")
        for i, query in enumerate(st.session_state.generated_queries):
            st.code(f"{i+1}. {query}")
    else:
        st.subheader("🔍 Сгенерированные запросы")
        st.info("Нет сгенерированных запросов")
    
    # Блок ключевых слов
    if st.session_state.get('generated_keywords'):
        st.subheader("🔑 Ключевые слова")
        st.write("Ключевые термины, использованные для поиска:")
        st.write(", ".join(st.session_state.generated_keywords))
    else:
        st.subheader("🔑 Ключевые слова")
        st.info("Нет сгенерированных ключевых слов")
    
    # Блок веб-результатов с группировкой по запросам
    if st.session_state.get('web_search_results'):
        st.subheader("🌐 Веб-результаты")
        
        # Группируем результаты по запросам
        queries = {res['query'] for res in st.session_state.web_search_results}
        
        for query in queries:
            query_results = [res for res in st.session_state.web_search_results if res['query'] == query]
            with st.expander(f"Запрос: '{query}' ({len(query_results)} результатов)", expanded=False):
                for i, res in enumerate(query_results):
                    st.markdown(f"**{i+1}. [{res['title']}]({res['url']})**")
                    st.caption(res['snippet'][:200] + "...")
    else:
        st.subheader("🌐 Веб-результаты")
        st.info("Нет веб-результатов")
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
    #if st.session_state.get('qdrant_chunks'):
        #st.subheader("📚 База знаний (Qdrant)")
        #for i, chunk in enumerate(st.session_state.qdrant_chunks[:5]):
            #st.text_area(
                #f"Фрагмент {i+1}", 
                #value=chunk[:2000], 
                #height=150,
                #key=f"qdrant_chunk_{i}"
            #)
    #else:
        #st.info("Нет результатов из базы знаний")

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
