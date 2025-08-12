#main.py
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


if 'index_builder' not in st.session_state:
    st.session_state.index_builder = IndexBuilder()
    # Принудительная загрузка при старте
    with st.spinner("Инициализация моделей..."):
        try:
            st.session_state.index_builder._load_models()
            st.session_state.models_loaded = True
        except Exception as e:
            st.error(f"Ошибка загрузки моделей: {str(e)}")
            st.session_state.models_loaded = False

if 'test_results' not in st.session_state:
    st.session_state.test_results = {
        'dense': None,
        'sparse': None,
        'last_test': None
    }

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
        "reasoning": "",  # Добавляем поле для рассуждений
        "expanded_queries": [],
        "expanded_keywords": []
    }
    
    # Заменяем все варианты разделителей на единый формат
    response = response.replace('\r\n', '\n').replace('\r', '\n')
    
    # Разделяем ответ на секции
    sections = {}
    current_section = None
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Определяем секции по заголовкам
        if line.startswith("Проблема:"):
            current_section = "problem"
            sections[current_section] = line.replace("Проблема:", "").strip()
        elif line.startswith("Предварительные рассуждения:"):
            current_section = "reasoning"
            sections[current_section] = ""
        elif line.startswith("Вопросы:"):
            current_section = "queries"
        elif line.startswith("Ключевые слова:"):
            current_section = "keywords"
            sections[current_section] = ""
        else:
            if current_section == "reasoning":
                sections[current_section] = sections.get(current_section, "") + line + "\n"
            elif current_section == "queries" and re.match(r'^\d+\.', line):
                if "expanded_queries" not in sections:
                    sections["expanded_queries"] = []
                sections["expanded_queries"].append(re.sub(r'^\d+\.\s*', '', line).strip())
            elif current_section == "keywords":
                # Обрабатываем ключевые слова
                if line and not line.startswith(("1.", "2.", "3.", "4.", "5.")):
                    # Если ключевые слова в одной строке через запятую
                    keywords = [k.strip() for k in line.split(',')]
                    sections["expanded_keywords"] = keywords
                else:
                    # Если ключевые слова по одному на строку
                    if "expanded_keywords" not in sections:
                        sections["expanded_keywords"] = []
                    clean_line = re.sub(r'^\d+\.\s*', '', line).strip()
                    if clean_line:
                        sections["expanded_keywords"].append(clean_line)
    
    # Заполняем результат
    result["problem_formulation"] = sections.get("problem", "")
    result["reasoning"] = sections.get("reasoning", "").strip()
    
    # Обрабатываем вопросы
    if "expanded_queries" in sections:
        result["expanded_queries"] = sections["expanded_queries"]
    else:
        # Попробуем найти вопросы другим способом
        queries = []
        in_queries = False
        for line in lines:
            line = line.strip()
            if line.startswith("Вопросы:"):
                in_queries = True
                continue
            if in_queries and re.match(r'^\d+\.', line):
                queries.append(re.sub(r'^\d+\.\s*', '', line).strip())
            elif line.startswith(("Ключевые слова:", "###")):
                in_queries = False
        result["expanded_queries"] = queries
    
    # Обрабатываем ключевые слова
    if "expanded_keywords" in sections:
        result["expanded_keywords"] = sections["expanded_keywords"]
    else:
        # Попробуем найти ключевые слова другим способом
        keywords = []
        in_keywords = False
        for line in lines:
            line = line.strip()
            if line.startswith("Ключевые слова:"):
                in_keywords = True
                continue
            if in_keywords and re.match(r'^\d+\.', line):
                keywords.append(re.sub(r'^\d+\.\s*', '', line).strip())
            elif line.startswith(("Вопросы:", "###", "Проблема:")):
                in_keywords = False
        result["expanded_keywords"] = keywords
    
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

# В обработке кнопки "Отправить" изменяем часть получения результатов:
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
        stage1_response = call_gemini_api(stage1_prompt, temperature=0.3)
        search_data = parse_stage1_response(stage1_response)
        # Сохраняем рассуждения Stage 1 в session_state
        st.session_state.stage1_reasoning = search_data.get("reasoning", "")
        
        # Фильтруем и ограничиваем запросы
        valid_queries = [
            q for q in search_data['expanded_queries'] 
            if q and len(q.split()) <= 20
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
    
            # Гибридный поиск в Qdrant
            qdrant_future = executor.submit(
                lambda: index_builder.hybrid_search(
                    queries=[user_input] + valid_queries,
                    top_k=5
                )
            )
            
            web_results = web_future.result()
            qdrant_results, search_error = qdrant_future.result()
            if search_error:
                st.warning(search_error)
            st.session_state.search_error = search_error

        progress_bar.progress(80)
        
        # Сохраняем результаты

        st.session_state.web_search_results = web_results
        st.session_state.hybrid_results = qdrant_results
        st.session_state.search_stats = {
            'total': len(qdrant_results),
            'dense': len([r for r in qdrant_results if r.get('vector_type') == 'dense']),
            'sparse': len([r for r in qdrant_results if r.get('vector_type') == 'sparse'])
        }
        st.session_state.primary_web_results = web_results.copy()
        st.session_state.primary_hybrid_results = qdrant_results.copy()
        # Этап 3: Формирование контекста
        status_text.text("Формирование ответа...")
        context_parts = [
            f"Проблема: {search_data['problem_formulation']}",
            "Веб-результаты:"
        ]
        
        for i, res in enumerate(web_results[:5]):
            # Используем полный контент вместо сниппета (ограничиваем 8000 символов)
            content = res.get('full_content', '')[:8000]
            context_parts.append(f"{i+1}. [{res['title']}]({res['url']}): {content}")
        if st.session_state.get('document_text'):
            context_parts.append("Загруженный документ:")
            context_parts.append(f"1. {st.session_state.document_text}")

        
        context_parts.append("Базовые знания:")
        
        # Используем расширенный контекст вместо обычного контента
        for i, res in enumerate(qdrant_results[:7]):  # Ограничиваем количеством
            # Используем расширенный контекст если он есть, иначе обычный контент
            content = res.get('expanded_context', res.get('content', ''))
            context_parts.append(f"{i+1}. Реквизиты источника: {res['id']}\n{content}")
        
        full_context = "\n\n".join(context_parts)[:200000]
        
        progress_bar.progress(100)
        status_text.text("Готово!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

    except Exception as e:
        progress_bar.empty()
        status_text.error(f"Ошибка обработки запроса: {str(e)}")
        st.stop()

    # В обработке кнопки "Отправить", после первичного поиска:

    # ... [существующий код до формирования full_context] ...

    # Этап 3: Генерация уточняющих запросов и предварительного проекта
    with st.spinner("Углубленный анализ..."):
        try:
            # Формируем промпт для генерации уточняющих запросов
            stage3_prompt = get_prompt("stage3", {
                "problem_formulation": search_data['problem_formulation'],
                "context": full_context
            })
        
            stage3_response = call_gemini_api(stage3_prompt, max_output_tokens=4000)
        
            # Парсим сгенерированные вопросы
            refined_queries = []
            if "Уточняющие вопросы:" in stage3_response:
                queries_section = stage3_response.split("Уточняющие вопросы:")[1]
                if "Предварительный проект:" in queries_section:
                    queries_section = queries_section.split("Предварительный проект:")[0]
                refined_queries = [
                    line.strip() 
                    for line in queries_section.split('\n') 
                    if re.match(r'^\d+\.', line)
                ][:5]
                st.session_state.refined_queries = refined_queries  # <-- ЭТО БЫЛО ПРОПУЩЕНО

            # Сохраняем предварительный проект
            if "Предварительный проект:" in stage3_response:
                st.session_state.preliminary_draft = stage3_response.split("Предварительный проект:")[1].strip()
            # Выполняем новый поиск по уточняющим запросам
            if refined_queries:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # Веб-поиск по новым запросам
                    web_future_refined = executor.submit(
                        lambda: [res for q in refined_queries for res in web_searcher.perform_search(q, 2)]
                    )
                
                    # Гибридный поиск в Qdrant
                    qdrant_future_refined = executor.submit(
                        lambda: index_builder.hybrid_search(
                            queries=refined_queries,
                            top_k=3
                        )
                    )
                
                    web_results_refined = web_future_refined.result()
                    qdrant_results_refined, _ = qdrant_future_refined.result()
                    st.session_state.refined_web_results = web_results_refined
                    st.session_state.refined_hybrid_results = qdrant_results_refined
                    # После получения refined_queries, web_results_refined и qdrant_results_refined
                    st.session_state.refined_queries = refined_queries

                

                
            # Обновляем контекст с новыми результатами
            context_parts.append("\n\nУточняющие результаты:")
            for i, res in enumerate(web_results_refined[:3]):
                context_parts.append(f"W{i+1}. [{res['title']}]({res['url']}): {res.get('full_content', '')[:2000]}")
            for i, res in enumerate(qdrant_results_refined[:2]):
                context_parts.append(f"Q{i+1}. {res.get('content', '')[:2000]}")
        
            full_context = "\n\n".join(context_parts)[:200000]

        except Exception as e:
            st.error(f"Ошибка углубленного анализа: {str(e)}")
            st.stop()
    
 
    


    # Этап 4: Генерация проекта заключения (бывший stage2)
    with st.spinner("Подготовка проекта заключения..."):
        try:
            stage4_prompt = get_prompt("stage4", {  # Бывший stage2
                "user_query": user_input,
                "problem_formulation": search_data['problem_formulation'],
                "context": full_context
            })
            # ... [остальной код без изменений] ...
    
            stage2_response = call_gemini_api(stage2_prompt, max_output_tokens=10000)
    
            try:
                opinion_data = parse_stage2_response(stage2_response)
                st.session_state.opinion_data = opinion_data
        
                # Для отладки
                with st.expander("Промежуточные результаты"):
                    st.subheader("Предварительные рассуждения (Stage 1)")
                    if st.session_state.get('stage1_reasoning'):
                        st.write(st.session_state.stage1_reasoning)
                    else:
                        st.write("Предварительные рассуждения не сгенерированы")
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
    

    # Этап 5: Финальная проверка (бывший stage3)
    with st.spinner("Финальная проверка..."):
        try:
            stage5_prompt = get_prompt("stage5", {  # Бывший stage3
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

with st.sidebar:
    st.subheader("Результаты поиска")
    
    # === ПЕРВИЧНЫЕ РЕЗУЛЬТАТЫ ===
    primary_web = st.session_state.get('primary_web_results', [])
    if primary_web:
        st.subheader("🌐 Первичные веб-результаты")

        queries = {res['query'] for res in web_results}
        for query in queries:
            query_results = [res for res in web_results if res['query'] == query]
            with st.expander(f"Запрос: '{query}' ({len(query_results)} результатов)", expanded=False):
                for i, res in enumerate(query_results):
                    st.markdown(f"**{i+1}. [{res['title']}]({res['url']})**")
                    st.caption(res['snippet'][:200] + "...")
    else:
        st.info("Нет веб-результатов")


    
    # Гибридные результаты

    primary_hybrid = st.session_state.get('primary_hybrid_results', [])
    if primary_hybrid:
        st.subheader("🔍 Первичный гибридный поиск")
    
        dense_results = [r for r in hybrid_results if r.get('vector_type') == 'dense']
        sparse_results = [r for r in hybrid_results if r.get('vector_type') == 'sparse']
    
        tab1, tab2 = st.tabs(["Плотные векторы", "Разреженные векторы"])
    
        with tab1:
            if dense_results:
                for i, res in enumerate(dense_results[:5]):
                    with st.expander(f"Плотный #{i+1} (score: {res['score']:.2f})", expanded=False):
                        st.write(f"**Запрос:** `{res.get('query', '')}`")
                        # Показываем оригинальный контент, а не расширенный
                        st.write(f"**Текст:** {res.get('content', '')[:10000]}...")
                        st.write(f"**ID:** `{res['id']}`")
                        # Добавляем информацию о расширенном контексте
                        st.caption(f"Контекст: {len(res.get('expanded_context', ''))} символов")
            else:
                st.info("Нет результатов по плотным векторам")
    
        with tab2:
            if sparse_results:
                for i, res in enumerate(sparse_results[:5]):
                    with st.expander(f"Разреженный #{i+1} (score: {res['score']:.2f})", expanded=False):
                        st.write(f"**Запрос:** `{res.get('query', '')}`")
                        st.write(f"**Текст:** {res.get('content', '')[:10000]}...")
                        st.write(f"**ID:** `{res['id']}`")
                        st.caption(f"Контекст: {len(res.get('expanded_context', ''))} символов")
            else:
                st.info("Нет результатов по разреженным векторам")
    else:
        st.info("Гибридный поиск не дал результатов")

    # Сгенерированные запросы (единый блок)
    generated_queries = st.session_state.get('generated_queries', [])
    st.subheader("🔍 Сгенерированные запросы")
    if generated_queries:
        for i, query in enumerate(generated_queries[:3]):
            st.code(f"{i+1}. {query}")
    else:
        st.info("Нет сгенерированных запросов")

    # Ключевые слова
    keywords = st.session_state.get('generated_keywords', [])
    st.subheader("🔑 Ключевые слова")
    if keywords:
        st.write(", ".join(keywords))
    else:
        st.info("Нет ключевых слов")
    # === УТОЧНЯЮЩИЕ РЕЗУЛЬТАТЫ ===
    refined_web = st.session_state.get('refined_web_results', [])
    if refined_web:
        st.subheader("🔍 Уточняющие веб-результаты")
        queries = {res['query'] for res in refined_web}
        for query in queries:
            query_results = [res for res in refined_web if res['query'] == query]
            with st.expander(f"Уточняющий запрос: '{query}' ({len(query_results)} результатов)", expanded=False):
                for i, res in enumerate(query_results):
                    st.markdown(f"**{i+1}. [{res['title']}]({res['url']})**")
                    st.caption(f"Домен: {res.get('domain', 'неизвестен')}")
                    st.caption(f"Сниппет: {res['snippet'][:200]}{'...' if len(res['snippet']) > 200 else ''}")
                    with st.expander("Показать полный контент", expanded=False):
                        st.write(res.get('full_content', '')[:1000] + "...")
    else:
        st.info("Нет уточняющих веб-результатов")

    refined_hybrid = st.session_state.get('refined_hybrid_results', [])
    if refined_hybrid:
        st.subheader("🔍 Уточняющий гибридный поиск")
    
        dense_results = [r for r in refined_hybrid if r.get('vector_type') == 'dense']
        sparse_results = [r for r in refined_hybrid if r.get('vector_type') == 'sparse']
    
        tab1, tab2 = st.tabs(["Плотные векторы", "Разреженные векторы"])
    
        with tab1:
            if dense_results:
                for i, res in enumerate(dense_results[:5]):
                    with st.expander(f"Плотный #{i+1} (score: {res['score']:.4f})", expanded=False):
                        st.write(f"**Запрос:** `{res.get('query', 'не указан')}`")
                        st.write(f"**ID документа:** `{res['id']}`")
                        st.write(f"**Источник:** `{res.get('source', 'неизвестен')}`")
                        st.write(f"**Дата:** `{res.get('date', 'не указана')}`")
                    
                        # Отображаем контент с ограничением
                        content = res.get('content', '')
                        if len(content) > 500:
                            with st.expander("Показать полный текст", expanded=False):
                                st.write(content)
                            st.caption(f"Кратко: {content[:500]}...")
                        else:
                            st.write(f"**Текст:** {content}")
                    
                        # Информация о контексте
                        if 'expanded_context' in res and res['expanded_context']:
                            st.caption(f"Расширенный контекст: {len(res['expanded_context'])} символов")
            else:
                st.info("Нет результатов по плотным векторам")
    
        with tab2:
            if sparse_results:
                for i, res in enumerate(sparse_results[:5]):
                    with st.expander(f"Разреженный #{i+1} (score: {res['score']:.4f})", expanded=False):
                        st.write(f"**Запрос:** `{res.get('query', 'не указан')}`")
                        st.write(f"**ID документа:** `{res['id']}`")
                        st.write(f"**Источник:** `{res.get('source', 'неизвестен')}`")
                        st.write(f"**Дата:** `{res.get('date', 'не указана')}`")
                    
                        # Отображаем контент с ограничением
                        content = res.get('content', '')
                        if len(content) > 500:
                            with st.expander("Показать полный текст", expanded=False):
                                st.write(content)
                            st.caption(f"Кратко: {content[:500]}...")
                        else:
                            st.write(f"**Текст:** {content}")
                    
                        # Информация о sparse-векторе
                        if 'sparse_vector' in res:
                            st.caption(f"Размерность sparse: {res['sparse_vector'].get('dim', 'N/A')}")
                            st.caption(f"Ненулевых элементов: {res['sparse_vector'].get('nnz', 'N/A')}")
            else:
                st.info("Нет результатов по разреженным векторам")
    else:
        st.info("Уточняющий гибридный поиск не дал результатов")

    # === УТОЧНЯЮЩИЕ ЗАПРОСЫ (РАБОТАЕТ ТЕПЕРЬ!) ===
    if st.session_state.get('refined_queries'):
        st.subheader("🔍 Сгенерированные уточняющие запросы")
        for i, query in enumerate(st.session_state.refined_queries):
            st.code(f"{i+1}. {query}")
    
        # Добавляем статистику по уточняющим запросам
        st.caption(f"Всего сгенерировано: {len(st.session_state.refined_queries)}")
    else:
        st.info("Нет сгенерированных уточняющих запросов") 
    
    # Блок уточняющих запросов
    if st.session_state.get('refined_queries'):
        st.subheader("🔍 Уточняющие запросы")
        for i, query in enumerate(st.session_state.refined_queries):
            st.code(f"{i+1}. {query}")
    
    # Блок предварительного проекта
    if st.session_state.get('preliminary_draft'):
        st.subheader("📝 Предварительный проект")
        with st.expander("Показать черновик", expanded=False):
            st.write(st.session_state.preliminary_draft[:2000] + "...")  

    # Блок загруженного документа
    if st.session_state.get('document_text'):
        st.subheader("📄 Загруженный документ")
        with st.expander("Показать текст документа", expanded=False):
            st.text_area(
                "Содержание документа",
                value=st.session_state.document_text[:10000],
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

    st.subheader("🧠 Когнитивные рассуждения")
    
    # Добавляем блок с рассуждениями Stage 1
    if st.session_state.get('stage1_reasoning'):
        with st.expander("Показать рассуждения", expanded=True):
            st.markdown("""
            <style>
                .reasoning-box {
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 10px;
                }
                .reasoning-stage {
                    font-weight: bold;
                    color: #1E88E5;
                    margin-top: 10px;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Форматируем рассуждения для лучшего отображения
            reasoning = st.session_state.stage1_reasoning
            
            # Добавляем стилизованные заголовки этапов
            reasoning = reasoning.replace("## Этап 1:", "<div class='reasoning-box'><div class='reasoning-stage'>Этап 1: Декомпозиция и Понимание</div>")
            reasoning = reasoning.replace("## Этап 2:", "</div><div class='reasoning-box'><div class='reasoning-stage'>Этап 2: Планирование и Поиск Знаний</div>")
            reasoning = reasoning.replace("## Этап 3:", "</div><div class='reasoning-box'><div class='reasoning-stage'>Этап 3: Критическая Оценка и Синтез</div>")
            reasoning = reasoning.replace("## Этап 4:", "</div><div class='reasoning-box'><div class='reasoning-stage'>Этап 4: Рефлексия и Проверка</div>")
            reasoning = reasoning + "</div>" * 3  # Закрываем все div
            
            # Добавляем маркированные списки
            reasoning = re.sub(r'\*   ', '- ', reasoning)
            
            st.markdown(reasoning, unsafe_allow_html=True)
    else:
        st.info("Предварительные рассуждения не сгенерированы")




