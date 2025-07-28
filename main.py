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

# Обработка запроса пользователя
if st.button("Отправить", key="send_btn"):
    with st.spinner("Обработка запроса..."):
        # 1. Веб-поиск (максимум 3 запроса)
        web_queries = [user_input] + (st.session_state.generated_queries[:2] 
                      if st.session_state.get('generated_queries') else [])
        web_results = []
        for query in web_queries[:3]:  # Не более 3 запросов
            results = web_searcher.perform_search(query, max_results=2)
            web_results.extend(results)
        
        # 2. Поиск в Qdrant
        qdrant_chunks = []
        for query in [user_input] + web_queries[:2]:
            chunks = data_processor.enhance_with_qdrant_search(query, top_k=5)
            qdrant_chunks.extend(chunks)
        
        # 3. Формирование контекста
        context_parts = []
        
        if st.session_state.get('document_text'):
            context_parts.append("Загруженный документ:\n" + st.session_state.document_text)
        
        if web_results:
            web_context = "\n\n".join(
                [f"Источник {i+1} ({res['title']}):\n{res['full_content'][:2000]}" 
                 for i, res in enumerate(web_results)]
            )
            context_parts.append("Веб-источники:\n" + web_context)
        
        if qdrant_chunks:
            context_parts.append("База знаний Qdrant:\n" + "\n\n".join(qdrant_chunks[:5]))
        
        full_context = "\n\n".join(context_parts)
        
        # 4. Отправка в LLM
        prompt = get_prompt("system", {
            "user_query": user_input,
            "context": full_context[:15000]  # Ограничение контекста
        })
        
        try:
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

    
    # Остальные элементы сайдбара (веб-результаты, Qdrant и т.д.)
    # ... (остальной код сайдбара остается без изменений)

    # Веб-результаты по основному запросу
    if st.session_state.get('initial_web_results'):
        st.subheader("Результаты веб-поиска по основному запросу")
        for i, result in enumerate(st.session_state.initial_web_results):
            with st.expander(f"{i+1}. {result['title']}", expanded=False):
                st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                st.markdown("**Сниппет:**")
                st.info(result.get('snippet', ''))
                
                if result.get('full_content'):
                    st.text_area("Контент", 
                                value=result['full_content'][:3000], 
                                height=200,
                                key=f"initial_web_content_{i}")

    # Фрагменты из Qdrant (по запросу)
    if st.session_state.get('all_qdrant_chunks'):
        st.subheader("Фрагменты из базы знаний (по запросу)")
        for i, chunk in enumerate(st.session_state.all_qdrant_chunks[:5]):
            st.text_area(f"Фрагмент {i+1}", 
                        value=chunk[:2000], 
                        height=150,
                        key=f"all_qdrant_chunk_{i}")

    # Веб-результаты по уточняющим запросам
    if st.session_state.get('web_search_results'):
        st.subheader("Веб-результаты по уточняющим запросам")
        for i, result in enumerate(st.session_state.web_search_results):
            with st.expander(f"{i+1}. {result['title']}", expanded=False):
                st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                st.markdown("**Сниппет:**")
                st.info(result.get('snippet', ''))
                
                if result.get('full_content'):
                    st.text_area("Контент", 
                                value=result['full_content'][:3000], 
                                height=200,
                                key=f"web_content_{i}_{hash(result['url'])}")
    
    # Фрагменты из Qdrant (из документа)
    if st.session_state.get('qdrant_chunks'):
        st.subheader("Фрагменты из базы знаний (из документа)")
        for i, chunk in enumerate(st.session_state.qdrant_chunks[:5]):
            st.text_area(f"Фрагмент {i+1}", 
                        value=chunk[:2000], 
                        height=150,
                        key=f"qdrant_chunk_{i}")
    
    # Граф документа
    if st.session_state.get('main_doc_name') and index_builder.document_graph:
        try:
            graph = nx.DiGraph()
            has_edges = False
            
            # Добавляем основной документ
            main_doc = st.session_state.main_doc_name
            graph.add_node(main_doc)
            
            # Добавляем связи
            for doc, refs in index_builder.document_graph.items():
                for ref in refs:
                    graph.add_edge(doc, ref)
                    has_edges = True
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if has_edges:
                pos = nx.spring_layout(graph)
                nx.draw(graph, pos, 
                       with_labels=True,
                       node_color='skyblue',
                       node_size=2000,
                       edge_color='gray',
                       font_size=10,
                       ax=ax)
            else:
                # Визуализация для одного узла
                nx.draw_networkx_nodes(graph, 
                                      pos={main_doc: [0,0]}, 
                                      nodelist=[main_doc],
                                  node_color='skyblue',
                                  node_size=2000,
                                  ax=ax)
                plt.text(0, 0.1, main_doc, 
                        ha='center',
                        bbox=dict(facecolor='white', alpha=0.8))
            
            st.subheader("Граф связанных документов")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Ошибка визуализации графа: {str(e)}")

    # История диалога
    if st.session_state.get('chat_log'):
        st.subheader("История диалога")
        st.text_area(label="История", 
                    value=st.session_state.chat_log, 
                    height=300, 
                    key="chat_history_unique",
                    disabled=True)
