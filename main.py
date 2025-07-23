import os

print(f"Текущая рабочая директория: {os.getcwd()}")
print(f"Список файлов: {os.listdir('.')}")
import streamlit as st
import time
import requests
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

logger = setup_logging()
initialize_session()

if "context_parts" not in st.session_state:
    st.session_state.context_parts = []

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.path.join(REPO_ROOT, "data", "embeddings")

if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)
    st.warning(f"Создана папка для эмбеддингов: {EMBEDDINGS_DIR}")

DOCUMENTS_DIR = os.path.join(REPO_ROOT, "documents")
if not os.path.exists(DOCUMENTS_DIR):
    os.makedirs(DOCUMENTS_DIR)
    st.info(f"Created documents directory: {DOCUMENTS_DIR}")

print(f"Documents directory: {DOCUMENTS_DIR}, exists: {os.path.exists(DOCUMENTS_DIR)}")

index_builder = IndexBuilder()
print(f"Путь к папке эмбеддингов: {os.path.abspath(index_builder.EMBEDDINGS_CACHE_DIR)}")
print(f"Папка существует: {os.path.exists(index_builder.EMBEDDINGS_CACHE_DIR)}")
data_processor = DataProcessor(index_builder)
query_generator = QueryGenerator()
web_searcher = WebSearcher()

if "web_searcher" not in st.session_state:
    st.session_state.web_searcher = web_searcher

if not index_builder.load_full_index():
    print("Полный индекс не найден, будет построен при обработке документа")

st.title("ИИ-помощник по подготовке правовых заключений")
uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

def perform_bm25_search(query: str, bm25_index, original_chunks) -> List[str]:
    """Универсальная функция поиска BM25"""
    if not query.strip() or not bm25_index:
        return []
    
    keywords = data_processor.extract_keywords(query, bm25_index)
    return data_processor.search_relevant_chunks(bm25_index, original_chunks, keywords)

if uploaded_file:
    with st.spinner("Анализ документа..."):
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        
        st.session_state.document_text = clean_text(file_text)
        bm25_index, original_chunks = index_builder.create_bm25_index()
        
        # Сохраняем индекс для последующего использования
        st.session_state.bm25_index = bm25_index
        st.session_state.original_chunks = original_chunks
        
        # Только специфичные для документа действия
        if uploaded_file.name.endswith(".txt"):
            st.session_state.main_doc_name = uploaded_file.name
        else:
            st.session_state.main_doc_name = uploaded_file.name.split('.')[0] + ".txt"
        
        keywords = data_processor.extract_keywords(st.session_state.document_text, bm25_index)
        st.session_state.document_keywords = keywords
        
        if uploaded_file.name.endswith(".txt"):
            st.session_state.main_doc_name = uploaded_file.name
        else:
            st.session_state.main_doc_name = uploaded_file.name.split('.')[0] + ".txt"
        
        relevant_chunks = data_processor.search_relevant_chunks(
            bm25_index, original_chunks, keywords)
        
        enhanced_chunks = data_processor.enhance_with_semantic_search(
            " ".join(keywords), relevant_chunks)
        
        keywords = data_processor.extract_keywords(st.session_state.document_text, bm25_index)
        qdrant_chunks = data_processor.enhance_with_qdrant_search(
            " ".join(keywords),  # Запрос для векторного поиска
            keywords,            # Ключевые слова для фильтрации
            relevant_chunks
        )
        st.session_state.qdrant_chunks = qdrant_chunks
        
        # В сборку контекста добавляем:
        if st.session_state.get('qdrant_chunks'):
            st.session_state.context_parts.append("Контекст из базы знаний Qdrant:\n" + 
                                "\n\n".join(st.session_state.qdrant_chunks[:5]))
        
        final_chunks = data_processor.enhance_with_graph_context(
            st.session_state.get('main_doc_name'),
            enhanced_chunks
        )
        
        st.session_state.document_relevant_chunks = final_chunks
        
        if st.session_state.document_relevant_chunks:
            st.subheader("Релевантные фрагменты из документа:")
            for i, chunk in enumerate(st.session_state.document_relevant_chunks):
                st.text_area(f"Фрагмент {i+1}", 
                            value=chunk[:5000], 
                            height=150,
                            key=f"doc_chunk_{i}_{hash(chunk[:50])}")
if 'bm25_index' not in st.session_state:
    bm25_index, original_chunks = index_builder.create_bm25_index()
    st.session_state.bm25_index = bm25_index
    st.session_state.original_chunks = original_chunks
user_input = st.text_area("Введите ваш вопрос:", height=150, max_chars=600, key="user_input")

if st.button("Отправить", key="send_btn"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    
    st.session_state.last_query = user_input
    
    with st.spinner("Обработка запроса..."):
        # Веб-поиск по исходному запросу
        initial_web_results = web_searcher.perform_search(user_input)
        initial_web_chunks = [result['full_content'] for result in initial_web_results if result['full_content']]
        st.session_state.initial_web_results = initial_web_results
        st.session_state.initial_web_chunks = initial_web_chunks
        
        bm25_index, original_chunks = index_builder.create_bm25_index()
        if not bm25_index:
            st.error("Ошибка индексации")
            st.stop()

        bm25_results = perform_bm25_search(
            user_input,
            st.session_state.bm25_index,
            st.session_state.original_chunks
        )
        st.session_state.current_bm25_results = bm25_results
            
        query_keywords = data_processor.extract_keywords(user_input, bm25_index)
        generated_queries = query_generator.generate(user_input, query_keywords)
        st.session_state.generated_queries = generated_queries
        
        all_knowledge_chunks = []
        additional_chunks = []
        
        for query in generated_queries:
            q_keywords = data_processor.extract_keywords(query, bm25_index)
            if not q_keywords:
                continue
                
            q_chunks = data_processor.search_relevant_chunks(bm25_index, original_chunks, q_keywords)
            unique_chunks = data_processor.get_unique_chunks(all_knowledge_chunks, q_chunks)
            additional_chunks.extend(unique_chunks)
            all_knowledge_chunks.extend(unique_chunks)
        
        st.session_state.additional_chunks = additional_chunks
        
        web_results = []
        for query in generated_queries:
            results = st.session_state.web_searcher.perform_search(query)
            web_results.extend(results)
        
        web_chunks = [result['full_content'] for result in web_results if result['full_content']]
        st.session_state.web_search_results = web_results
        st.session_state.web_search_chunks = web_chunks[:3]
        
        # Добавляем поиск в Qdrant по всем запросам
        all_qdrant_chunks = []
        for query in [user_input] + generated_queries:
            q_chunks = data_processor.enhance_with_qdrant_search(query, [])
            all_qdrant_chunks.extend(q_chunks)
            st.session_state.all_qdrant_chunks = all_qdrant_chunks
    
        # Добавляем в контекст
        if all_qdrant_chunks:
            st.session_state.context_parts.append("Семантический поиск из Qdrant:\n" +
                               "\n\n".join(all_qdrant_chunks[:10]))
        
        # Веб-контекст по основному запросу
        if st.session_state.get('initial_web_chunks'):
            st.session_state.context_parts.append("Веб-контекст по основному запросу:\n" + 
                                "\n\n".join(st.session_state.initial_web_chunks[:2]))
        
        if st.session_state.document_relevant_chunks:
            st.session_state.context_parts.append("Контекст из документа:\n" + 
                                "\n\n".join(st.session_state.document_relevant_chunks[:3]))
        
        if all_knowledge_chunks:
            st.session_state.context_parts.append("Основной контекст из базы знаний:\n" + 
                                "\n\n".join(all_knowledge_chunks[:3]))
        
        if additional_chunks:
            st.session_state.context_parts.append("Дополнительный контекст:\n" + 
                                "\n\n".join(additional_chunks[:3]))
        
        if st.session_state.web_search_chunks:
            st.session_state.context_parts.append("Контекст из веб-поиска по уточняющим запросам:\n" + 
                                "\n\n".join(st.session_state.web_search_chunks))
        
        full_context = "\n\n".join(st.session_state.context_parts)
        
        prompt = get_prompt("system", {
            "user_query": user_input,
            "context": full_context[:15000]
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

# Основной ответ остается в главной области
if st.session_state.get('llm_response'):
    st.subheader("Ответ ассистента:")
    st.markdown(st.session_state.llm_response)

# ===== ВСЕ ВСПОМОГАТЕЛЬНЫЕ ЭЛЕМЕНТЫ ПЕРЕНОСИМ В САЙДБАР =====
with st.sidebar:
    # Сгенерированные запросы
    if st.session_state.get('generated_queries'):
        st.subheader("Сгенерированные уточняющие запросы:")
        for i, query in enumerate(st.session_state.generated_queries):
            st.write(f"{i+1}. {query}")

    
    # Результаты bm25
    if st.session_state.get('bm25_results'):
        st.subheader("Результаты BM25 поиска")
        for i, chunk in enumerate(st.session_state.bm25_results[:5]):  # Показываем первые 5 результатов
            st.text_area(f"Фрагмент {i+1} (BM25)", 
                        value=chunk[:2000],  # Ограничиваем длину для отображения
                        height=150,
                        key=f"bm25_chunk_{i}")

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
