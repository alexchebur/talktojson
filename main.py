import os
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

def get_index_path():
    """Определяем путь к файлу индекса без использования Streamlit элементов"""
    try:
        # Пробуем несколько возможных путей
        possible_paths = [
            Path(__file__).parent / "data" / "bm25_index.pkl",
            Path.cwd() / "data" / "bm25_index.pkl",
            Path("/content/drive/MyDrive/data sources/Talk2JsonDocsRAG/bm25_index.pkl"),
            Path("data/bm25_index.pkl")
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
    except Exception:
        return None

def load_bm25_index():
    """Загрузка индекса без использования Streamlit элементов"""
    try:
        index_path = get_index_path()
        if not index_path:
            return None, None
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            
            if not isinstance(data, dict):
                return None, None
                
            if 'index' not in data or 'original_chunks' not in data:
                return None, None
                
            return data['index'], data['original_chunks']
            
    except Exception:
        return None, None

def initialize_session_with_index():
    """Инициализация сессии с загрузкой индекса"""
    if 'bm25_initialized' not in st.session_state:
        # Загрузка индекса (без кэширования)
        bm25_index, original_chunks = load_bm25_index()
        
        # Сохраняем в session_state
        st.session_state.update({
            'bm25_index': bm25_index,
            'original_chunks': original_chunks,
            'bm25_initialized': True
        })
        
        # Проверка и отображение ошибок
        if not bm25_index:
            st.error("### Не удалось загрузить индекс BM25")
            
            # Показываем информацию о путях
            index_path = get_index_path()
            if index_path:
                st.write(f"Файл найден по пути: `{index_path}`")
                st.write("**Возможные причины:**")
                st.write("- Неправильный формат файла")
                st.write("- Несовместимость версий Python")
            else:
                st.write("**Файл не найден.** Проверьте пути:")
                st.write("- `data/bm25_index.pkl`")
                st.write("- `/content/drive/MyDrive/data sources/Talk2JsonDocsRAG/bm25_index.pkl`")
            #st.stop()

# Инициализация при запуске приложения
initialize_session_with_index()

# Инициализация веб-поиска
if "web_searcher" not in st.session_state:
    st.session_state.web_searcher = web_searcher

# Загрузка полного индекса (если нужно)
if not index_builder.load_full_index():
    print("Полный индекс не найден, будет построен при обработке документа")

# Интерфейс приложения
st.title("ИИ-помощник по подготовке правовых заключений")
uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

def perform_bm25_search(query: str) -> List[str]:
    """Унифицированная функция поиска BM25 с использованием глобального индекса"""
    if not query.strip() or not st.session_state.bm25_index:
        return []
    
    keywords = data_processor.extract_keywords(query, st.session_state.bm25_index)
    return data_processor.search_relevant_chunks(
        st.session_state.bm25_index, 
        st.session_state.original_chunks, 
        keywords
    )

if uploaded_file:
    with st.spinner("Анализ документа..."):
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        
        st.session_state.document_text = clean_text(file_text)
        
        # Используем глобальный индекс BM25
        keywords = data_processor.extract_keywords(st.session_state.document_text, st.session_state.bm25_index)
        st.session_state.document_keywords = keywords
        
        # Сохраняем имя документа
        if uploaded_file.name.endswith(".txt"):
            st.session_state.main_doc_name = uploaded_file.name
        else:
            st.session_state.main_doc_name = uploaded_file.name.split('.')[0] + ".txt"
        
        # Поиск релевантных фрагментов
        relevant_chunks = perform_bm25_search(" ".join(keywords))
        st.session_state.document_relevant_chunks = relevant_chunks
        
        # Дополнительные улучшения контекста
        enhanced_chunks = data_processor.enhance_with_semantic_search(
            " ".join(keywords), relevant_chunks)
        
        qdrant_chunks = data_processor.enhance_with_qdrant_search(
            " ".join(keywords), relevant_chunks)
        st.session_state.qdrant_chunks = qdrant_chunks
        
        # Добавляем контекст в session_state
        if st.session_state.get('qdrant_chunks'):
            st.session_state.context_parts.append(
                "Контекст из базы знаний Qdrant:\n" + 
                "\n\n".join(st.session_state.qdrant_chunks[:5])
            )
        
        final_chunks = data_processor.enhance_with_graph_context(
            st.session_state.get('main_doc_name'),
            enhanced_chunks
        )
        
        # Отображение результатов
        if st.session_state.document_relevant_chunks:
            st.subheader("Релевантные фрагменты из документа:")
            for i, chunk in enumerate(st.session_state.document_relevant_chunks[:3]):  # Показываем только топ-3
                st.text_area(
                    f"Фрагмент {i+1}", 
                    value=chunk[:5000], 
                    height=150,
                    key=f"doc_chunk_{i}"
                )

user_input = st.text_area("Введите ваш вопрос:", height=150, max_chars=600, key="user_input")

if st.button("Отправить", key="send_btn"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    
    st.session_state.last_query = user_input
    
    with st.spinner("Обработка запроса..."):
        # Сбрасываем состояние перед новым поиском
        st.session_state.current_bm25_results = []
        st.session_state.bm25_search_error = None
        
        try:
            # 1. Поиск по BM25 индексу
            if st.session_state.bm25_index is None:
                st.session_state.bm25_search_error = "Индекс BM25 не загружен"
            else:
                bm25_results = perform_bm25_search(user_input)
                st.session_state.current_bm25_results = bm25_results
                
                if not bm25_results:
                    st.session_state.bm25_search_error = "Не найдено релевантных фрагментов"
                
            # ... (остальная обработка запроса) ...

        except Exception as e:
            st.session_state.bm25_search_error = f"Ошибка поиска BM25: {str(e)}"
            logger.error(f"BM25 search failed: {str(e)}")
        
        # 1. Веб-поиск по исходному запросу
        initial_web_results = web_searcher.perform_search(user_input)
        initial_web_chunks = [result['full_content'] for result in initial_web_results if result['full_content']]
        st.session_state.initial_web_results = initial_web_results
        st.session_state.initial_web_chunks = initial_web_chunks
        
        # 2. Поиск по BM25 индексу
        bm25_results = perform_bm25_search(user_input)
        st.session_state.current_bm25_results = bm25_results
        
        # 3. Генерация дополнительных запросов
        query_keywords = data_processor.extract_keywords(user_input, st.session_state.bm25_index)
        generated_queries = query_generator.generate(user_input, query_keywords)
        st.session_state.generated_queries = generated_queries
        
        # 4. Поиск по сгенерированным запросам
        all_knowledge_chunks = []
        additional_chunks = []
        
        for query in generated_queries:
            q_chunks = perform_bm25_search(query)
            unique_chunks = data_processor.get_unique_chunks(all_knowledge_chunks, q_chunks)
            additional_chunks.extend(unique_chunks)
            all_knowledge_chunks.extend(unique_chunks)
        
        st.session_state.additional_chunks = additional_chunks
        
        # 5. Веб-поиск по уточняющим запросам
        web_results = []
        for query in generated_queries:
            results = st.session_state.web_searcher.perform_search(query)
            web_results.extend(results)
        
        web_chunks = [result['full_content'] for result in web_results if result['full_content']]
        st.session_state.web_search_results = web_results
        st.session_state.web_search_chunks = web_chunks[:3]
        
        # 6. Семантический поиск в Qdrant
        all_qdrant_chunks = []
        for query in [user_input] + generated_queries:
            q_chunks = data_processor.enhance_with_qdrant_search(query, [])
            all_qdrant_chunks.extend(q_chunks)
        
        st.session_state.all_qdrant_chunks = all_qdrant_chunks
        
        # Формирование контекста
        st.session_state.context_parts = []  # Очищаем перед формированием нового контекста
        
        # Добавляем самые релевантные BM25 результаты
        if st.session_state.current_bm25_results:
            top_bm25_chunks = st.session_state.current_bm25_results[:3]
            st.session_state.context_parts.append(
                "Топ-3 релевантных фрагмента (BM25):\n" +
                "\n\n".join([f"Фрагмент {i+1}:\n{chunk[:2000]}" for i, chunk in enumerate(top_bm25_chunks)])
            )
        
        # Добавляем остальные части контекста
        if st.session_state.get('initial_web_chunks'):
            st.session_state.context_parts.append(
                "Веб-контекст по основному запросу:\n" + 
                "\n\n".join(st.session_state.initial_web_chunks[:2])
            )
        
        if st.session_state.get('document_relevant_chunks'):
            st.session_state.context_parts.append(
                "Контекст из документа:\n" + 
                "\n\n".join(st.session_state.document_relevant_chunks[:3])
            )
        
        if all_knowledge_chunks:
            st.session_state.context_parts.append(
                "Основной контекст из базы знаний:\n" + 
                "\n\n".join(all_knowledge_chunks[:3])
            )
        
        if st.session_state.get('all_qdrant_chunks'):
            st.session_state.context_parts.append(
                "Семантический поиск из Qdrant:\n" +
                "\n\n".join(st.session_state.all_qdrant_chunks[:5])
            )
        
        # Формируем финальный контекст
        full_context = "\n\n".join(st.session_state.context_parts)
        
        # Отправка запроса в LLM
        prompt = get_prompt("system", {
            "user_query": user_input,
            "context": full_context[:15000]  # Ограничиваем размер контекста
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
    # Блок BM25 результатов - всегда отображается
    st.subheader("Результаты BM25 поиска")
    
    if st.session_state.get('bm25_search_error'):
        st.error(st.session_state.bm25_search_error)
    elif st.session_state.get('current_bm25_results'):
        if len(st.session_state.current_bm25_results) > 0:
            for i, chunk in enumerate(st.session_state.current_bm25_results[:3]):
                st.text_area(
                    f"Фрагмент {i+1}",
                    value=chunk[:2000],
                    height=200,
                    key=f"bm25_result_{i}"
                )
        else:
            st.info("Нет релевантных фрагментов")
    else:
        st.info("Выполните поиск для отображения результатов")

    # Блок информации о загруженном индексе (для отладки)
    with st.expander("Информация об индексе", expanded=False):
        if st.session_state.get('bm25_index'):
            st.success("Индекс BM25 успешно загружен")
            st.write(f"Всего фрагментов: {len(st.session_state.original_chunks)}")
            
            # Проверка чтения файла индекса
            try:
                BM25_INDEX_PATH = os.path.join(REPO_ROOT, "data", "bm25_index.pkl")
                file_size = os.path.getsize(BM25_INDEX_PATH)
                st.write(f"Размер файла индекса: {file_size} байт")
                st.write(f"Путь: {BM25_INDEX_PATH}")
            except Exception as e:
                st.error(f"Ошибка проверки файла индекса: {str(e)}")
        else:
            st.error("Индекс BM25 не загружен")
    
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
