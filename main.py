import streamlit as st
import time
import os
from utils import initialize_session, setup_logging
from config import GEMINI_API_KEY
from web_search import WebSearcher
from .query_generator import QueryGenerator
from indexing import IndexBuilder
from processing import DataProcessor
from utils import file_to_text, clean_text
from prompts import get_prompt
from api import check_gemini_api_key, send_gemini_request

# Инициализация
logger = setup_logging()
initialize_session()

# Проверка API ключа
if not check_gemini_api_key():
    st.error("⚠️ Неверный API ключ для Gemini. Пожалуйста, проверьте конфигурацию.")
    st.stop()

# Инициализация компонентов
index_builder = IndexBuilder()
data_processor = DataProcessor(index_builder)
query_generator = QueryGenerator()
web_searcher = WebSearcher()

# Интерфейс
st.title("Юридический консультант AI")
uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Анализ документа..."):
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        
        # Обработка документа
        st.session_state.document_text = clean_text(file_text)
        bm25_index, original_chunks = index_builder.create_bm25_index()
        
        if not bm25_index:
            st.error("Не удалось создать поисковый индекс")
            st.stop()
        
        keywords = data_processor.extract_keywords(st.session_state.document_text, bm25_index)
        st.session_state.document_keywords = keywords
        st.session_state.document_relevant_chunks = data_processor.search_relevant_chunks(
            bm25_index, original_chunks, keywords)
        
        # Отображение результатов
        if st.session_state.document_relevant_chunks:
            st.subheader("Релевантные фрагменты из документа:")
            for i, chunk in enumerate(st.session_state.document_relevant_chunks):
                st.text_area(f"Фрагмент {i+1}", value=chunk[:5000], height=150)

# Блок чата
user_input = st.text_area("Введите ваш вопрос:", height=150, max_chars=600)

if st.button("Отправить"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    
    st.session_state.last_query = user_input
    
    with st.spinner("Обработка запроса..."):
        # Шаг 1: Подготовка данных
        bm25_index, original_chunks = index_builder.create_bm25_index()
        query_keywords = data_processor.extract_keywords(user_input, bm25_index)
        
        # Шаг 2: Генерация дополнительных запросов
        generated_queries = query_generator.generate(user_input, query_keywords)
        st.session_state.generated_queries = generated_queries
        
        # Шаг 3: Поиск по сгенерированным запросам
        # [Логика поиска и сбора данных]
        
        # Шаг 4: Построение контекста
        full_context = data_processor.build_context(
            st.session_state.document_relevant_chunks,
            st.session_state.query_relevant_chunks,
            additional_chunks
        )
        
        # Шаг 5: Формирование и отправка промпта
        prompt = get_prompt("system", {
            "user_query": user_input,
            "context": full_context
        })
        
        response = send_gemini_request(prompt)
        st.session_state.llm_response = response
        
    # Отображение результатов
    st.subheader("Ответ юридического ассистента:")
    st.markdown(st.session_state.llm_response)
    
    # Отображение релевантных фрагментов с УНИКАЛЬНЫМИ ключами
    if st.session_state.get('query_relevant_chunks'):
        st.subheader("Релевантные фрагменты из базы знаний:")
        for i, chunk in enumerate(st.session_state.query_relevant_chunks):
            unique_key = f"chunk_{int(time.time())}_{i}"
            st.text_area(label="", value=chunk[:2000], height=150, key=unique_key)


    # ВСТАВЛЯЕМ НОВЫЕ БЛОКИ ЗДЕСЬ
    if st.session_state.get('generated_queries'):
        st.subheader("Сгенерированные уточняющие запросы:")
        for i, query in enumerate(st.session_state.generated_queries):
            st.write(f"{i+1}. {query}")

    if st.session_state.get('additional_chunks'):
        st.subheader("Дополнительные релевантные фрагменты:")
        for i, chunk in enumerate(st.session_state.additional_chunks):
            unique_key = f"add_chunk_{int(time.time())}_{i}"
            st.text_area(label="", value=chunk[:2000], height=150, key=unique_key)

    # После блока с выводами LLM добавьте:
    if st.session_state.get('web_search_results'):
        st.subheader("Результаты веб-поиска")
    
        for i, result in enumerate(st.session_state.web_search_results):
            with st.expander(f"{i+1}. {result['title']}", expanded=False):
                st.markdown(f"**URL:** [{result['url']}]({result['url']})")
            
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image("https://via.placeholder.com/150?text=Preview", width=150)
                
                with col2:
                    st.markdown("**Сниппет:**")
                    st.info(result.get('snippet', ''))
            
                if result.get('full_content'):
                    st.markdown("**Извлеченное содержимое:**")
                    st.text_area("", 
                                value=result['full_content'][:3000] + ("..." if len(result['full_content']) > 3000 else ""), 
                                height=200,
                                key=f"web_content_{i}")

# Обновленный блок истории
if st.session_state.chat_log:
    st.subheader("История диалога")
    st.text_area(label="", value=st.session_state.chat_log, height=300, key="chat_history", disabled=True)
