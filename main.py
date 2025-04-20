import os
import re
import time
import chardet
import requests
import numpy as np
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader
from typing import List, Optional
from rank_bm25 import BM25Okapi
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from config import API_KEY, API_URL

# Конфигурация приложения
SYSTEM_PROMPT = "Ты юрист-консультант. Отвечай доброжелательно и структурированно. Запрещено выдумывать законы и судебные решения. Оперируй только известной информацией из контекста USER_CONTEXT."
INITIAL_USER_CONTEXT = "USER_CONTEXT: "
API_TIMEOUT = 60
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 800
DOC_STOP_WORDS = {"на", "под", "в", "среди", "перед", "затем", "после", "до", "сразу"}
QUERY_STOP_WORDS = {"как", "что", "на", "под", "в", "со", "и", "или"}

morph = MorphAnalyzer()

def initialize_session():
    required_keys = {
        "bm25_index": None,
        "chat_log": "",
        "user_context": INITIAL_USER_CONTEXT,
        "user_input": "",
        "document_text": "",
        "idf_values": {},
        "lemmatized_chunks": []
    }
    for key, default in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

initialize_session()

def lemmatize_word(word: str) -> str:
    try:
        return morph.parse(word)[0].normal_form
    except:
        return word

def process_text(text: str) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def detect_file_encoding(file_path: str) -> str:
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read(10000))['encoding']

def create_bm25_index():
    try:
        if not os.path.exists("documents"):
            os.makedirs("documents")

        all_chunks = []
        lemmatized_chunks = []
        
        # Сбор всех терминов корпуса
        corpus_vocabulary = set()
        
        for filename in [f for f in os.listdir("documents") if f.endswith(".txt")]:
            file_path = os.path.join("documents", filename)
            try:
                with open(file_path, 'r', encoding=detect_file_encoding(file_path), errors='replace') as f:
                    text = f.read()
                
                chunks = process_text(text)
                all_chunks.extend(chunks)
                
                for chunk in chunks:
                    words = re.findall(r'\b[а-яё]+\b', chunk.lower())
                    lemmas = [lemmatize_word(w) for w in words 
                             if len(w) >= 3 and w not in DOC_STOP_WORDS]
                    lemmatized_chunks.append(lemmas)
                    corpus_vocabulary.update(lemmas)
                
            except Exception as e:
                st.error(f"Ошибка обработки {filename}: {str(e)}")
                continue

        if not lemmatized_chunks:
            st.error("Нет данных для индексации!")
            return None, None

        # Создаем BM25
        bm25 = BM25Okapi(lemmatized_chunks, k1=2.2, b=0.65)
        
        # Собираем IDF вручную
        idf = {}
        vocabulary = list(corpus_vocabulary)
        for i, term in enumerate(vocabulary):
            idf[term] = bm25.idf[i]
        
        st.session_state.idf_values = idf
        st.session_state.lemmatized_chunks = lemmatized_chunks
        
        return bm25, all_chunks

    except Exception as e:
        st.error(f"Ошибка создания индекса: {str(e)}")
        return None, None

def file_to_text(uploaded_file) -> Optional[str]:
    try:
        if uploaded_file.name.endswith('.txt'):
            return uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.name.endswith('.docx'):
            return "\n".join([p.text for p in Document(uploaded_file).paragraphs])
        elif uploaded_file.name.endswith('.pdf'):
            return "\n".join([p.extract_text() for p in PdfReader(uploaded_file).pages])
    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
        return None

def extract_keywords_tfidf(text: str) -> List[str]:
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50,
            stop_words=list(DOC_STOP_WORDS)
        )
        
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        sorted_indices = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
        return [feature_names[i] for i in sorted_indices[:20]]
    
    except Exception as e:
        st.error(f"Ошибка TF-IDF: {str(e)}")
        return []

def process_query(text: str) -> List[str]:
    words = re.findall(r'\b[а-яё]+\b', text.lower())
    query_terms = []
    
    for word in words:
        if len(word) < 3 or word in QUERY_STOP_WORDS:
            continue
        lemma = lemmatize_word(word)
        if lemma in st.session_state.idf_values:
            query_terms.append(lemma)
    
    return query_terms

# Интерфейс Streamlit
st.title("Юридический консультант AI")
uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Анализ документа..."):
        st.session_state.user_context = INITIAL_USER_CONTEXT
        file_text = file_to_text(uploaded_file)
        
        if file_text:
            st.session_state.document_text = file_text
            bm25_index, original_chunks = create_bm25_index()
            
            if bm25_index:
                st.session_state.bm25_index = bm25_index
                st.session_state.original_chunks = original_chunks
                
                keywords = extract_keywords_tfidf(file_text)
                if keywords:
                    st.session_state.user_context += f"Ключевые термины: {', '.join(keywords)}"
                    
                    st.write("## Отладка")
                    st.write("Ключевые слова:", keywords)
                    st.write("Количество чанков:", len(original_chunks))

# Поисковая часть
if uploaded_file and st.session_state.bm25_index:
    user_input = st.text_area("Введите ваш вопрос:", key="user_input")
    
    if st.button("Поиск"):
        with st.spinner("Поиск релевантных фрагментов..."):
            try:
                query_terms = process_query(user_input)
                
                if not query_terms:
                    st.error("Не удалось извлечь значимые термины из запроса")
                    st.stop()
                
                # Взвешивание терминов запроса
                weighted_query = []
                for term in query_terms:
                    weight = int(st.session_state.idf_values.get(term, 1)) + 1
                    weighted_query.extend([term] * weight)
                
                # Получение оценок
                doc_scores = np.array(st.session_state.bm25_index.get_scores(weighted_query))
                
                # Boost для точных совпадений
                for i, chunk in enumerate(st.session_state.original_chunks):
                    if user_input.strip().lower() in chunk.lower():
                        doc_scores[i] *= 1.5
                
                # Сортировка и выбор топ-5
                top_indices = np.argsort(doc_scores)[::-1][:5]
                top_chunks = [st.session_state.original_chunks[i] for i in top_indices if doc_scores[i] > 0]
                
                st.subheader("Релевантные фрагменты:")
                for i, chunk in enumerate(top_chunks):
                    st.text_area(f"Фрагмент {i+1}", value=chunk[:5000], height=150)
                    
            except Exception as e:
                st.error(f"Ошибка поиска: {str(e)}")

# Блок чата и остальная логика остается аналогичной оригинальной версии
        #st.stop()

# Блок чата
chat_container = st.container()
with chat_container:
    if uploaded_file:
        st.subheader("Контекст анализа:")
        #st.write(st.session_state.user_context)
    
# Виджет ввода
user_input = st.text_area(
    "Введите ваш вопрос:", 
    value=st.session_state.user_input,  # Привязка к состоянию
    height=150,
    max_chars=600,
    key="user_input",  # Совпадает с ключом в session_state
    help="Максимум 600 символов"
)   
col1, col2 = st.columns([1, 4])
with col1:
    send_button = st.button("Отправить", use_container_width=True, key="send_button_unique")  # Уникальный ключ




# Обработка запроса
if send_button and uploaded_file:
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    
    # Добавляем вопрос в контекст
    st.session_state.user_context += f"\n\nВопрос пользователя: {user_input.strip()}"
    
    # Обновляем историю
    st.session_state.chat_log += f"\nПользователь: {user_input.strip()}"
    
    # Выполняем запрос к LLM
    with st.spinner("Формируем ответ..."):
        try:
            # Формируем полный контекст
            full_context = (
                f"{st.session_state.user_context}\n\n"
                f"Полный текст документа:\n{st.session_state.document_text[:30000]}...\n\n"  # Ограничение в 30k символов
                f"Релевантные фрагменты:\n" + "\n\n".join(relevant_chunks)
            )
            
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_context},
                        {"role": "assistant", "content": st.session_state.chat_log}
                    ],
                    "temperature": 0.3
                },
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            
            answer = response.json()['choices'][0]['message']['content']
            
            # Обновляем интерфейс
            with chat_container:
                st.subheader("Юридическая оценка:")
                st.write(answer)
                
            # Сохраняем в историю
            st.session_state.chat_log += f"\nАссистент: {answer}"
            
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка API: {str(e)}")
            
    def clear_input():
        st.session_state.user_input = ""
        clear_input()  # Вызов функции очистки
        st.rerun()
# Отображение истории
st.subheader("История консультаций")
st.text_area("Лог переговоров", 
           value=st.session_state.chat_log, 
           height=300,
           key="chat_history")
