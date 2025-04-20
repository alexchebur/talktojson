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
from config import API_KEY, API_URL

# Конфигурация приложения
SYSTEM_PROMPT = "Ты юрист-консультант. Отвечай доброжелательно и структурированно. Запрещено выдумывать законы и судебные решения. Оперируй только известной информацией из контекста USER_CONTEXT."
INITIAL_USER_CONTEXT = "USER_CONTEXT: "
API_TIMEOUT = 60
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000

def initialize_session():
    required_keys = {
        "chat_log": "",
        "user_context": INITIAL_USER_CONTEXT,
        "user_input": "",
        "document_text": "",
        "document_keywords": [],
        "document_chunks": [],
        "query_keywords": [],
        "query_chunks": []
    }
    for key, default in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

initialize_session()

def process_text(text: str) -> List[str]:
    """Разделение текста на чанки с перекрытием"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def detect_file_encoding(file_path: str) -> str:
    """Определение кодировки файла"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    return chardet.detect(raw_data)['encoding']

def create_bm25_index():
    """Создание BM25 индекса с учетом документов и временных данных"""
    all_chunks = []
    original_texts = []
    
    try:
        # Чтение документов из папки
        if os.path.exists("documents"):
            txt_files = [f for f in os.listdir("documents") if f.endswith(".txt")]
            for filename in txt_files:
                file_path = os.path.join("documents", filename)
                try:
                    encoding = detect_file_encoding(file_path)
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        text = f.read()
                    all_chunks.extend(process_text(text))
                    original_texts.extend(process_text(text))
                except Exception as e:
                    st.error(f"Ошибка чтения {filename}: {str(e)}")
        
        # Добавление временного документа из сессии
        if st.session_state.document_text:
            doc_chunks = process_text(st.session_state.document_text)
            all_chunks.extend(doc_chunks)
            original_texts.extend(doc_chunks)
        
        if not all_chunks:
            return None, None
        
        tokenized_chunks = [doc.split() for doc in all_chunks]
        return BM25Okapi(tokenized_chunks, k1=1.8, b=0.75), original_texts
    
    except Exception as e:
        st.error(f"Ошибка создания индекса: {str(e)}")
        return None, None

def file_to_text(uploaded_file) -> Optional[str]:
    """Конвертация файла в текст"""
    try:
        if uploaded_file.name.endswith('.txt'):
            return uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            return "\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
        return None

def clean_keyword(word: str) -> str:
    """Очистка ключевых слов"""
    while len(word) > 0 and word[-1] in 'аеёийоуыэюя':
        word = word[:-1]
    return word

def extract_keywords(text: str, bm25: BM25Okapi) -> List[str]:
    """Извлечение ключевых слов"""
    try:
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        stop_words = {"на", "под", "в", "среди", "перед", "затем", "после", "до", "сразу"}
        
        filtered = [
            word for word in words
            if len(word) >= 5 
            and word not in stop_words
            and not re.search(r'\d', word)
        ]

        scores = bm25.get_scores(filtered)
        scored_words = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
        
        unique_words = []
        seen = set()
        for word, _ in scored_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
                if len(unique_words) == 20:
                    break

        return [clean_keyword(word) for word in unique_words]
    except Exception as e:
        st.error(f"Ошибка извлечения ключевых слов: {str(e)}")
        return []

# Интерфейс Streamlit
st.title("Юридический консультант AI")
uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Анализ документа..."):
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        
        st.session_state.document_text = file_text
        bm25, original_chunks = create_bm25_index()
        
        if bm25 and original_chunks:
            keywords = extract_keywords(file_text, bm25)
            st.session_state.document_keywords = keywords
            
            query_weights = {term: 2 for term in keywords}
            weighted_query = []
            for term, weight in query_weights.items():
                weighted_query.extend([term] * weight)
            
            doc_scores = np.array(bm25.get_scores(weighted_query))
            sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
            top_chunks = [original_chunks[i] for i in sorted_indices[:5] if doc_scores[i] > 0]
            
            st.session_state.document_chunks = top_chunks
            st.session_state.user_context += f"Документные ключевые термины: {', '.join(keywords)}\n"
            
            st.subheader("Фрагменты документа:")
            for i, chunk in enumerate(top_chunks):
                st.text_area(f"Фрагмент {i+1}", value=chunk[:5000], height=150, key=f"doc_chunk_{i}")

# Блок чата
user_input = st.text_area(
    "Введите ваш вопрос:", 
    height=150,
    max_chars=600,
    key="user_input",
    help="Максимум 600 символов"
)   

if st.button("Отправить"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    
    with st.spinner("Обработка запроса..."):
        bm25, original_chunks = create_bm25_index()
        if not bm25 or not original_chunks:
            st.error("Нет данных для анализа. Добавьте документы в папку или загрузите файл.")
            st.stop()
        
        # Обработка пользовательского запроса
        query = user_input.strip()
        keywords = extract_keywords(query, bm25)
        st.session_state.query_keywords = keywords
        
        # Поиск релевантных фрагментов
        query_weights = {term: 2 for term in keywords}
        weighted_query = []
        for term, weight in query_weights.items():
            weighted_query.extend([term] * weight)
        
        doc_scores = np.array(bm25.get_scores(weighted_query))
        sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        top_chunks = [original_chunks[i] for i in sorted_indices[:5] if doc_scores[i] > 0]
        st.session_state.query_chunks = top_chunks
        
        # Формирование контекста
        context = SYSTEM_PROMPT + "\n\n"
        if st.session_state.document_keywords:
            context += f"Ключевые термины документа: {', '.join(st.session_state.document_keywords)}\n"
        if st.session_state.document_chunks:
            context += "Релевантные фрагменты документа:\n" + "\n".join(st.session_state.document_chunks[:3]) + "\n\n"
        
        context += f"Ключевые термины запроса: {', '.join(keywords)}\n"
        context += "Релевантные фрагменты запроса:\n" + "\n".join(top_chunks[:3])
        
        # Отправка запроса к LLM
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
                {"role": "user", "content": query}
            ]
            
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": messages,
                    "temperature": 0.3
                },
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            
            answer = response.json()['choices'][0]['message']['content']
            st.session_state.chat_log += f"\nПользователь: {query}\nАссистент: {answer}"
            
            st.subheader("Ответ:")
            st.write(answer)
            
        except Exception as e:
            st.error(f"Ошибка запроса: {str(e)}")

# Отображение истории
if st.session_state.chat_log:
    st.subheader("История консультаций")
    st.text_area("Лог переговоров", 
               value=st.session_state.chat_log, 
               height=300,
               key="chat_history")
