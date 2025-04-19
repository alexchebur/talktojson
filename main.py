
import os
import re
import time
import requests
import streamlit as st
from rank_bm25 import BM25Okapi
from docx import Document
from PyPDF2 import PdfReader
from typing import List, Optional
try:
    from config import API_KEY, API_URL
except ImportError:
    st.error("Ошибка: Создайте файл config.py с переменными API_KEY и API_URL")
    API_KEY = ""
    API_URL = "https://api.vsegpt.ru/v1/chat/completions"

SYSTEM_PROMPT = "Ты юрист-консультант. Отвечай доброжелательно и структурированно. Запрещено выдумывать законы и судебные решения. Оперируй только известной информацией из контекста USER_CONTEXT."
USER_CONTEXT = "USER_CONTEXT: "
CHAT_LOG = ""
API_TIMEOUT = 60

# Session state initialization
if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None
if "chat_log" not in st.session_state:
    st.session_state.chat_log = CHAT_LOG
if "user_context" not in st.session_state:
    st.session_state.user_context = USER_CONTEXT

def process_text(text: str) -> List[str]:
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + 1500
        chunks.append(text[start:end])
        start += 1000  # 500 overlap
    return chunks

def create_bm25_index():
    """Create BM25 index from all txt files in documents folder"""
    all_chunks = []
    
    if not os.path.exists("documents"):
        os.makedirs("documents")
    
    txt_files = [f for f in os.listdir("documents") if f.endswith(".txt")]
    for i, filename in enumerate(txt_files):
        with open(os.path.join("documents", filename), "r", encoding="utf-8") as f:
            text = f.read()
            chunks = process_text(text)
            all_chunks.extend(chunks)
    
    tokenized_chunks = [doc.split() for doc in all_chunks]
    return BM25Okapi(tokenized_chunks)

def extract_keywords(text: str, bm25: BM25Okapi) -> List[str]:
    """Extract keywords using BM25 scoring with filters"""
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter words
    filtered = []
    stop_words = {"на", "под", "в", "среди", "перед", "затем", "после", "до", "сразу"}
    for word in words:
        if len(word) < 4:
            continue
        if word in stop_words:
            continue
        if re.match(r'^[^а-яё]*$', word):
            continue
        filtered.append(word)
    
    # Score words
    scores = bm25.get_scores(filtered)
    scored_words = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
    
    # Select top unique words
    keywords = []
    seen = set()
    for word, _ in scored_words:
        if word not in seen:
            seen.add(word)
            keywords.append(word)
            if len(keywords) == 15:
                break
    
    # Remove vowel endings
    cleaned = []
    for word in keywords:
        while len(word) > 0 and word[-1] in 'аеёиоуыэюя':
            word = word[:-1]
        cleaned.append(word)
    
    return cleaned

def file_to_text(uploaded_file) -> Optional[str]:
    """Convert uploaded file to text"""
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

# Streamlit UI
st.title("Юридический консультант")

# File uploader
uploaded_file = st.file_uploader("Загрузите документ", type=["txt", "docx", "pdf"])

if uploaded_file:
    with st.spinner("Обработка документа..."):
        # Save uploaded file
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        
        # Create BM25 index
        progress_bar = st.progress(0)
        st.session_state.bm25_index = create_bm25_index()
        progress_bar.progress(100)
        
        # Extract keywords
        keywords = extract_keywords(file_text, st.session_state.bm25_index)
        st.session_state.user_context += "Ключевые термины: " + ", ".join(keywords)
        
        # Find relevant chunks
        query = " ".join(keywords)
        tokenized_query = query.split()
        top_chunks = st.session_state.bm25_index.get_top_n(tokenized_query, [chunk for chunk in st.session_state.bm25_index.corpus], n=5)
        
        # Display context and chunks
        st.subheader("Контекст:")
        st.write(st.session_state.user_context)
        
        st.subheader("Релевантные фрагменты:")
        relevant_chunks = []
        for i, chunk in enumerate(top_chunks):
            chunk_text = " ".join(chunk)
            relevant_chunks.append(chunk_text)
            st.write(f"Фрагмент {i+1}: {chunk_text[:1500]}...")
        
        # Update context
        st.session_state.user_context += "\nРелевантные данные:\n" + "\n\n".join(relevant_chunks)
        
        # Prepare LLM request
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "your_model_name",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": st.session_state.user_context},
                {"role": "assistant", "content": st.session_state.chat_log}
            ],
            "temperature": 0.7
        }
        
        # Get LLM response
        try:
            response = requests.post(API_URL, json=data, headers=headers, timeout=API_TIMEOUT)
            response.raise_for_status()
            answer = response.json()['choices'][0]['message']['content']
            
            st.subheader("Ответ:")
            st.write(answer)
            
            # Update chat log
            st.session_state.chat_log += "\nАссистент: " + answer
            st.session_state.user_context = USER_CONTEXT  # Reset context
            
        except Exception as e:
            st.error(f"Ошибка запроса к API: {str(e)}")

# Display chat history
st.subheader("История диалога")
st.text_area("Лог чата", value=st.session_state.chat_log, height=300, key="chat_log_display")
