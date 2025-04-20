import os
import re
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
        "bm25_index": None,
        "original_chunks": [],
        "chat_log": "",
        "user_context": INITIAL_USER_CONTEXT,
        "document_text": "",
        "relevant_chunks": []
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
    """Создание BM25 индекса на основе документов в папке 'documents'"""
    all_chunks = []
    original_texts = []
    
    try:
        if not os.path.exists("documents"):
            os.makedirs("documents")

        txt_files = [f for f in os.listdir("documents") if f.endswith(".txt")]
        if not txt_files:
            return None, None

        for filename in txt_files:
            file_path = os.path.join("documents", filename)
            try:
                encoding = detect_file_encoding(file_path)
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    text = f.read()
                chunks = process_text(text)
                all_chunks.extend(chunks)
                original_texts.extend(chunks)
            except Exception as e:
                st.error(f"Ошибка чтения {filename}: {str(e)}")
                continue

        if not all_chunks:
            return None, None

        tokenized_chunks = [doc.split() for doc in all_chunks]
        return BM25Okapi(tokenized_chunks, k1=1.8, b=0.75), original_texts

    except Exception as e:
        st.error(f"Ошибка при создании индекса: {str(e)}")
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
        return None
    except Exception as e:
        st.error(f"Ошибка обработки файла: {str(e)}")
        return None

def clean_keyword(word: str) -> str:
    """Очистка ключевых слов"""
    while len(word) > 0 and word[-1] in 'аеёийоуыэюя':
        word = word[:-1]
    return word

def extract_keywords(text: str, bm25: Optional[BM25Okapi] = None) -> List[str]:
    """Извлечение ключевых слов с использованием BM25 или частоты"""
    try:
        words = re.findall(r'\b[а-яё]+\b', text.lower())
        stop_words = {"на", "под", "в", "среди", "перед", "затем", "после", "до", "сразу"}
        
        filtered = [
            word for word in words
            if len(word) >= 5 
            and word not in stop_words
            and not re.search(r'\d', word)
        ]

        if not filtered:
            return []

        if bm25 is not None:
            scores = bm25.get_scores(filtered)
            scored_words = sorted(zip(filtered, scores), key=lambda x: x[1], reverse=True)
        else:
            freq = {}
            for word in filtered:
                freq[word] = freq.get(word, 0) + 1
            scored_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            scored_words = [(word, freq) for word, freq in scored_words]

        unique_words = []
        seen = set()
        for item in scored_words:
            word = item[0] if isinstance(item, tuple) else item
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
        if file_text:
            st.session_state.document_text = file_text
            st.success("Документ успешно загружен!")
        else:
            st.session_state.document_text = ""

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

    # Обновляем историю чата
    st.session_state.chat_log += f"\nПользователь: {user_input.strip()}"
    
    # Создаем BM25 индекс
    with st.spinner("Создание индекса..."):
        bm25_index, original_chunks = create_bm25_index()
    
    # Определяем источник для ключевых слов
    source_text = st.session_state.document_text if st.session_state.document_text else user_input
    
    # Извлекаем ключевые слова
    with st.spinner("Извлечение ключевых слов..."):
        keywords = extract_keywords(source_text, bm25_index)
    
    # Формируем контекст пользователя
    user_context = INITIAL_USER_CONTEXT
    if keywords:
        user_context += f"Ключевые термины: {', '.join(keywords)}"
    
    # Поиск релевантных фрагментов
    relevant_chunks = []
    if bm25_index and original_chunks:
        with st.spinner("Поиск релевантных фрагментов..."):
            try:
                query_weights = {term: 2 for term in keywords}
                weighted_query = []
                for term, weight in query_weights.items():
                    weighted_query.extend([term] * weight)
                
                doc_scores = np.array(bm25_index.get_scores(weighted_query))
                sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
                top_indices = [i for i in sorted_indices if doc_scores[i] > 0][:5]
                relevant_chunks = [original_chunks[i] for i in top_indices]
            except Exception as e:
                relevant_chunks = []
    
    # Формируем запрос к LLM
    with st.spinner("Формирование ответа..."):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context}
            ]

            if relevant_chunks:
                messages.append({
                    "role": "assistant",
                    "content": "Релевантные фрагменты документов:\n" + "\n\n".join(relevant_chunks[:3])
                })
            
            messages.append({"role": "user", "content": user_input.strip()})

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
            st.session_state.chat_log += f"\nАссистент: {answer}"
            
            # Отображение ответа
            st.subheader("Ответ:")
            st.write(answer)
            
            # Отображение релевантных фрагментов
            if relevant_chunks:
                st.subheader("Релевантные фрагменты:")
                for i, chunk in enumerate(relevant_chunks):
                    st.text_area(f"Фрагмент {i+1}", value=chunk[:5000], height=150, disabled=True)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка API: {str(e)}")
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

# Отображение истории чата
st.subheader("История диалога")
st.text_area("Лог", value=st.session_state.chat_log, height=300, key="chat_history", disabled=True)

if st.button("Очистить историю"):
    st.session_state.chat_log = ""
    st.rerun()
