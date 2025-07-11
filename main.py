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
SYSTEM_PROMPT = """
Вы - опытный юрист, специализирующийся на подготовке правовых заключений. 
На основе предоставленных данных подготовьте подробное правовое заключение по запросу пользователя:

**Запрос клиента:**
{user_query}

**Контекст:**
Извлеченные данные поиска:
{context}


**Требования к заключению:**
1. Проведите детальный анализ правовой проблемы
2. Ссылайтесь на конкретные нормы законов и подзаконных актов (запрещено приводить несуществующие нормы)
3. Учитывайте релевантную судебную практику (запрещено приводить несуществующую судебную практику)
4. Структурируйте заключение по следующему плану:
   - Поставленный на исследование вопрос, соответствующий {user_query}
   - Фактические обстоятельства дела
   - Правовая квалификация ситуации и анализ применимых норм права
   - Выводы и рекомендации
5. Избегайте избыточной информации, не относящейся к делу
6. Используйте профессиональную юридическую терминологию
7. Объем заключения: не менее 5000 знаков (не указывайте в ответе объем)

**Важно:** Заключение должно быть готово к использованию в суде или для представления клиенту.
"""

API_TIMEOUT = 60
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000

def initialize_session():
    required_keys = {
        "chat_log": "",
        "user_input": "",
        "document_text": "",
        "document_keywords": [],
        "document_relevant_chunks": [],
        "query_keywords": [],
        "query_relevant_chunks": [],
        "llm_response": "",  # Добавлено хранилище для ответа
        "last_query": ""     # Для отслеживания последнего запроса
    }
    for key in required_keys:
        if key not in st.session_state:
            st.session_state[key] = required_keys[key]

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
    """Создание BM25 индекса на основе документов в папке"""
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
    """Извлечение ключевых слов с учетом BM25"""
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

def search_relevant_chunks(bm25: BM25Okapi, original_chunks: List[str], keywords: List[str]) -> List[str]:
    """Поиск релевантных фрагментов"""
    try:
        query_weights = {term: 2 for term in keywords}
        weighted_query = []
        for term, weight in query_weights.items():
            weighted_query.extend([term] * weight)
        
        doc_scores = np.array(bm25.get_scores(weighted_query))
        sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
        return [original_chunks[i] for i in sorted_indices if doc_scores[i] > 0.0][:5]
    
    except Exception as e:
        st.error(f"Ошибка поиска: {str(e)}")
        return []

# Интерфейс
st.title("Юридический консультант AI")
uploaded_file = st.file_uploader("Загрузите документ (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Анализ документа..."):
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        
        st.session_state.document_text = file_text
        bm25_index, original_chunks = create_bm25_index()
        
        if not bm25_index or not original_chunks:
            st.stop()
        
        keywords = extract_keywords(file_text, bm25_index)
        if not keywords:
            st.error("Не удалось извлечь ключевые слова")
            st.stop()
        
        st.session_state.document_keywords = keywords
        st.session_state.document_relevant_chunks = search_relevant_chunks(bm25_index, original_chunks, keywords)
        
        if st.session_state.document_relevant_chunks:
            st.subheader("Релевантные фрагменты из документа:")
            for i, chunk in enumerate(st.session_state.document_relevant_chunks):
                st.text_area(f"Фрагмент {i+1}", value=chunk[:5000], height=150, key=f"doc_chunk_{i}")

# Блок чата
user_input = st.text_area(
    "Введите ваш вопрос:", 
    height=150,
    max_chars=600,
    key="user_input"
)

if st.button("Отправить"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()
    st.session_state.last_query = user_input
    
    with st.spinner("Обработка запроса..."):
        # Создание индекса и обработка запроса
        bm25_index, original_chunks = create_bm25_index()
        if not bm25_index or not original_chunks:
            st.error("Не удалось создать поисковый индекс")
            st.stop()
        
        # Извлечение ключевых слов из запроса
        query_keywords = extract_keywords(user_input, bm25_index)
        if not query_keywords:
            st.error("Не удалось извлечь ключевые слова из запроса")
            st.stop()
        
        # Поиск релевантных фрагментов
        query_chunks = search_relevant_chunks(bm25_index, original_chunks, query_keywords)
        st.session_state.query_relevant_chunks = query_chunks
        
        
        # Формирование КОРРЕКТНОГО контекста
        context_parts = []
        if st.session_state.document_relevant_chunks:
            context_parts.append(
                "Контекст из документа:\n" + 
                "\n\n".join(st.session_state.document_relevant_chunks[:3])  # Ограничиваем количество
            )
        
        if query_chunks:
            context_parts.append(
                "Контекст из базы знаний:\n" + 
                "\n\n".join(query_chunks[:3])  # Ограничиваем количество
            )
        
        full_context = "\n\n".join(context_parts)
        
        # Формирование ПРАВИЛЬНОГО запроса к LLM
        messages = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT.format(
                    user_query=user_input,       # Совпадает с {user_query} в SYSTEM_PROMPT
                    context=full_context         # Совпадает с {context} в SYSTEM_PROMPT
                )
            },
            {
                "role": "user", 
                "content": "Сформируйте подробное юридическое заключение."
            }
        ]
        
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 5000  # Добавьте при необходимости
                },
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Универсальное извлечение ответа для разных API
            if 'choices' in response_data:
                answer = response_data['choices'][0]['message']['content']
            elif 'candidates' in response_data:  # Для Google Gemini
                answer = response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                answer = str(response_data)  # Фолбэк для отладки
            
            # Сохраняем ответ
            st.session_state.llm_response = answer
            st.session_state.chat_log += f"\nПользователь: {user_input}\nАссистент: {answer}"
            
        except Exception as e:
            st.error(f"Ошибка API: {str(e)}")
            if 'response' in locals():
                st.error(f"Тело ответа: {response.text}")

# Отображение ответа ПОСЛЕ обработки кнопки
if st.session_state.llm_response and st.session_state.last_query == user_input:
    st.subheader("Ответ юридического ассистента:")
    st.markdown(st.session_state.llm_response)
    
    # Отображение релевантных фрагментов с УНИКАЛЬНЫМИ ключами
    if st.session_state.query_relevant_chunks:
        st.subheader("Релевантные фрагменты из базы знаний:")
        for i, chunk in enumerate(st.session_state.query_relevant_chunks):
            unique_key = f"chunk_{int(time.time())}_{i}"
            st.text_area(label="", value=chunk[:2000], height=150, key=unique_key)

# Обновленный блок истории
if st.session_state.chat_log:
    st.subheader("История диалога")
    st.text_area(label="", value=st.session_state.chat_log, height=300, key="chat_history", disabled=True)
