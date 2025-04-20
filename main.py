import os
import re
import time
import chardet
import requests
import numpy as np  # Добавлено здесь
import streamlit as st
from docx import Document
from PyPDF2 import PdfReader
from typing import List, Optional
from rank_bm25 import BM25Okapi
from config import API_KEY, API_URL

# Конфигурация приложения
#API_URL = "your_api_endpoint"
#API_KEY = "your_api_key"
SYSTEM_PROMPT = "Ты юрист-консультант. Отвечай доброжелательно и структурированно. Запрещено выдумывать законы и судебные решения. Оперируй только известной информацией из контекста USER_CONTEXT."
INITIAL_USER_CONTEXT = "USER_CONTEXT: "
API_TIMEOUT = 60
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000

# Инициализация состояния сессии
def initialize_session():
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = ""
    if "user_context" not in st.session_state:
        st.session_state.user_context = INITIAL_USER_CONTEXT
    if "user_input" not in st.session_state:  # Правильный ключ
        st.session_state.user_input = ""

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
    """Создание BM25 индекса с сохранением оригинальных текстов"""
    all_chunks = []
    original_texts = []  # Сохраняем оригинальные тексты чанков
    
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
                original_texts.extend(chunks)  # Сохраняем оригинальные чанки
            except Exception as e:
                st.error(f"Ошибка чтения {filename}: {str(e)}")
                continue

        if not all_chunks:
            st.error("Нет данных для индексации!")
            return None, None

        tokenized_chunks = [doc.split() for doc in all_chunks]
        return BM25Okapi(tokenized_chunks, k1=1.8, b=0.75), original_texts  # Возвращаем оба объекта

    except Exception as e:
        st.error(f"Критическая ошибка при создании индекса: {str(e)}")
        return None, None

def file_to_text(uploaded_file) -> Optional[str]:
    """Конвертация файла в текст с обработкой ошибок"""
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
    # Удаление гласных окончаний
    while len(word) > 0 and word[-1] in 'аеёийоуыэюя':
        word = word[:-1]
    return word

def extract_keywords(text: str, bm25: BM25Okapi) -> List[str]:
    """Извлечение ключевых слов с фильтрацией"""
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
        # Сброс предыдущего состояния
        st.session_state.user_context = INITIAL_USER_CONTEXT
        # Конвертация файла
        file_text = file_to_text(uploaded_file)
        if not file_text:
            st.stop()
        # Сохраняем текст документа в сессии
        st.session_state.document_text = file_text  # <-- НОВАЯ СТРОКА
        
        # Создание индекса
        st.session_state.bm25_index, st.session_state.original_chunks = create_bm25_index()
        if not st.session_state.bm25_index or not st.session_state.original_chunks:
            st.stop()
        # Извлечение ключевых слов
        keywords = extract_keywords(file_text, st.session_state.bm25_index)
        if not keywords:
            st.error("Не удалось извлечь ключевые слова")
            st.stop()
        # Обновление контекста
        st.session_state.user_context += f"Ключевые термины: {', '.join(keywords)}"
    # Проверка ключевых слов
    st.write("## Отладка")
    st.write("Ключевые слова:", keywords)
    st.write("Количество чанков:", len(st.session_state.original_chunks))
        # В блоке поиска замените текущий код на этот:
    try:
        #Взвешивание через повторение терминов
        query_weights = {term: 2 for term in keywords}  # Используем целые веса
        weighted_query = []
        for term, weight in query_weights.items():
            weighted_query.extend([term] * weight)
    
        # Получаем оценки как numpy массив
        doc_scores = np.array(st.session_state.bm25_index.get_scores(weighted_query))
    
        # Фильтрация и сортировка
        sorted_indices = sorted(
            range(len(doc_scores)), 
            key=lambda i: doc_scores[i], 
            reverse=True
        )
    
        # Явное сравнение с плавающей точкой
        top_indices = [i for i in sorted_indices if doc_scores[i] > 0.0][:5]
    
        top_chunks = [st.session_state.original_chunks[i] for i in top_indices]
         # Отображение информации
        st.subheader("Контекст анализа:")
        st.write(st.session_state.user_context)

        st.subheader("Релевантные фрагменты:")
        relevant_chunks = []
        for i, chunk in enumerate(top_chunks):
            relevant_chunks.append(chunk)
            st.text_area(f"Фрагмент {i+1}", value=chunk[:5000], height=150)

    except Exception as e:
        st.error(f"Ошибка поиска: {str(e)}")
        st.stop()

# Блок чата
chat_container = st.container()
with chat_container:
    if uploaded_file:
        st.subheader("Контекст анализа:")
        st.write(st.session_state.user_context)
    
    # История диалога - ИЗМЕНИЛИ КЛЮЧ
    st.subheader("История консультаций")
    st.text_area("Лог переговоров", 
                value=st.session_state.chat_log, 
                height=300,
                key="chat_history_display",  # Уникальный ключ
                disabled=True)

# Блок ввода пользователя
#input_container = st.container()
#with input_container:
#    user_input = st.text_area(
#        "Введите ваш вопрос:", 
#        height=150,
#        max_chars=600,
#        key="user_input_field",  # Уникальный ключ
#        help="Максимум 600 символов"
#    )
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
            
    # Очищаем поле ввода
    #st.session_state.user_input_field = ""  # Используем тот же ключ, что и в text_area
    st.session_state.user_input = ""
    st.experimental_rerun()  # Принудительное обновление
# Отображение истории
st.subheader("История консультаций")
st.text_area("Лог переговоров", 
           value=st.session_state.chat_log, 
           height=300,
           key="chat_history")
