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
from config import GEMINI_API_KEY, API_URL
import logging
import random
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from urllib.parse import unquote, urlparse, parse_qs
from typing import List, Dict, Any
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



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

QUERY_GENERATION_PROMPT = """
Как опытный юрист, сгенерируй 3-5 дополнительных уточняющих запросов для поиска правовой информации 
на основе ключевых терминов из исходного запроса. Запросы должны быть краткими (5-10 слов) и 
охватывать различные аспекты проблемы.

**Исходный запрос:**
{user_query}

**Ключевые термины:**
{keywords}

**Требования:**
1. Каждый запрос должен быть самостоятельным вопросом или тезисом
2. Используй профессиональную юридическую терминологию
3. Избегай дублирования исходного запроса
"""

API_TIMEOUT = 60
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000


def check_gemini_api_key():
    test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro?key={GEMINI_API_KEY}"
    try:
        response = requests.get(test_url, timeout=10)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

# В начале скрипта
if not check_gemini_api_key():
    st.error("⚠️ Неверный API ключ для Gemini. Пожалуйста, проверьте конфигурацию.")
    st.stop()

# ДОБАВЛЯЕМ КЛАСС ДЛЯ ВЕБ-ПОИСКА
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.78 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
]

class WebSearcher:
    def __init__(self, delay_range=(1.0, 3.0)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        
        # Настройки Google CSE (ЗАМЕНИТЕ НА СВОИ КЛЮЧИ!)
        self.api_key = "AIzaSyCNVeNmUgrt-kL5ZI4EkHFoTjTzRSWATX4"
        self.cse_id = "a4f17489c6a0a4414"
        
    def perform_search(self, query: str, max_results: int = 3) -> List[Dict]:
        """Выполняет поиск через Google Custom Search API"""
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': query,
                'num': max_results,
                'lr': 'lang_ru',
                'cr': 'countryRU',
                'hl': 'ru'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('items', [])[:max_results]:
                # Получаем полный текст страницы
                full_content = self.get_full_page_content(item.get('link', '#'))
                
                results.append({
                    'title': item.get('title', 'Без названия')[:150],
                    'url': item.get('link', '#'),
                    'snippet': item.get('snippet', 'Без описания')[:500],
                    'full_content': full_content[:5000]  # Ограничиваем объем
                })
            
            return results
        except Exception as e:
            logger.error(f"Ошибка Google CSE: {str(e)}")
            return []

    def get_full_page_content(self, url: str) -> str:
        """Получение полного текста страницы"""
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Определяем кодировку
            if response.encoding == 'ISO-8859-1':
                response.encoding = 'utf-8'
            
            # Парсим основной контент
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Удаляем ненужные элементы
            for tag in soup(['script', 'style', 'footer', 'nav', 'aside', 'header']):
                tag.decompose()
            
            # Извлекаем текст
            text = soup.get_text(separator=' ', strip=True)
            
            # Удаляем лишние пробелы
            text = re.sub(r'\s+', ' ', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Ошибка получения контента: {str(e)}")
            return ""

# ИНИЦИАЛИЗАЦИЯ СЕССИИ (ОБНОВЛЕННАЯ)
def initialize_session():
    """Инициализация всех необходимых переменных в session_state"""
    default_state = {
        "chat_log": "",
        "user_input": "",
        "document_text": "",
        "document_keywords": [],
        "document_relevant_chunks": [],
        "query_keywords": [],
        "query_relevant_chunks": [],
        "llm_response": "",
        "last_query": "",
        "web_searcher": WebSearcher(),
        "web_search_results": [],
        "web_search_chunks": [],
        "generated_queries": [],
        "additional_chunks": []
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
def generate_queries(user_query: str, keywords: List[str]) -> List[str]:
    """Генерация уточняющих запросов с помощью LLM"""
    try:
        # Формирование системного промпта
        prompt = QUERY_GENERATION_PROMPT.format(
            user_query=user_query,
            keywords=", ".join(keywords)
        )
        
        # Подготовка данных для запроса
        request_data = {
            "contents": [
                {
                    "parts": [
                        {"text": SYSTEM_PROMPT.format(
                            user_query=user_query,  # Исправлено: было user_input
                            context=keywords  # Исправлено: было full_context
                        )}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 5000
            }
        }
        
        # Выполнение запроса
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},  # Ключ передается как параметр
            json=request_data,  # Используем json вместо data
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        
        # Получение и обработка ответа
        response_data = response.json()  # Добавлено: получение данных ответа
        
        # Извлечение запросов из нумерованного списка
        queries = []
        content = ""
        
        if 'choices' in response_data:
            content = response_data['choices'][0]['message']['content']
        elif 'candidates' in response_data:
            content = response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            content = str(response_data)
            
        for line in content.split('\n'):
            if re.match(r'^\d+[\.\)]', line.strip()):
                query = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                if query:
                    queries.append(query)
                    
        return queries[:5]  # Ограничиваем 5 запросами
        
    except Exception as e:
        st.error(f"Ошибка генерации запросов: {str(e)}")
        return []

def get_unique_chunks(main_chunks: List[str], new_chunks: List[str]) -> List[str]:
    """Фильтрация дублирующихся фрагментов с порогом схожести"""
    unique_chunks = []
    for new_chunk in new_chunks:
        is_duplicate = False
        for main_chunk in main_chunks:
            # Считаем дубликатом если >80% содержимого совпадает
            if SequenceMatcher(None, main_chunk[:1000], new_chunk[:1000]).ratio() > 0.8:
                is_duplicate = True
                break
        if not is_duplicate and new_chunk not in unique_chunks:
            unique_chunks.append(new_chunk)
    return unique_chunks
   
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
    key="user_input_unique"  # Фиксированный ключ
)

# Кнопка с фиксированным ключом
if st.button("Отправить", key="send_button_unique"):
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
        
        if st.button("Отправить"):
            if not user_input.strip():
                st.error("Введите текст вопроса")
                st.stop()
            st.session_state.last_query = user_input
    
        
        # ШАГ 1: Генерация дополнительных запросов
        generated_queries = generate_queries(user_input, query_keywords)
        st.session_state.generated_queries = generated_queries  # Сохраняем для отображения
        
        # ШАГ 2: Поиск по сгенерированным запросам
        all_knowledge_chunks = st.session_state.query_relevant_chunks.copy()
        additional_chunks = []
        
        for query in generated_queries:
            with st.spinner(f"Поиск по запросу: '{query[:30]}...'"):
                # Извлечение ключевых слов для сгенерированного запроса
                q_keywords = extract_keywords(query, bm25_index)
                if not q_keywords:
                    continue
                
                # Поиск релевантных фрагментов
                q_chunks = search_relevant_chunks(bm25_index, original_chunks, q_keywords)
                
                # Фильтрация дубликатов
                unique_chunks = get_unique_chunks(all_knowledge_chunks, q_chunks)
                additional_chunks.extend(unique_chunks)
                all_knowledge_chunks.extend(unique_chunks)
        
        st.session_state.additional_chunks = additional_chunks  # Сохраняем для отображения
        
        # Формирование контекста с расширенными данными
        context_parts = []
        
        # Контекст из документа (если есть)
        if st.session_state.document_relevant_chunks:
            context_parts.append(
                "Контекст из документа:\n" + 
                "\n\n".join(st.session_state.document_relevant_chunks[:3])
            )
        
        # Основной контекст из базы знаний
        if st.session_state.query_relevant_chunks:
            context_parts.append(
                "Основной контекст из базы знаний:\n" + 
                "\n\n".join(st.session_state.query_relevant_chunks[:3])
            )
        
        # Дополнительный контекст из сгенерированных запросов
        if additional_chunks:
            context_parts.append(
                "Дополнительный контекст из базы знаний:\n" + 
                "\n\n".join(additional_chunks[:3])
            )

        
        # ШАГ 3: ВЕБ-ПОИСК ПО СГЕНЕРИРОВАННЫМ ЗАПРОСАМ
        web_results = []
        for query in generated_queries:
            with st.spinner(f"Веб-поиск: '{query[:30]}...'"):
                results = st.session_state.web_searcher.perform_search(query)
                web_results.extend(results)
        
        # Извлекаем фрагменты контента
        web_chunks = [result['full_content'] for result in web_results if result['full_content']]
        
        # Фильтруем дубликаты
        unique_web_chunks = []
        seen_chunks = set()
        for chunk in web_chunks:
            # Хэшируем для быстрого сравнения
            chunk_hash = hash(chunk[:1000])
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                unique_web_chunks.append(chunk)
        
        st.session_state.web_search_results = web_results
        st.session_state.web_search_chunks = unique_web_chunks[:3]  # Берем 3 уникальных фрагмента
        
        # ДОБАВЛЯЕМ ВЕБ-ФРАГМЕНТЫ В КОНТЕКСТ
        if st.session_state.web_search_chunks:
            context_parts.append(
                "Контекст из веб-поиска:\n" + 
                "\n\n".join(st.session_state.web_search_chunks)
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
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
                json={
                    "model": "google/gemini-2.5-flash-lite-preview-06-17",
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
if st.session_state.get('llm_response') and st.session_state.get('last_query') == user_input:
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

# Обновленный блок истории
if st.session_state.chat_log:
    st.subheader("История диалога")
    st.text_area(label="", value=st.session_state.chat_log, height=300, key="chat_history", disabled=True)
