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
1. Проведите детальный анализ правовой проблемы, разбейте проблему на подзадачи и решайте пошагово
2. Ссылайтесь на конкретные нормы законов и подзаконных актов (запрещено приводить несуществующие нормы)
3. Учитывайте релевантную судебную практику (запрещено приводить несуществующую судебную практику)
4. Структурируйте заключение по следующему плану:
   - Поставленный на исследование вопрос, соответствующий {user_query}
   - Фактические обстоятельства дела
   - Правовая квалификация ситуации и анализ применимых норм права с перечнем нормативных актов, относящихся к проблеме
   - Выводы и рекомендации
5. Избегайте избыточной информации, не относящейся к делу. Запрещено ссылаться на нормативные акты или судебные решения, не относящиеся к вопросу, даже если они представлены в контексте
6. Используйте профессиональную юридическую терминологию
7. Не допускайте ошибок в указании реквизитов нормативных актов:
 7.1. Нормативные акты в сфере теплоснабжения:
   - Гражданский кодекс Российской Федерации
   - Федеральный закон от 29.07.2017 № 279-ФЗ "О внесении изменений в Федеральный закон "О теплоснабжении" и отдельные законодательные акты Российской Федерации по вопросам совершенствования системы отношений в сфере теплоснабжения"  
   - Федеральный закон от 27.07.2010 № 190-ФЗ "О теплоснабжении" 
   - Постановление Правительства РФ от 22.10.2012 № 1075 "О ценообразовании в сфере теплоснабжения"
   - Постановление Правительства РФ от 15.12.2017 № 1562 "Об определении в ценовых зонах теплоснабжения предельного уровня цены на тепловую энергию (мощность), включая индексацию предельного уровня цены на тепловую энергию (мощность), и технико-экономических параметров работы котельных и тепловых сетей, используемых для расчета предельного уровня цены на тепловую энергию (мощность)"
   - Постановление Правительства РФ от 23.07.2018 № 860 "Об отдельных вопросах ценообразования на тепловую энергию (мощность) в ценовых зонах теплоснабжения"
   - Постановление Правительства РФ от 08.08.2012 № 808 "Об организации теплоснабжения в Российской Федерации и о внесении изменений в некоторые акты Правительства Российской Федерации"
   - Постановление Правительства РФ от 30.11.2021 № 2115 "Об утверждении Правил подключения (технологического присоединения) к системам теплоснабжения, включая правила недискриминационного доступа к услугам по подключению (технологическому присоединению) к системам теплоснабжения, Правил недискриминационного доступа к услугам по передаче тепловой энергии, теплоносителя ..."
   - Постановление Правительства РФ от 22.02.2012 №154 "О требованиях к схемам теплоснабжения, порядку их разработки и утверждения"
   - Постановление Правительства РФ от 08.07.2023 № 1130 "Об утверждении правил вывода в ремонт и из эксплуатации источников тепловой энергии и тепловых сетей ..."
   - Постановление Правительства РФ от 14.02.2012 № 124 "О правилах, обязательных при заключении договоров снабжения коммунальными ресурсами"
   - Постановление Правительства РФ от 06.05.2011 № 354 "О предоставлении коммунальных услуг собственникам и пользователям помещений в многоквартирных домах и жилых домов"
   - Постановление Правительства РФ от 26.01.2023 № 110 "О стандартах раскрытия информации теплоснабжающими организациями, теплосетевыми организациями и органами регулирования тарифов в сфере теплоснабжения"
   - Приказ Минэнерго России от 06.09.2024 № 1250 "Об утверждении Порядка подготовки предложений об отнесении или неотнесении поселений, муниципальных округов, городских округов к ценовым зонам теплоснабжения"
   - Приказ Минэнерго России от 09.10.2024 № 1800 "Об утверждении методических рекомендаций по внедрению целевой модели рынка тепловой энергии на территории поселения, муниципального округа, городского округа"


8. Объем заключения: не менее 5000 знаков (не указывайте в ответе объем)

**Важно:** Заключение должно быть готово к использованию в суде или для представления клиенту.
"""

QUERY_GENERATION_PROMPT = """
Как опытный юрист, руководствуясь сформулированной проблемой, сгенерируй 3-5  уточняющих поисковых запросов в API Google Custom Search Engine для поиска правовой информации 
на основе ключевых терминов из исходного запроса. Запросы должны быть краткими (5-10 слов) и состоять из ключевых слов, используемых в веб-поиске, без ответов на запросы
охватывать различные аспекты проблемы.

**Исходный запрос:**
{user_query}

**Ключевые термины:**
{keywords}

**Требования:**
- Каждый запрос должен быть направлен на проверку гипотез
- Список запросов должен быть пронумерован: 1... 2...3..
- Используй профессиональную юридическую терминологию
- Пример списка поисковых запросов: 
1. Перечень документов для заключения договора теплоснабжения
2. Порядок заключения договора теплоснабжения
3. Документы для договора поставки тепловой энергии


"""

API_TIMEOUT = 60
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000


def check_gemini_api_key():
    test_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17?key={GEMINI_API_KEY}"
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
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.api_key,
                'cx': self.cse_id,
                'q': query,
                'num': max_results,
                'lr': 'lang_ru',
                'hl': 'ru'
            }
        
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
        
            results = []
            for item in data.get('items', [])[:max_results]:
                # ДОБАВЛЯЕМ ИЗВЛЕЧЕНИЕ ПОЛНОГО КОНТЕНТА
                full_content = self.get_full_page_content(item.get('link', ''))
            
                results.append({
                    'title': item.get('title', 'Без названия')[:150],
                    'url': item.get('link', '#'),
                    'snippet': item.get('snippet', 'Без описания')[:500],
                    'full_content': full_content  # Сохраняем полный контент
                })
        
            return results
        except Exception as e:
            logger.error(f"Ошибка Google CSE: {str(e)}")
            return []

    def get_full_page_content(url: str) -> str:
        """Получение полного текста страницы с улучшенным парсингом"""
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        
            # Определяем кодировку
            if response.encoding == 'ISO-8859-1':
                response.encoding = 'utf-8'
        
            # Упрощенный парсинг основного контента
            soup = BeautifulSoup(response.text, 'html.parser')
        
            # Удаляем ненужные элементы
            for tag in soup(['script', 'style', 'footer', 'nav', 'aside', 'header']):
                tag.decompose()
        
            # Удаляем пустые элементы
            for tag in soup.find_all():
                if len(tag.get_text(strip=True)) == 0:
                    tag.decompose()
        
            # Извлекаем текст
            text = ' '.join(soup.stripped_strings)
        
            # Удаляем лишние пробелы
            text = re.sub(r'\s+', ' ', text)
        
            return text[:15000]  # Ограничение до 15k символов
        
        except Exception as e:
            logger.error(f"Ошибка получения контента для {url}: {str(e)}")
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

def parse_generated_queries(llm_response: str) -> list[str]:
    """Извлекает уточняющие запросы из текста LLM"""
    queries = []
    # Для нумерованных списков (1. ... 2. ...)
    for line in llm_response.split('\n'):
        if re.match(r'^\d+[\.\)]\s*', line.strip()):
            query = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if query:
                queries.append(query)
    
    # Для маркированных списков (- ... * ...)
    if not queries:
        for line in llm_response.split('\n'):
            if re.match(r'^[\-\*]\s*', line.strip()):
                query = re.sub(r'^[\-\*]\s*', '', line).strip()
                if query:
                    queries.append(query)
    
    # Фильтрация пустых и слишком коротких запросов
    return [q for q in queries if len(q) > 10][:5]  # Возвращаем до 5 запросов



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
    key="user_input_unique"
)

def parse_generated_queries(llm_response: str) -> list[str]:
    """Извлекает уточняющие запросы из текста LLM"""
    queries = []
    for line in llm_response.split('\n'):
        if re.match(r'^\d+[\.\)]\s*', line.strip()):
            query = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if query and len(query) > 10:  # Исключаем слишком короткие
                queries.append(query)
    return queries[:5]  # Ограничиваем 5 запросами

def is_duplicate(content: str, existing: list[str]) -> bool:
    """Проверяет дубликаты контента"""
    if not content or not existing:
        return False
    content_sample = content[:500]
    return any(SequenceMatcher(None, content_sample, e[:500]).ratio() > 0.75 for e in existing)

if st.button("Отправить", key="send_button_unique"):
    if not user_input.strip():
        st.error("Введите текст вопроса")
        st.stop()

    st.session_state.last_query = user_input
    
    with st.spinner("🔍 Обработка запроса..."):
        # Инициализация поискового индекса
        bm25_index, original_chunks = create_bm25_index()
        if not bm25_index or not original_chunks:
            st.error("Не удалось создать поисковый индекс")
            st.stop()
        
        # Извлечение ключевых слов
        query_keywords = extract_keywords(user_input, bm25_index)
        if not query_keywords:
            st.error("Не удалось извлечь ключевые слова")
            st.stop()

        # ШАГ 1: Генерация уточняющих запросов
        generated_queries_raw = generate_queries(user_input, query_keywords)
        st.session_state.generated_queries = parse_generated_queries
        
        # ШАГ 2: Поиск по уточняющим запросам
        st.session_state.second_search_results = []
        additional_chunks = []
        
        if st.session_state.generated_queries:
            for query in st.session_state.generated_queries:
                with st.spinner(f"🔎 Уточняющий поиск: '{query[:30]}...'"):
                    # Поиск в локальной базе
                    q_keywords = extract_keywords(query, bm25_index)
                    if q_keywords:
                        q_chunks = search_relevant_chunks(bm25_index, original_chunks, q_keywords)
                        additional_chunks.extend(get_unique_chunks(st.session_state.query_relevant_chunks, q_chunks))
                    
                    # Веб-поиск
                    web_results = st.session_state.web_searcher.perform_search(query)
                    st.session_state.second_search_results.extend(web_results)
                    
                    # Добавление уникального контента
                    for result in web_results:
                        if result['full_content'] and not is_duplicate(result['full_content'], additional_chunks):
                            additional_chunks.append(result['full_content'][:10000])  # Обрезка длинных текстов

        st.session_state.additional_chunks = additional_chunks
        
        # Формирование контекста
        context_parts = []
        if st.session_state.document_relevant_chunks:
            context_parts.append("Документ:\n" + "\n\n".join(st.session_state.document_relevant_chunks[:3]))
        if st.session_state.query_relevant_chunks:
            context_parts.append("База знаний:\n" + "\n\n".join(st.session_state.query_relevant_chunks[:3]))
        if additional_chunks:
            context_parts.append("Доп.материалы:\n" + "\n\n".join(additional_chunks[:3]))
        
        full_context = "\n\n".join(context_parts)

        # Запрос к LLM
        full_prompt = SYSTEM_PROMPT.format(user_query=user_input, context=full_context)
        request_data = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 5000}
        }

        try:
            response = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                params={"key": GEMINI_API_KEY},
                json=request_data,
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
            if hasattr(e, 'response'):
                st.error(f"Тело ответа: {e.response.text}")

# Отображение результатов
if st.session_state.get('llm_response') and st.session_state.get('last_query') == user_input:
    st.subheader("📝 Ответ юридического ассистента:")
    st.markdown(st.session_state.llm_response)
    
    # Блок сгенерированных запросов
    if st.session_state.get('generated_queries'):
        with st.expander("🔍 Сгенерированные уточняющие запросы", expanded=False):
            for i, query in enumerate(st.session_state.generated_queries):
                st.markdown(f"{i+1}. `{query}`")
    
    # Блок дополнительных материалов
    if st.session_state.get('additional_chunks'):
        with st.expander("📂 Дополнительные материалы", expanded=False):
            for i, chunk in enumerate(st.session_state.additional_chunks[:3]):
                st.text_area(f"Материал {i+1}", value=chunk[:2000], height=150, key=f"add_chunk_{i}")
    
    # Блок результатов веб-поиска
    if st.session_state.get('second_search_results'):
        with st.expander("🌐 Результаты уточняющего поиска", expanded=False):
            for i, result in enumerate(st.session_state.second_search_results[:3]):
                st.markdown(f"### {i+1}. [{result['title']}]({result['url']})")
                if result.get('snippet'):
                    st.info(f"**Сниппет:** {result['snippet']}")
                if result.get('full_content'):
                    st.text_area("Контент", value=result['full_content'][:2000]+"...", 
                               height=200, key=f"web_result_{i}")

# Обновленный блок истории
if st.session_state.chat_log:
    st.subheader("История диалога")
    st.text_area(label="", value=st.session_state.chat_log, height=300, key="chat_history", disabled=True)
