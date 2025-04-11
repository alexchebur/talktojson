# -*- coding: utf-8 -*-
import streamlit as st
import os
import re
import json
import io
from docx import Document
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
import requests
from typing import List, Dict, Any
import shutil
from pathlib import Path
import glob
from json import JSONDecodeError

try:
    from config import API_KEY, API_URL
except ImportError:
    st.error("Ошибка: Создайте файл config.py с переменными API_KEY и API_URL")
    API_KEY = ""
    API_URL = "https://api.vsegpt.ru/v1/chat/completions"

# Конфигурация
DATA_DIR = "data"
MAX_CONTEXT_LENGTH = 20000
MAX_ANSWER_LENGTH = 15000
TEMPERATURE = 0.2
os.makedirs(DATA_DIR, exist_ok=True)

# Промпт для генерации ключевых слов
KEYWORDS_PROMPT = """
Задача:
Преобразуй пользовательский запрос в оптимизированную форму для BM25-поиска в базе юридических документов (нормативных документов и образцов процессуальных документов) по ключевым словам и выражениям. Учитывай особенности юридической лексики и следующие требования:

Саммари:
Сформулируй одним предложением, о чем документ.

Семантическое расширение:
К десяти самым семантически значимым в запросе словам и выражениям в запросе добавь по 2-3 синонима/близко связанных слова/тождественных по смыслу выражения через ИЛИ:
"расторжение договора" → "расторжение ИЛИ прекращение ИЛИ аннулирование договора"
"теплоснабжение" → "поставка тепловой энергии"

Укажи альтернативные формулировки законов:
"ГК РФ" → "Гражданский кодекс РФ (ГК РФ)"

Контекстуализация:
Для общих понятий добавь конкретику:
"нарушение сроков" → "просрочка исполнения обязательств (ст. 395 ГК РФ)"
Укажи ближайшие смежные правовые аспекты:
"неустойка" → "неустойка (штраф, пеня)"

Сохранение структуры:
НЕ ИЗМЕНЯЙ номера статей/документов:
"ст. 15.25 КоАП" → "статья 15.25 Кодекса об административных правонарушениях (КоАП)"
Сохраняй специальные обозначения:
"№ 127-ФЗ" → "Федеральный закон № 127-ФЗ"

Обработка ошибок:
Исправь очевидные опечатки:
"Эллектронный документ" → "электронный документ"
Предложи варианты для неоднозначных терминов:
"иск" → "исковое заявление (ИСК) ИЛИ индивидуальный инвестиционный счет (ИИС)"

НЕ ПРИДУМЫВАЙ несуществующих нормативных или судебных актов.

Формат вывода:
Основные термины и выражения (включая исправленные)
Синонимы через "ИЛИ"
Уточняющие конструкции в скобках
Номера документов в полной форме
"""

# Системные промпты
SYSTEM_PROMPT = """Ты - профессиональный опытный юрист-литигатор из энергетической компании. Ты можешь критически оценивать
процессуальные документы, комментировать позиции и прогнозировать развитие споров. правила цитирования: ЗАПРЕЩЕНО цитировать и ссылаться на конкретные нормы, пункты, правила и конкретные законы,
отсутствующие в предоставленных документах. Все выводы и цитаты должны основываться на тексте предоставленных документов."""

BUTTON_PROMPTS = {
    "quality": """Ты юрист-литигатор в энергетической компании. Оцени сильные и слабые стороны процессуального документа НА ОСНОВЕ ЭТИХ ФРАГМЕНТОВ:\n{context}\n\n. Правила анализа: 
    1. Полноту, последовательность и структуру аргументации, 
    2. относимость и достаточность доказательств, 
    3. полноту указания в подкрепление аргументов применимых норм права. 
    4. суммарное количество недостатков в формате оценки от 1 до 10 ["Я бы поставил итоговую оценку документу ... из 10]
    5. правила цитирования: ЗАПРЕЩЕНО цитировать и ссылаться на конкретные нормы и конкретные законы, отсутствующие в предоставленных документах. Все выводы и цитаты должны основываться на тексте предоставленных документов.
    6.[ВАЖНО] критично оцени документ на предмет соответствия методическим указаниям по процессуальным документам из фрагментов  \n{context}\n\n  (если рекомендации представлены в документе), при выявлении нарушений - процитируй нарушенные требования методических указаний, если требования указаны в предоставленных документах.""",

    "strategy": """Ты - юрист-литигатор в энергетической компании. Оцени правовую ситуацию с точки зрения слабых и сильных мест в позиции оппонента и компании, неопределенности каких-либо фактических обстоятельств, недостатков доказательств. Напиши пошагово стратегию процессуального поведения, например проведения экспертиз, истребования доказательств, смены акцентов в обосновании позиции энергетической компании. Предложи, на чем необходимо сосредоточиться: на сборе доказательств, повышении качества представления интересов, дополнения позиции аргументами или сменой позиции.""",

    "prediction": """Ты - юрист-литигатор в энергетической компании. Оцени правовую ситуацию с точки зрения слабых и сильных мест в позиции энергетической компании, неопределенности каких-либо фактических обстоятельств, недостатков доказательств. Предположи три варианта ответных действий второй стороны после получения процессуального документа энергетической компании, например: [оппонент может пытаться доказывать…], [оппонент может усилить аргументацию в части…], [оппонент попытается опровергать…], с описанием существа действий. Предположи, каков может быть исход дела при неблагоприятном для энергетической компании развитии ситуации."""
}

def merge_json_parts(base_filename: str) -> dict:
    """
    Объединяет части JSON файлов в один словарь в памяти в порядке номеров частей.
    Возвращает объединенные данные или None в случае ошибки.
    """
    try:
        # Извлекаем базовое имя без номера части
        base_name = re.sub(r'_part\d+', '', base_filename)
        base_name = re.sub(r'\.json$', '', base_name)
        
        # Находим все соответствующие файлы
        pattern = os.path.join(DATA_DIR, f"{base_name}_part*.json")
        part_files = glob.glob(pattern)
        
        if not part_files:
            print(f"Не найдены файлы по шаблону: {pattern}")
            return None
        
        # Сортируем файлы по номеру части
        def get_part_number(filename):
            match = re.search(r'_part(\d+)\.json$', filename)
            return int(match.group(1)) if match else 0
            
        part_files = sorted(part_files, key=get_part_number)
        print(f"Найдены файлы для объединения (в порядке): {part_files}")
        
        merged_data = {'metadata': [], 'processed_files': []}
        success_count = 0
        
        for part_file in part_files:
            try:
                print(f"Обработка файла: {part_file}")
                part_data = safe_read_json(part_file)
                
                if not part_data:
                    print(f"Файл {part_file} не содержит данных или не может быть прочитан")
                    continue
                    
                # Объединяем метаданные
                if 'metadata' in part_data and isinstance(part_data['metadata'], list):
                    merged_data['metadata'].extend(part_data['metadata'])
                    success_count += 1
                    
                # Объединяем processed_files
                if 'processed_files' in part_data and isinstance(part_data['processed_files'], list):
                    merged_data['processed_files'].extend(part_data['processed_files'])
                
            except Exception as e:
                print(f"Ошибка при обработке файла {part_file}: {str(e)}")
                continue
        
        if not merged_data['metadata']:
            print("Нет данных metadata для объединения")
            return None
            
        # Удаляем дубликаты
        merged_data['processed_files'] = list(set(merged_data['processed_files']))
        print(f"Успешно объединено {success_count}/{len(part_files)} файлов")
        
        return merged_data
        
    except Exception as e:
        print(f"Критическая ошибка при объединении JSON частей: {str(e)}")
        return None

def safe_read_json(file_path: str) -> dict:
    """Безопасное чтение JSON с восстановлением и обработкой ошибок"""
    try:
        # Создаем резервную копию перед любыми изменениями
        backup_path = str(Path(file_path).with_suffix('.bak'))
        shutil.copy2(file_path, backup_path)
        
        # Чтение файла с обработкой BOM (Byte Order Mark)
        with open(file_path, 'rb') as f:
            content_bytes = f.read()
        
        # Попробуем декодировать как utf-8-sig (автоматически убирает BOM)
        try:
            content = content_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            # Если не utf-8, пробуем другие кодировки
            try:
                content = content_bytes.decode('cp1251')
            except UnicodeDecodeError:
                content = content_bytes.decode('latin-1')
        
        # Удаляем нулевые байты и другие непечатаемые символы
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        
        # Удаляем BOM, если остался
        if content.startswith('\ufeff'):
            content = content[1:]
        
        # Пытаемся найти начало и конец JSON
        start = content.find('{')
        end = content.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("Не найдены JSON-скобки")
        
        json_content = content[start:end]
        
        # Удаляем завершающие запятые перед } или ]
        json_content = re.sub(r',\s*([}\]])', r'\1', json_content)
        
        # Пытаемся загрузить JSON
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            # Если не получилось, пробуем восстановить
            try:
                # Удаляем все, что после последней закрывающей скобки
                last_brace = json_content.rfind('}')
                if last_brace != -1:
                    json_content = json_content[:last_brace+1]
                
                # Удаляем все, что перед первой открывающей скобкой
                first_brace = json_content.find('{')
                if first_brace != -1:
                    json_content = json_content[first_brace:]
                
                return json.loads(json_content)
            except json.JSONDecodeError:
                # Если все еще ошибка, пробуем прочитать резервную копию
                if os.path.exists(backup_path):
                    with open(backup_path, 'rb') as f:
                        backup_content = f.read().decode('utf-8-sig')
                        return json.loads(backup_content)
                raise
                
    except Exception as e:
        print(f"Критическая ошибка при чтении файла {file_path}: {str(e)}")
        return None

class TextPreprocessor:
    def __init__(self):
        self.regex = re.compile(r'[^\w\s]')

    def preprocess(self, text: str) -> List[str]:
        text = self.regex.sub(' ', text.lower())
        return text.split()

class BM25SearchEngine:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.bm25 = None
        self.chunks_info = []
        self.is_index_loaded = False
        self.llm_keywords = []
        self.data_dir = "data"  # Папка с файлами индекса
        self._load_index()

    def _find_part_files(self):
        """Находит все файлы индекса с part в названии в директории data"""
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir, exist_ok=True)
                return []

            part_files = []
            for filename in os.listdir(self.data_dir):
                if "part" in filename.lower() and filename.endswith(".json"):
                    full_path = os.path.join(self.data_dir, filename)
                    part_files.append(full_path)

            # Сортируем файлы по номеру части
            def extract_part_number(f):
                match = re.search(r'_part(\d+)\.json$', f, re.IGNORECASE)
                return int(match.group(1)) if match else 0

            return sorted(part_files, key=extract_part_number)
        except Exception as e:
            print(f"Ошибка поиска файлов индекса: {str(e)}")
            return []

    def _normalize_processed(self, processed_data):
        """Нормализует processed данные для индексации"""
        if processed_data is None:
            return []
            
        if isinstance(processed_data, str):
            try:
                # Пробуем распарсить строку как JSON
                processed_data = json.loads(processed_data)
            except json.JSONDecodeError:
                # Если не JSON, обрабатываем как текст
                return self.preprocessor.preprocess(processed_data)
        
        if isinstance(processed_data, list):
            # Удаляем пустые элементы и преобразуем к строке
            return [str(item).strip() for item in processed_data if item and str(item).strip()]
        
        # Для других типов преобразуем в строку и обрабатываем
        return self.preprocessor.preprocess(str(processed_data))

    def _read_json_with_recovery(self, file_path):
        """Чтение JSON с восстановлением при ошибках"""
        try:
            # Чтение с обработкой BOM
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8-sig')

            # Удаление непечатаемых символов
            content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
            
            # Попытка стандартного чтения
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Попробуем восстановить структуру
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > 0:
                    return json.loads(content[start:end])
                raise
        except Exception as e:
            print(f"Ошибка чтения файла {file_path}: {str(e)}")
            raise

    def _load_index(self):
        """Загрузка и построение индекса из частей"""
        try:
            part_files = self._find_part_files()
            if not part_files:
                print("Не найдены файлы индекса для загрузки")
                return False

            merged_data = {'metadata': [], 'processed_files': set()}
            
            for file_path in part_files:
                try:
                    file_data = self._read_json_with_recovery(file_path)
                    if not file_data:
                        continue

                    # Обработка metadata
                    if 'metadata' in file_data and isinstance(file_data['metadata'], list):
                        for item in file_data['metadata']:
                            if isinstance(item, dict) and 'original' in item:
                                # Нормализуем processed данные
                                item['processed'] = self._normalize_processed(item.get('processed', ''))
                                merged_data['metadata'].append(item)

                    # Обработка processed_files
                    if 'processed_files' in file_data and isinstance(file_data['processed_files'], list):
                        merged_data['processed_files'].update(file_data['processed_files'])

                except Exception as e:
                    print(f"Ошибка обработки файла {file_path}: {str(e)}")
                    continue

            if not merged_data['metadata']:
                print("Нет данных для построения индекса")
                return False

            # Построение индекса BM25
            corpus = []
            valid_metadata = []
            
            for item in merged_data['metadata']:
                processed = item.get('processed', [])
                if processed:
                    corpus.append(processed)
                    valid_metadata.append(item)

            if not corpus:
                print("Нет данных для индексации после нормализации")
                return False

            #self.bm25 = BM25Okapi(corpus)
            self.bm25 = BM25Okapi(corpus, k1=1.2, b=0.6)  # в методе _load_index
            self.chunks_info = valid_metadata
            self.is_index_loaded = True
            print(f"Индекс успешно загружен. Фрагментов: {len(corpus)}")
            return True

        except Exception as e:
            print(f"Критическая ошибка загрузки индекса: {str(e)}")
            return False

    def search(self, query, top_n=5, min_score=0.2):  # было 0.1
        """Поиск по индексу"""
        if not self.is_index_loaded or not query:
            return []

        # Улучшаем запрос сгенерированными ключевыми словами
        enhanced_query = f"{query} {' '.join(self.llm_keywords * 2)}" if self.llm_keywords else query
        tokens = self.preprocessor.preprocess(enhanced_query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        best_indices = [idx for idx in np.argsort(scores)[-top_n:][::-1] 
                       if scores[idx] >= min_score and idx < len(self.chunks_info)]

        return [{
            'doc_id': self.chunks_info[idx].get('file_id', ''),
            'doc_name': self.chunks_info[idx].get('doc_name', 'Документ'),
            'chunk_text': self.chunks_info[idx].get('original', '')[:2000],
            'score': round(float(scores[idx]), 4)
        } for idx in best_indices]
class LLMClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.is_initialized = True

    def query(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        try:
            payload = {
                "model": "google/gemini-2.0-flash-lite-001",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", error_detail)
                except:
                    pass
                
                raise Exception(f"API Error {response.status_code}: {error_detail}")

            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка соединения: {str(e)}")
        except Exception as e:
            raise Exception(f"Ошибка обработки ответа: {str(e)}")

class DocumentAnalyzer:
    def __init__(self, api_url: str = None, api_key: str = None):
        self.preprocessor = TextPreprocessor()
        self.search_engine = BM25SearchEngine(self.preprocessor)
        self.current_docx = None
        self.llm_client = None
        self.llm_initialized = False
        self._initialize_llm(api_url, api_key)

    def _initialize_llm(self, api_url: str, api_key: str) -> None:
        """Инициализирует клиент LLM"""
        if not api_url or not api_key:
            self.llm_initialized = False
            return
            
        try:
            self.llm_client = LLMClient(api_url, api_key)
            self.llm_initialized = True
        except Exception as e:
            print(f"Ошибка инициализации LLM: {e}")
            self.llm_initialized = False

    def _generate_keywords_from_text(self, text: str) -> List[str]:
        """Генерирует ключевые слова из текста с помощью LLM"""
        if not self.llm_initialized or not text:
            return []
            
        try:
            messages = [
                {"role": "system", "content": "Ты - эксперт по юридической терминологии. Определи существо спора, подготовь но саммари с цитированием правовых актов и составь список из 10 ключевых слов (один синоним на каждое слово) и десяти синонимичных по смыслу коротких выражений по этому саммари."},
                {"role": "user", "content": f"{KEYWORDS_PROMPT}\n\nТекст для анализа:\n{text[:10000]}"}
            ]
            
            response = self.llm_client.query(messages, TEMPERATURE, MAX_ANSWER_LENGTH)
            
            # Обработка ответа и извлечение ключевых слов
            keywords = []
            for line in response.split('\n'):
                if '→' in line:  # Обрабатываем строки с синонимами
                    parts = line.split('→')
                    for part in parts:
                        keywords.extend(re.findall(r'[\w\-]+', part.strip()))
                else:
                    keywords.extend(re.findall(r'[\w\-]+', line.strip()))
            
            # Удаляем дубликаты и пустые значения
            keywords = list(set(k.lower() for k in keywords if k.strip()))
            return keywords
            
        except Exception as e:
            print(f"Ошибка генерации ключевых слов: {e}")
            return []

    def analyze_document(self, prompt_type: str) -> str:
        """Анализирует документ с использованием LLM"""
        if not self.current_docx:
            return "Пожалуйста, загрузите DOCX файл"
            
        docx_text = self.current_docx["content"]
        
        # Генерация ключевых слов из текста документа
        if not self.search_engine.llm_keywords:
            with st.spinner("Генерация ключевых слов..."):
                self.search_engine.llm_keywords = self._generate_keywords_from_text(docx_text)
                st.sidebar.info(f"Сгенерированные ключевые слова: {', '.join(self.search_engine.llm_keywords)}")
        
        query = self._generate_search_query(prompt_type, docx_text)
        chunks = self.search_engine.search(query)
        context = self._build_context(docx_text, chunks)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": BUTTON_PROMPTS[prompt_type] + f"\n\nВес контента: {st.session_state.doc_weight_slider:.1f}\n\nКОНТЕКСТ:\n" + context}
        ]

        st.sidebar.header("Итоговый запрос к LLM")
        st.sidebar.markdown("### Запрос пользователя:")
        st.sidebar.markdown(BUTTON_PROMPTS[prompt_type] + f"\n\nВес контента: {st.session_state.doc_weight_slider:.1f}\n\nКОНТЕКСТ:\n" + context)
        
        return self.llm_client.query(messages, TEMPERATURE, MAX_ANSWER_LENGTH)

    def load_documents(self, uploaded_files) -> None:
        """Загружает DOCX файлы"""
        if not uploaded_files:
            return

        try:
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.size == 0:
                        st.warning(f"Файл {uploaded_file.name} пуст")
                        continue

                    file_bytes = io.BytesIO(uploaded_file.read())
                    doc = Document(file_bytes)
                    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

                    if not text:
                        st.warning(f"Файл {uploaded_file.name} не содержит текста")
                        continue

                    if len(text) > MAX_CONTEXT_LENGTH:
                        text = text[:MAX_CONTEXT_LENGTH]
                        st.warning(f"Документ {uploaded_file.name} обрезан до {MAX_CONTEXT_LENGTH} символов")

                    self.current_docx = {
                        "name": uploaded_file.name,
                        "content": text
                    }

                    st.success(f"Документ {uploaded_file.name} загружен для анализа")

                except Exception as e:
                    st.error(f"Ошибка обработки файла {uploaded_file.name}: {str(e)}")
                    continue

        except Exception as e:
            st.error(f"Общая ошибка при загрузке документов: {str(e)}")

    def _generate_search_query(self, prompt_type: str, docx_text: str) -> str:
        """Генерирует поисковый запрос для BM25"""
        base_queries = {
            "quality": "оценка качества документа структура аргументации доказательства нормы права",
            "strategy": "стратегия спора доказательства процессуальное поведение",
            "prediction": "позиция второй стороны прогнозирование аргументы оппонента"
        }
        
        return f"{base_queries[prompt_type]} {docx_text[:10000]}"

    def _build_context(self, docx_text: str, chunks: List[Dict]) -> str:
        """Строит контекст для LLM из DOCX и найденных фрагментов"""
        context_parts = [
            "=== ЗАГРУЖЕННЫЙ ДОКУМЕНТ ===",
            docx_text,
            "\n=== РЕЛЕВАНТНЫЕ ФРАГМЕНТЫ ИЗ БАЗЫ ЗНАНИЙ ==="
        ]
        
        for chunk in chunks:
            context_parts.append(f"\n📄 {chunk['doc_name']} (релевантность: {chunk['score']:.3f}):")
            context_parts.append(chunk['chunk_text'][:3000])
        
        return "\n".join(context_parts)

def main():
    st.set_page_config(page_title="El Documente", layout="wide", initial_sidebar_state="collapsed")
    gif_path = "data/maracas-sombrero-hat.gif"
    st.image(gif_path, caption="Hola!", width=64)
    st.title("El Documente: проверьте свой процессуальный документ")
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = DocumentAnalyzer(API_URL, API_KEY)
    
    analyzer = st.session_state.analyzer
    
    st.sidebar.header("Настройки поиска")
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        weight = st.slider(
            "Вес контента документа",
            0.1, 1.0, 0.7, 0.1,
            key="doc_weight_slider",
            help="Регулирует влияние текста документа на результаты поиска"
        )

    with col2:
        st.metric("Текущее значение", f"{weight:.1f}")

    st.sidebar.write(f"Выбрано значение: {weight}")
    
    if not analyzer.llm_initialized:
        st.sidebar.error("LLM не инициализирован. Проверьте API ключ и URL")
    
    st.header("Загрузка документа")
    uploaded_files = st.file_uploader(
        "Выберите документ в формате DOCX", 
        type=["docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Обработка документов..."):
            analyzer.load_documents(uploaded_files)
        st.success(f"Загружено документов: {len(uploaded_files)}")

    CHAT_TEMPERATURE = 0.6
    CHAT_SYSTEM_PROMPT = """Ты - опытный дружелюбный юрист энергетической компании, отвечающий на правовые вопросы. ЗАПРЕЩЕНО:  1. ссылаться на выдуманные законы и судебную практику. 2. указывать в ответе, что ты ознакомился с документ, просто поддерживай диалог. Ответы излагай в деловом стиле, без категорических мнений."""

    st.header("Чат")

    user_input = st.text_area(
        "Обсудить с наставником Карлосом",
        max_chars=500,
        height=100
    )

    ask_button = st.button("Спросить", disabled=not (uploaded_files))# and user_input))

    if 'docx_added' not in st.session_state:
        st.session_state.docx_added = False

    if ask_button:
        # 1. Готовим контекст
        doc_summary = analyzer.current_docx["content"][:3000]  # или используйте суммаризацию
        relevant_chunks = analyzer.search_engine.search(user_input)
    
        # 2. Формируем сообщения с разделением ролей
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "assistant", "content": f"Анализируемый документ (сокращённо):\n{doc_summary}"},
            *[
                {"role": "assistant", "content": f"Релевантный фрагмент ({chunk['doc_name']}):\n{chunk['chunk_text'][:800]}"}
                for chunk in relevant_chunks[:2]  # Только топ-2 фрагмента
            ],
            {"role": "user", "content": f"Диалог:\n{'\n'.join(st.session_state.get('chat_history', []))[-2:]}"},
            {"role": "user", "content": user_input}
        ]
    
        # 3. Отправка запроса
        response = analyzer.llm_client.query(messages, temperature=0.7, max_tokens=1500)
        response_container = st.empty()
        response_container.markdown("### Ответ от Карлоса")
        response_container.markdown(response)
        # 4. Сохраняем историю
        st.session_state.setdefault('chat_history', []).extend([user_input, response])


    st.header("Анализ документа")
    col1, col2, col3 = st.columns(3)

    buttons_disabled = not (uploaded_files and analyzer.llm_initialized)
    
    with col1:
        if st.button("Оценить качество документа", disabled=buttons_disabled):
            with st.spinner("Анализ документа..."):
                result = analyzer.analyze_document("quality")
                st.markdown("### Результат оценки")
                st.markdown(result)
    
    with col2:
        if st.button("Дать рекомендации по стратегии спора", disabled=buttons_disabled):
            with st.spinner("Формирование рекомендаций..."):
                result = analyzer.analyze_document("strategy")
                st.markdown("### Рекомендации по стратегии")
                st.markdown(result)
    
    with col3:
        if st.button("Спрогнозировать позицию второй стороны", disabled=buttons_disabled):
            with st.spinner("Прогнозирование позиции..."):
                result = analyzer.analyze_document("prediction")
                st.markdown("### Прогноз позиции оппонента")
                st.markdown(result)

if __name__ == "__main__":
    main()
