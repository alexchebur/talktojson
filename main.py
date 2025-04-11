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

try:
    from config import API_KEY, API_URL
except ImportError:
    st.error("Ошибка: Создайте файл config.py с переменными API_KEY и API_URL")
    API_KEY = ""
    API_URL = "https://api.vsegpt.ru/v1/chat/completions"

# Конфигурация
DATA_DIR = "data"
MAX_CONTEXT_LENGTH = 15000
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
    Объединяет части JSON файлов в один словарь в памяти.
    Возвращает объединенные данные или None в случае ошибки.
    """
    try:
        # Находим все файлы с базовым именем
        pattern = re.sub(r'_part\d+\.json$', '_part*.json', base_filename)
        part_files = sorted(glob.glob(pattern))
        
        if not part_files:
            # Если нет разбитых файлов, пробуем загрузить как единый файл
            single_file = base_filename.replace('_part*.json', '.json')
            if os.path.exists(single_file):
                return safe_read_json(single_file)
            return None
        
        merged_data = {'metadata': [], 'processed_files': []}
        
        for part_file in part_files:
            part_data = safe_read_json(part_file)
            if not part_data:
                continue
                
            # Объединяем метаданные
            if 'metadata' in part_data and isinstance(part_data['metadata'], list):
                merged_data['metadata'].extend(part_data['metadata'])
                
            # Объединяем processed_files
            if 'processed_files' in part_data and isinstance(part_data['processed_files'], list):
                merged_data['processed_files'].extend(part_data['processed_files'])
        
        # Удаляем дубликаты
        merged_data['processed_files'] = list(set(merged_data['processed_files']))
        
        return merged_data
        
    except Exception as e:
        print(f"Ошибка при объединении JSON частей: {e}")
        return None

def safe_read_json(file_path: str) -> dict:
    """Безопасное чтение JSON с восстановлением"""
    try:
        # Создаем резервную копию
        backup_path = str(Path(file_path).with_suffix('.bak'))
        shutil.copy2(file_path, backup_path)
        
        # Чтение с обработкой BOM
        with open(file_path, 'rb') as f:
            content = f.read().decode('utf-8-sig')
        
        # Удаление непечатаемых символов
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        content = content.replace('\\u0002', '')
        # Извлечение JSON части
        start = content.find('{')
        end = content.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("Не найдены JSON-скобки")
            
        return json.loads(content[start:end])
        
    except Exception as e:
        st.sidebar.error(f"Ошибка чтения JSON:")
        # Пробуем прочитать резервную копию
        if os.path.exists(backup_path):
            try:
                with open(backup_path, 'rb') as f:
                    return json.loads(f.read().decode('utf-8-sig'))
            except Exception:
                pass
        raise

class TextPreprocessor:
    def __init__(self):
        self.regex = re.compile(r'[^\w\s]')

    def preprocess(self, text: str) -> List[str]:
        text = self.regex.sub(' ', text.lower())
        return text.split()

class BM25SearchEngine:
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.bm25 = None
        self.chunks_info = []
        self.is_index_loaded = False
        self.cache_path = os.path.join("data", "bm25_index.json")
        self._load_index()
        self.llm_keywords = []  # Добавляем список для хранения ключевых слов от LLM
        
    def _normalize_processed(self, processed_data: Any) -> List[str]:
        """Нормализует поле processed в единый формат списка токенов"""
        if processed_data is None:
            return []
    
        if isinstance(processed_data, str):
            return [token for token in processed_data.split() if token]
        elif isinstance(processed_data, list):
            result = []
            for item in processed_data:
                if isinstance(item, str):
                    result.extend(item.split())
                elif isinstance(item, (int, float)):
                    result.append(str(item))
            return result
        else:
            return [str(processed_data)]

    def _load_index(self) -> bool:
        """Загрузка индекса с проверкой"""
        # Сначала пробуем загрузить объединенные данные из частей
        merged_data = merge_json_parts(self.cache_path.replace('.json', '_part*.json'))
        
        if not merged_data:
            # Если нет разбитых файлов, пробуем загрузить как единый файл
            if not os.path.exists(self.cache_path):
                st.sidebar.warning(f"Файл индекса не найден: {self.cache_path}")
                return False

            try:
                merged_data = safe_read_json(self.cache_path)
            except Exception as e:
                st.sidebar.error(f"Ошибка загрузки индекса: {str(e)}")
                return False
        
        if not isinstance(merged_data, dict):
            st.sidebar.error("Индекс должен быть словарем")
            return False
            
        if 'metadata' not in merged_data:
            st.sidebar.error("Отсутствует ключ 'metadata' в индексе")
            return False
        
        processed_texts = []
        valid_metadata = []
    
        for i, item in enumerate(merged_data.get('metadata', [])):
            if not isinstance(item, dict):
                st.sidebar.warning(f"Пропущен элемент {i} - не является словарем")
                continue
            
            original_text = item.get('original', '')
            processed = self._normalize_processed(item.get('processed', []))
        
            if not processed and original_text:
                processed = self.preprocessor.preprocess(original_text)
            
            if processed:
                processed_texts.append(processed)
                valid_metadata.append(item)
    
        if not processed_texts:
            st.sidebar.error("Индекс не содержит валидных документов")
            return False
    
        self.bm25 = BM25Okapi(processed_texts)
        self.chunks_info = valid_metadata
        self.is_index_loaded = True
    
        st.sidebar.success(f"Загружен индекс с {len(processed_texts)} фрагментами данных")
        return True

    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        """Поиск с обработкой ошибок"""
        if not self.is_index_loaded:
            return []

        try:
            # Добавляем ключевые слова от LLM к запросу
            enhanced_query = f"{query} {' '.join(self.llm_keywords)}"
            tokens = self.preprocessor.preprocess(enhanced_query)
            
            if not tokens:
                return []

            if not hasattr(self.bm25, 'doc_freqs') or len(self.bm25.doc_freqs) == 0:
                return []

            if len(self.bm25.doc_len) == 0:
                return []

            scores = self.bm25.get_scores(tokens)
            if scores is None or len(scores) == 0:
                return []

            best_indices = np.argsort(scores)[-top_n:][::-1]

            return [
                {
                    'doc_id': self.chunks_info[idx].get('file_id', ''),
                    'doc_name': self.chunks_info[idx].get('doc_name', 'Документ'),
                    'chunk_text': self.chunks_info[idx].get('original', '')[:2000],
                    'score': round(float(scores[idx]), 4)
                }
                for idx in best_indices
                if idx < len(self.chunks_info)
            ]
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []            

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
                {"role": "system", "content": "Ты - эксперт по юридической терминологии. Извлекай ключевые слова и выражения из текста."},
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
            context_parts.append(f"\n📄 {chunk['doc_name']} (релевантность: {chunk['score']:.2f}):")
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
