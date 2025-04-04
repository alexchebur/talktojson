import streamlit as st
import json
import os
from pathlib import Path
import uuid
import re
from PyPDF2 import PdfReader
import docx
import requests
import numpy as np
import time
from rank_bm25 import BM25Okapi
from config import API_KEY, API_URL

# Constants
LLM = "anthropic/claude-3-haiku"
CHUNK_SIZE = 20000
CONTEXT_SUM = 4000
MAX_ANSWER_LENGTH = 4000
TEMPERATURE = 0.4

# Base prompts
DEFAULT_PROMPT = """Извлеки данные из юридического/публицистического документа. Формат ответа строго соблюдай:
doc_name: [полное название документа]
doc_date: [дата в формате ГГГГ-ММ-ДД или "Не указана"]
doc_type: [тип: закон, указ, постановление, судебный акт, статья или иное]
chunk_summary: [три тезиса о содержании]
qa_pairs: [3 пары в формате "вопрос:: ответ" (разделитель - два двоеточия)]
chunk_keywords: [3 ключевых слова, 3 ключевых фразы]"""

class DocumentProcessor:
    def __init__(self):
        self.knowledge_base = {}

    def read_file(self, file):
        try:
            if file.name.endswith('.pdf'):
                reader = PdfReader(file)
                return "\n".join([page.extract_text() for page in reader.pages])
            elif file.name.endswith('.docx'):
                doc = docx.Document(file)
                return "\n".join([para.text for para in doc.paragraphs])
            else:
                return file.getvalue().decode('utf-8')
        except Exception as e:
            st.error(f"Ошибка чтения файла {file.name}: {str(e)}")
            return ""

    def process_document(self, text, prompt_template):
        chunks = self.split_text(text)
        doc_id = str(uuid.uuid4())
        doc_data = {
            "doc_id": doc_id,
            "chunks": [],
            "doc_summary": "",
            "doc_keywords": [],
        }

        for i, chunk in enumerate(chunks):
            response = self.send_llm_request(
                prompt_template + f"\n\nТекст: {chunk[:5000]}..."
            )
            chunk_data = self.parse_llm_response(response, i == 0)
            chunk_data['chunk_text'] = chunk

            if i == 0:
                doc_data.update({
                    "doc_name": chunk_data.get('doc_name', 'Неизвестно'),
                    "doc_date": chunk_data.get('doc_date', 'Не указана'),
                    "doc_type": chunk_data.get('doc_type', 'Неизвестен')
                })

            doc_data['chunks'].append(chunk_data)

        return doc_data

    def split_text(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > CHUNK_SIZE and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)

        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def send_llm_request(self, prompt):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": LLM,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 2000
        }

        max_retries = 3
        base_delay = 2  # базовая задержка в секундах

        # Принудительная задержка в 1 секунду между запросами
        time.sleep(1)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Экспоненциальная задержка с добавлением случайности
                    delay = base_delay * (2 ** attempt) + np.random.uniform(0, 1)
                    time.sleep(delay)

                # Увеличиваем timeout и добавляем retries для requests
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(max_retries=3)
                session.mount('https://', adapter)
                
                response = session.post(API_URL, json=data, headers=headers, timeout=60)
                response.raise_for_status()
                response_data = response.json()

                if 'choices' in response_data and len(response_data['choices']) > 0:
                    return response_data['choices'][0]['message']['content']

                raise ValueError("Неверный формат ответа от API")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    continue
                st.error(f"Ошибка API: {str(e)}\nСтатус: {e.response.status_code}\nОтвет: {e.response.text}")
            except Exception as e:
                st.error(f"Ошибка API: {str(e)}")

            if attempt == max_retries - 1:
                return ""

    def parse_llm_response(self, response, is_first_chunk=False):
        parsed = {
            'chunk_summary': '',
            'chunk_text': '',
            'qa_pairs': [],
            'chunk_keywords': []
        }

        if is_first_chunk:
            parsed.update({
                'doc_name': 'Неизвестно',
                'doc_date': 'Не указана',
                'doc_type': 'Неизвестен'
            })

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if is_first_chunk:
                if line.startswith('doc_name:'):
                    parsed['doc_name'] = line.split(':', 1)[1].strip()
                elif line.startswith('doc_date:'):
                    parsed['doc_date'] = line.split(':', 1)[1].strip()
                elif line.startswith('doc_type:'):
                    parsed['doc_type'] = line.split(':', 1)[1].strip()

            if line.startswith('chunk_summary:'):
                current_section = 'summary'
                parsed['chunk_summary'] = line.split(':', 1)[1].strip()
            elif line.startswith('qa_pairs:'):
                current_section = 'qa'
            elif line.startswith('chunk_keywords:'):
                current_section = 'keywords'
            elif '::' in line and current_section == 'qa':
                q, a = line.split('::', 1)
                parsed['qa_pairs'].append({'question': q.strip(), 'answer': a.strip()})
            elif current_section == 'keywords':
                parsed['chunk_keywords'].append(line.strip())

        return parsed

class SearchEngine:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.bm25 = None
        self.chunks_info = []

    def build_index(self, knowledge_base):
        corpus = []
        self.chunks_info = []

        for doc in knowledge_base.get('documents', []):
            for chunk in doc.get('chunks', []):
                text_parts = [
                    str(chunk.get('chunk_summary', '')),
                    ' '.join(map(str, chunk.get('chunk_keywords', []))),
                    str(chunk.get('chunk_text', ''))
                ]
                text_for_index = ' '.join(text_parts)
                corpus.append(text_for_index)

                self.chunks_info.append({
                    'doc_name': doc.get('doc_name', ''),
                    'chunk_summary': chunk.get('chunk_summary', ''),
                    'chunk_text': chunk.get('chunk_text', ''),
                    'chunk_keywords': chunk.get('chunk_keywords', [])
                })

        tokenized_corpus = [self.preprocessor.preprocess(doc) for doc in corpus]
        if not tokenized_corpus:
            st.warning("База знаний пуста. Пожалуйста, загрузите и обработайте документы.")
            return
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query, top_n=5):
        if not self.bm25 or not self.chunks_info:
            self.build_index(st.session_state.knowledge_base)
            if not self.bm25:
                return []

        tokens = self.preprocessor.preprocess(query)
        scores = self.bm25.get_scores(tokens)
        best_indices = np.argsort(scores)[-top_n:][::-1]

        results = []
        for idx in best_indices:
            if scores[idx] > 0.1:
                result = {**self.chunks_info[idx], 'score': float(scores[idx])}
                results.append(result)
        return results

class TextPreprocessor:
    def __init__(self):
        self.regex = re.compile(r'[^\w\s]')

    def preprocess(self, text):
        text = self.regex.sub(' ', text.lower())
        return text.split()

def main():
    st.set_page_config(page_title="Документ-Ассистент", layout="wide")
    st.title("Документ-Ассистент")

    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
        st.session_state.search_engine = SearchEngine()
        
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = {'documents': []}

    # Сайдбар для управления базой знаний (упрощённая версия)
    with st.sidebar:
        st.header("Управление базой знаний")
        
        # Загрузка существующей базы (если нужно)
        kb_file = st.file_uploader("Загрузить базу знаний", type=['json'])
        if kb_file:
            st.session_state.knowledge_base = json.load(kb_file)
            st.session_state.search_engine.build_index(st.session_state.knowledge_base)
            st.success("База знаний загружена")

        # Кнопка скачивания (всегда видна, но активна только при наличии данных)
        if st.session_state.knowledge_base.get('documents'):
            json_str = json.dumps(st.session_state.knowledge_base, ensure_ascii=False, indent=2)
            st.download_button(
                label="Скачать базу знаний",
                data=json_str,
                file_name="knowledge_base.json",
                mime="application/json",
                help="Скачать текущую базу знаний в формате JSON"
            )
        else:
            st.warning("База знаний пуста")
   
    # Основной интерфейс обработки документов (без изменений)
    tab1, tab2 = st.tabs(["Обработка документов", "Поиск информации"])

    with tab1:
        st.header("Обработка документов")
        uploaded_files = st.file_uploader(
            "Загрузите документы",
            type=['txt', 'pdf', 'docx'],
            accept_multiple_files=True
        )

        with st.spinner("Обновление поискового индекса..."):
            st.session_state.search_engine.build_index(st.session_state.knowledge_base)
        with st.expander("Настройка промпта"):
            prompt = st.text_area("Промпт для обработки", DEFAULT_PROMPT, height=300)

        if st.button("Обработать документы") and uploaded_files:
            total_files = len(uploaded_files)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Обработка файла {i+1}/{total_files}: {file.name}")
                text = st.session_state.processor.read_file(file)
                if text:
                    doc_data = st.session_state.processor.process_document(text, prompt)
                    if 'documents' not in st.session_state.knowledge_base:
                        st.session_state.knowledge_base['documents'] = []
                    st.session_state.knowledge_base['documents'].append(doc_data)

                progress_bar.progress((i + 1) / total_files)

            status_text.text("Обработка завершена!")
            st.session_state.search_engine.build_index(st.session_state.knowledge_base)
            st.success(f"Обработано файлов: {total_files}. Индекс поиска обновлён!")

        with tab2:
        st.header("Поиск информации")
        query = st.text_input("Введите запрос")

        with st.expander("Настройка промпта для LLM"):
            llm_prompt = st.text_area(
                "Промпт для LLM",
                "Ты - AI ассистент, анализирующий документы. Ссылайся на номер статей и пунктов.",
                height=100
            )

        if st.button("Искать") and query:
            # Проверяем, что база знаний не пуста
            if not st.session_state.knowledge_base.get('documents'):
                st.warning("База знаний пуста. Сначала обработайте документы.")
                return
            
            # Проверяем, что индекс построен
            if not hasattr(st.session_state.search_engine, 'bm25'):
                st.session_state.search_engine.build_index(st.session_state.knowledge_base)
            
            results = st.session_state.search_engine.search(query)
            if results:
                context = build_llm_context(query, results)
                response = st.session_state.processor.send_llm_request(
                    f"{llm_prompt}\n\n{context}"
                )
                st.write("Ответ:")
                st.write(response)
            
                # Показываем найденные документы для прозрачности
                with st.expander("Показать найденные фрагменты"):
                    for i, result in enumerate(results[:3]):  # Показываем первые 3 результата
                        st.markdown(f"**Документ {i+1}:** {result.get('doc_name', '')}")
                        st.caption(f"Рейтинг соответствия: {result.get('score', 0):.2f}")
                        st.write(result.get('chunk_summary', ''))
            else:
                st.warning("По запросу ничего не найдено")

def build_llm_context(query, chunks):
    context_parts = [
        f"Запрос пользователя: {query}",
        "Релевантные фрагменты из документов:"
    ]

    for chunk in chunks:
        context_parts.extend([
            f"\nДокумент: {chunk.get('doc_name', '')}",
            f"Ключевые слова: {', '.join(chunk.get('chunk_keywords', []))}",
            f"Содержание: {chunk.get('chunk_text', '')[:1000]}"
        ])

    return '\n'.join(context_parts)[:CONTEXT_SUM]

if __name__ == "__main__":
    main()
