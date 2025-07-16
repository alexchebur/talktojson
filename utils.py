import os
import chardet
import re
import logging
from typing import Optional
from docx import Document
from PyPDF2 import PdfReader
import streamlit as st

def detect_file_encoding(file_path: str) -> str:
    """Определение кодировки файла"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
    return chardet.detect(raw_data)['encoding']

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
            text = "\n".join([page.extract_text() for page in reader.pages])
            return text if text.strip() else None
        
    except Exception as e:
        logging.error(f"Ошибка обработки файла: {str(e)}")
        return None

def clean_keyword(word: str) -> str:
    """Очистка ключевых слов"""
    while len(word) > 0 and word[-1] in 'аеёийоуыэюя':
        word = word[:-1]
    return word

def initialize_session():
    """Инициализация сессии Streamlit"""
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
        "web_search_results": [],
        "web_search_chunks": [],
        "generated_queries": [],
        "additional_chunks": []
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Очистка текста от лишних пробелов и спецсимволов"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
