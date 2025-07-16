import os
import chardet
import re
from docx import Document
from PyPDF2 import PdfReader
from typing import Optional
import streamlit as st

def detect_file_encoding(file_path: str) -> str:
    # [Реализация как в оригинале]

def file_to_text(uploaded_file) -> Optional[str]:
    # [Реализация как в оригинале]

def initialize_session():
    """Инициализация сессии Streamlit"""
    default_state = {
        "chat_log": "",
        # ... [все остальные переменные сессии]
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Очистка текста от лишних пробелов и спецсимволов"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
