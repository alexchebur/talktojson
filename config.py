
import os
GEMINI_API_KEY = "AIzaSyCGC2JB3BgfBMycbt4us1eq6D5exNOvKT8"
MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-06-17:generateContent"
#API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# Настройки обработки документов
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000
API_TIMEOUT = 60

# Настройки веб-поиска
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.6422.78 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"
]

PRIORITY_SITES = [
    "rosteplo.ru",
    "consultant.ru",
    "garant.ru",
    "zakon.ru",
    "zhane.ru"
]

