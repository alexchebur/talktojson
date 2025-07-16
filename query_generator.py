import requests
import re
from typing import List
from config import GEMINI_API_KEY, API_URL, API_TIMEOUT
from prompts import get_prompt
import logging

logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self):
        self.api_url = API_URL
        self.api_key = GEMINI_API_KEY
        
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

    
    def expand_query(self, initial_query: str, context: str) -> List[str]:
        """Расширенная генерация запросов с учетом контекста (для будущего развития)"""
        # Можно добавить многоэтапную генерацию
        pass
