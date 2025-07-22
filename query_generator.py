import requests
import re
import logging
from typing import List
from prompts import get_prompt
from config import GEMINI_API_KEY, API_URL, API_TIMEOUT

logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self):
        self.api_url = API_URL
        self.api_key = GEMINI_API_KEY
        
    def generate(self, user_query: str, keywords: List[str]) -> List[str]:
        """Генерация уточняющих запросов с помощью LLM"""
        try:
            # Формирование промпта
            prompt = get_prompt("query_generation", {
                "user_query": user_query,
                "keywords": ", ".join(keywords)
            })  # <-- Закрывающая скобка добавлена
            
            # Подготовка данных для запроса
            request_data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 500
                }
            }
            
            # Выполнение запроса
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json=request_data,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Извлечение запросов из нумерованного списка
            queries = []
            content = ""
            
            if 'candidates' in response_data and response_data['candidates']:
                content = response_data['candidates'][0]['content']['parts'][0]['text']
            else:
                logger.error(f"Неожиданный ответ API: {response_data}")
                return []
                
            for line in content.split('\n'):
                if re.match(r'^\d+[\.\)]', line.strip()):
                    query = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                    if query:
                        queries.append(query)
                        
            return queries[:3]  # Ограничиваем 3 запросами
            
        except Exception as e:
            logger.error(f"Ошибка генерации запросов: {str(e)}")
            return []
    
    def expand_query(self, initial_query: str, context: str) -> List[str]:
        """Расширенная генерация запросов с учетом контекста"""
        # Формируем расширенный промпт
        expanded_prompt = f"""
        На основе исходного запроса и контекста сгенерируй 3 уточняющих запроса:
        
        **Исходный запрос**: {initial_query}
        
        **Контекст**: {context[:1000]}...
        
        **Требования**:
        - Каждый запрос должен быть самостоятельным
        - Используй профессиональную юридическую терминологию
        - Учитывай информацию из контекста
        - Формат: нумерованный список
        """
        
        try:
            # Подготовка данных для запроса
            request_data = {
                "contents": [
                    {
                        "parts": [
                            {"text": expanded_prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 300
                }
            }
            
            # Выполнение запроса
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                params={"key": self.api_key},
                json=request_data,
                timeout=API_TIMEOUT
            )
            response.raise_for_status()
            response_data = response.json()
            
            # Обработка ответа
            queries = []
            if 'candidates' in response_data and response_data['candidates']:
                content = response_data['candidates'][0]['content']['parts'][0]['text']
                for line in content.split('\n'):
                    if re.match(r'^\d+[\.\)]', line.strip()):
                        query = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                        if query:
                            queries.append(query)
            
            return queries[:3]
            
        except Exception as e:
            logger.error(f"Ошибка расширенной генерации: {str(e)}")
            return []
