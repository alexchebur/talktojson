import requests
import random
import re
import chardet
import logging
from bs4 import BeautifulSoup
from typing import List, Dict
from config import USER_AGENTS, PRIORITY_SITES, API_TIMEOUT

logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self, delay_range=(1.0, 3.0)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        
        # Настройки Google CSE
        self.api_key = "AIzaSyCNVeNmUgrt-kL5ZI4EkHFoTjTzRSWATX4"
        self.cse_id = "a4f17489c6a0a4414"
        self.priority_sites = PRIORITY_SITES  # Используем из конфига
        
    def perform_search(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            priority_results = []
            
            # ШАГ 1: Поиск по приоритетным сайтам
            for site in self.priority_sites:
                if len(priority_results) >= max_results:
                    break
                    
                site_query = f"site:{site} {query}"
                results = self._execute_search(site_query, max_results - len(priority_results))
                priority_results.extend(results)

            # ШАГ 2: Общий поиск если не набрано достаточно результатов
            final_results = priority_results
            if len(final_results) < max_results:
                general_results = self._execute_search(
                    query, 
                    max_results - len(final_results))
                final_results.extend(general_results)

            return final_results[:max_results]

        except Exception as e:
            logger.error(f"Ошибка Google CSE: {str(e)}")
            return []

    def _execute_search(self, query: str, max_results: int) -> List[Dict]:
        """Внутренний метод для выполнения запроса к API"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'num': min(max_results, 3),  # Ограничение API
            'lr': 'lang_ru',
            'hl': 'ru'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('items', [])[:max_results]:
                full_content = self.get_full_page_content(item.get('link', ''))
                results.append({
                    'title': item.get('title', 'Без названия')[:150],
                    'url': item.get('link', '#'),
                    'snippet': item.get('snippet', 'Без описания')[:500],
                    'full_content': full_content
                })

            print(f"Searching: {url} with params: {params}")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка выполнения поиска: {str(e)}")
            return []

    @staticmethod
    def get_full_page_content(url: str) -> str:
        """Получение полного текста страницы"""
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()

            # Определение кодировки
            if response.encoding == 'ISO-8859-1':
                raw_data = response.content[:50000]
                encoding = chardet.detect(raw_data)['encoding']
                response.encoding = encoding if encoding else 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')

            # Удаляем ненужные элементы
            for tag in soup(['script', 'style', 'footer', 'nav', 'aside', 'header', 'iframe', 'form', 'button']):
                tag.decompose()

            # Извлекаем контент
            text_parts = []
            for tag in soup.find_all(['main', 'article', 'section', 'div', 'p']):
                text = tag.get_text(' ', strip=True)
                if len(text) > 100:
                    text_parts.append(text)

            full_text = ' '.join(text_parts)
            return re.sub(r'\s+', ' ', full_text)[:30000] if full_text else ""
            
        except Exception as e:
            logger.error(f"Ошибка получения контента: {str(e)}")
            return ""
