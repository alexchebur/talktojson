import requests
import random
import re
import chardet
import logging
import time
from bs4 import BeautifulSoup
from typing import List, Dict
from config import USER_AGENTS, PRIORITY_SITES, API_TIMEOUT
from tenacity import retry, stop_after_attempt, wait_exponential

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
        
        # Кэширование для DuckDuckGo (5 минут как рекомендовано в документации)
        self.ddg_cache = {}
        self.cache_ttl = 300  # 5 минут в секундах
    
    def perform_search(self, query: str, max_results: int = 1, query_type="generated") -> List[Dict]:
        try:
            results = self._execute_search(query, max_results)
            for res in results:
                res['query'] = query  # Сохраняем запрос
                res['query_type'] = query_type  # Тип запроса
            return results
        except Exception as e:
            logger.error(f"Ошибка поиска: {str(e)}")
            return []

    def _execute_search(self, query: str, max_results: int) -> List[Dict]:
        """Выполняет поиск сначала через DuckDuckGo, при неудаче - через Google CSE"""
        cache_key = f"{query}:{max_results}"
        
        # Проверяем кэш для DuckDuckGo (рекомендовано в документации)
        if cache_key in self.ddg_cache:
            cached_time, results = self.ddg_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.info(f"Используем кэшированные результаты DuckDuckGo для запроса: {query}")
                return results
        
        # Сначала пробуем DuckDuckGo
        duckduckgo_results = self._duckduckgo_search(query, max_results)
        
        # Если получили результаты из DuckDuckGo, возвращаем их и кэшируем
        if duckduckgo_results:
            logger.info(f"Получено {len(duckduckgo_results)} результатов из DuckDuckGo")
            self.ddg_cache[cache_key] = (time.time(), duckduckgo_results)
            return duckduckgo_results
        
        # Иначе пробуем Google CSE
        logger.warning("DuckDuckGo не вернул результатов, переключаемся на Google CSE")
        google_results = self._google_cse_search(query, max_results)
        return google_results

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict]:
        """Поиск через DuckDuckGo API с экспоненциальной задержкой при ошибках"""
        # Добавляем небольшую случайную задержку перед запросом (рекомендовано в документации)
        time.sleep(random.uniform(1.5, 2.5))
        
        # Формируем URL для DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'pretty': '1',
            'no_html': '1',
            'skip_disambig': '1',
            't': 'myapp'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Обрабатываем основные результаты
            if 'Results' in data and data['Results']:
                for item in data['Results'][:max_results]:
                    if 'FirstURL' in item and item['FirstURL']:
                        full_content = self.get_full_page_content(item['FirstURL'])
                        custom_snippet = self._generate_snippet(full_content)
                        
                        results.append({
                            'title': item.get('Text', 'Без названия')[:150],
                            'url': item['FirstURL'],
                            'snippet': custom_snippet,
                            'full_content': full_content
                        })
            
            # Если недостаточно результатов, берем из RelatedTopics
            if len(results) < max_results and 'RelatedTopics' in data:
                for topic in data['RelatedTopics']:
                    if 'FirstURL' in topic and topic['FirstURL'] and len(results) < max_results:
                        full_content = self.get_full_page_content(topic['FirstURL'])
                        custom_snippet = self._generate_snippet(full_content)
                        
                        results.append({
                            'title': topic.get('Text', 'Без названия')[:150],
                            'url': topic['FirstURL'],
                            'snippet': custom_snippet,
                            'full_content': full_content
                        })
            
            # Применяем фильтрацию по приоритетным сайтам, если есть
            if self.priority_sites:
                filtered_results = []
                for site in self.priority_sites:
                    for res in results:
                        if site in res['url'] and len(filtered_results) < max_results:
                            filtered_results.append(res)
                if filtered_results:
                    return filtered_results[:max_results]
            
            return results[:max_results]
            
        except Exception as e:
            st.error(f"Ошибка при поиске через DuckDuckGo: {str(e)}")
            # Если это ошибка ограничения запросов, делаем дополнительную задержку
            if "limit" in str(e).lower():
                time.sleep(5)
            raise  # Для работы с retry

    def _google_cse_search(self, query: str, max_results: int) -> List[Dict]:
        """Поиск через Google Custom Search Engine"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.api_key,
            'cx': self.cse_id,
            'q': query,
            'num': min(max_results, 10),
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
                custom_snippet = self._generate_snippet(full_content)
            
                results.append({
                    'title': item.get('title', 'Без названия')[:150],
                    'url': item.get('link', '#'),
                    'snippet': custom_snippet,
                    'full_content': full_content
                })
            time.sleep(1.5)
            return results
        except Exception as e:
            logger.error(f"Ошибка выполнения поиска через Google CSE: {str(e)}")
            return []

    def _generate_snippet(self, full_content: str) -> str:
        """Генерация улучшенного сниппета из полного контента"""
        return full_content[:300] + "..." if len(full_content) > 300 else full_content

    @staticmethod
    def get_full_page_content(url: str) -> str:
        """Получение полного текста страницы с улучшенным парсингом (5000-10000 символов)"""
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
        
            # Определение кодировки
            if response.encoding == 'ISO-8859-1':
                raw_data = response.content[:10000]
                encoding = chardet.detect(raw_data)['encoding']
                response.encoding = encoding if encoding else 'utf-8'
        
            # Парсинг HTML
            soup = BeautifulSoup(response.text, 'html.parser')
        
            # Удаляем ненужные элементы
            for tag in soup(['script', 'style', 'footer', 'nav', 'aside', 'header', 'iframe', 'form', 'button', 'noscript', 'svg']):
                tag.decompose()
        
            # Удаляем пустые элементы
            for tag in soup.find_all():
                if not tag.get_text(strip=True):
                    tag.decompose()
        
            # Извлекаем основной контент с приоритетом на определенные теги
            main_content = []
        
            # Сначала ищем в основных контейнерах
            priority_containers = soup.find_all(['main', 'article', 'section', 'div[role="main"]', 'div.content', 'div.post'])
            if priority_containers:
                for container in priority_containers:
                    text = container.get_text(separator=' ', strip=True)
                    if len(text) > 200:  # Минимальная длина для значимого контента
                        main_content.append(text)
            else:
                # Если основных контейнеров нет, берем весь текст
                main_content.append(' '.join(soup.stripped_strings))
        
            # Объединяем и очищаем текст
            full_text = ' '.join(main_content)
            cleaned_text = re.sub(r'\s+', ' ', full_text).strip()
        
            # Возвращаем ограниченное количество символов (5000-10000)
            return cleaned_text[:10000] if cleaned_text else "Контент не найден"
        
        except Exception as e:
            logger.error(f"Ошибка получения контента для {url}: {str(e)}")
            return ""
