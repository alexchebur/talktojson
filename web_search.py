import requests
import random
import re
import chardet
import logging
import time
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
from config import USER_AGENTS, PRIORITY_SITES, API_TIMEOUT
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
import threading

logger = logging.getLogger(__name__)

class RateLimiter:
    """Простой лимитер запросов для DuckDuckGo API"""
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.last_reset = time.time()
        self.num_calls = 0
        self.lock = threading.Lock()
    
    def acquire(self):
        """Ждем, пока не сможем сделать запрос"""
        with self.lock:
            current = time.time()
            time_since_reset = current - self.last_reset
            
            # Если прошло достаточно времени, сбрасываем счетчик
            if time_since_reset > self.period:
                self.num_calls = 0
                self.last_reset = current
            
            # Если достигнут лимит, ждем
            if self.num_calls >= self.calls:
                sleep_time = self.period - time_since_reset
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # После сна обновляем время сброса
                    self.last_reset = time.time()
                    self.num_calls = 0
                else:
                    # Если период уже прошел, просто сбрасываем счетчик
                    self.num_calls = 0
                    self.last_reset = current
            
            # Увеличиваем счетчик запросов
            self.num_calls += 1

class WebSearcher:
    def __init__(self, delay_range=(1.0, 3.0)):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': random.choice(USER_AGENTS)})
        
        # Настройки Google CSE
        self.api_key = "AIzaSyCNVeNmUgrt-kL5ZI4EkHFoTjTzRSWATX4"
        self.cse_id = "a4f17489c6a0a4414"
        self.priority_sites = PRIORITY_SITES  # Используем из конфига
        
        # Кэширование для DuckDuckGo
        self.ddg_cache = {}
        self.cache_ttl = 300  # 5 минут в секундах
        
        # Лимитер для DuckDuckGo (1 запрос в 2.5 секунды для большей надежности)
        self.ddg_limiter = RateLimiter(calls=1, period=2.5)
        
        # Отслеживание статуса DuckDuckGo
        self.ddg_available = True
        self.last_ddg_error = None
        self.ddg_error_cooldown = 0
        
        # Переменная для временного хранения ошибок (будет обработана основным потоком)
        self.temp_ddg_errors = []
        self.temp_ddg_errors_lock = threading.Lock()
    
    def clear_temp_errors(self):
        """Очищает временные ошибки после их обработки основным потоком"""
        with self.temp_ddg_errors_lock:
            self.temp_ddg_errors = []
    
    def get_temp_errors(self) -> List[dict]:
        """Возвращает временные ошибки и очищает список"""
        with self.temp_ddg_errors_lock:
            errors = self.temp_ddg_errors.copy()
            self.temp_ddg_errors = []
            return errors
    
    def _log_ddg_error(self, error_type: str, error_msg: str, query: str, error_level: str = 'error'):
        """Логгирует ошибку DuckDuckGo во временный буфер"""
        logger.error(f"DuckDuckGo {error_level}: {error_msg} для запроса: {query}")
        
        with self.temp_ddg_errors_lock:
            self.temp_ddg_errors.append({
                'timestamp': time.time(),
                'error_type': error_type,
                'error': error_msg,
                'query': query,
                'type': error_level
            })
    
    def perform_search(self, query: str, max_results: int = 1, query_type="generated") -> Tuple[List[Dict], List[dict]]:
        """Возвращает результаты поиска и список ошибок"""
        try:
            results = self._execute_search(query, max_results)
            for res in results:
                res['query'] = query  # Сохраняем запрос
                res['query_type'] = query_type  # Тип запроса
            
            # Получаем ошибки, накопленные во время поиска
            errors = self.get_temp_errors()
            return results, errors
        except Exception as e:
            error_msg = f"Ошибка поиска: {str(e)}"
            logger.error(error_msg)
            self._log_ddg_error("Общая ошибка поиска", str(e), query, 'error')
            return [], self.get_temp_errors()

    def _execute_search(self, query: str, max_results: int) -> List[Dict]:
        """Выполняет поиск сначала через DuckDuckGo, при неудаче - через Google CSE"""
        cache_key = f"{query}:{max_results}"
        
        # Проверяем кэш для DuckDuckGo
        if cache_key in self.ddg_cache:
            cached_time, results = self.ddg_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.info(f"Используем кэшированные результаты DuckDuckGo для запроса: {query}")
                return results
        
        # Проверяем, не в "черном списке" ли DuckDuckGo из-за предыдущих ошибок
        if not self._is_ddg_available():
            logger.warning("DuckDuckGo временно недоступен из-за предыдущих ошибок, используем Google CSE")
            return self._google_cse_search(query, max_results)
        
        # Сначала пробуем DuckDuckGo
        try:
            duckduckgo_results = self._duckduckgo_search(query, max_results)
            
            # Если получили результаты из DuckDuckGo, возвращаем их и кэшируем
            if duckduckgo_results:
                logger.info(f"Получено {len(duckduckgo_results)} результатов из DuckDuckGo")
                self.ddg_cache[cache_key] = (time.time(), duckduckgo_results)
                return duckduckgo_results
            else:
                logger.warning(f"DuckDuckGo вернул пустые результаты для запроса: {query}")
                self._log_ddg_error("Пустой результат", "DuckDuckGo вернул пустые результаты", query, 'warning')
        except Exception as e:
            logger.error(f"Ошибка при поиске через DuckDuckGo: {str(e)}")
            self._handle_ddg_error(e, query)
        
        # Если DuckDuckGo не сработал, используем Google CSE
        logger.warning("DuckDuckGo не вернул результатов, переключаемся на Google CSE")
        return self._google_cse_search(query, max_results)

    def _is_ddg_available(self) -> bool:
        """Проверяет, доступен ли DuckDuckGo после предыдущих ошибок"""
        if not self.ddg_available:
            # Если прошло достаточно времени с последней ошибки, пробуем снова
            if time.time() > self.ddg_error_cooldown:
                logger.info("Сброс состояния недоступности DuckDuckGo, пробуем снова")
                self.ddg_available = True
                return True
            return False
        return True

    def _handle_ddg_error(self, error, query):
        """Обрабатывает ошибки DuckDuckGo и устанавливает временный запрет на использование"""
        self.last_ddg_error = str(error)
        self.ddg_available = False
        
        # Сохраняем информацию во временный буфер
        self._log_ddg_error("Ошибка API", str(error), query, 'error')
        
        # Устанавливаем разное время простоя в зависимости от типа ошибки
        if "limit" in str(error).lower() or "429" in str(error) or "rate limit" in str(error).lower():
            # При ошибках лимита ждем дольше - 5-10 минут
            cooldown = random.uniform(300, 600)
            logger.warning(f"Обнаружен лимит запросов DuckDuckGo. Приостанавливаем использование на {cooldown:.0f} секунд")
        elif "timeout" in str(error).lower():
            # Для таймаутов ждем меньше - 1-3 минуты
            cooldown = random.uniform(60, 180)
            logger.warning(f"Таймаут DuckDuckGo: {error}. Приостанавливаем использование на {cooldown:.0f} секунд")
        else:
            # Для других ошибок ждем меньше - 2-5 минут
            cooldown = random.uniform(120, 300)
            logger.warning(f"Ошибка DuckDuckGo: {error}. Приостанавливаем использование на {cooldown:.0f} секунд")
        
        self.ddg_error_cooldown = time.time() + cooldown

    @retry(
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict]:
        """Поиск через DuckDuckGo API с улучшенной обработкой ошибок и лимитов"""
        # Используем лимитер для контроля частоты запросов
        self.ddg_limiter.acquire()
        
        # Добавляем небольшую случайную задержку для дополнительной защиты
        time.sleep(random.uniform(0.8, 1.5))
        
        # Формируем URL для DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1',
            't': 'myapp'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Проверка на наличие ошибки в ответе
            if 'Error' in data and data['Error']:
                error_msg = f"DuckDuckGo API вернул ошибку: {data['Error']}"
                self._log_ddg_error("Ошибка API", data['Error'], query, 'error')
                raise Exception(error_msg)
            
            results = []
            
            # 1. Обрабатываем основные результаты (из Results)
            if 'Results' in data and data['Results']:
                for item in data['Results'][:max_results]:
                    if 'FirstURL' in item and item['FirstURL']:
                        full_content = self.get_full_page_content(item['FirstURL'])
                        custom_snippet = self._generate_snippet(full_content)
                        
                        results.append({
                            'title': item.get('Text', 'Без названия')[:150],
                            'url': item['FirstURL'],
                            'snippet': custom_snippet,
                            'full_content': full_content,
                            'source': 'Results',
                            'query': query
                        })
            
            # 2. Обрабатываем связанные темы (RelatedTopics)
            if len(results) < max_results and 'RelatedTopics' in data:
                for topic in data['RelatedTopics']:
                    if 'FirstURL' in topic and topic['FirstURL'] and len(results) < max_results:
                        full_content = self.get_full_page_content(topic['FirstURL'])
                        custom_snippet = self._generate_snippet(full_content)
                        
                        results.append({
                            'title': topic.get('Text', 'Без названия')[:150],
                            'url': topic['FirstURL'],
                            'snippet': custom_snippet,
                            'full_content': full_content,
                            'source': 'RelatedTopics',
                            'query': query
                        })
            
            # 3. Обрабатываем основные результаты (из Heading)
            if len(results) < max_results and 'Heading' in data and data['Heading'] and 'AbstractURL' in data and data['AbstractURL']:
                full_content = self.get_full_page_content(data['AbstractURL'])
                custom_snippet = self._generate_snippet(full_content)
                
                results.append({
                    'title': data.get('Heading', 'Абстракт')[:150],
                    'url': data['AbstractURL'],
                    'snippet': custom_snippet,
                    'full_content': full_content,
                    'source': 'Abstract',
                    'query': query
                })
            
            # Проверяем, есть ли реальные результаты
            if not results:
                self._log_ddg_error("Пустой результат", "DuckDuckGo вернул пустые результаты", query, 'warning')
                return []
            
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
            # Ошибка будет залоггирована в _execute_search
            raise

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
                    'full_content': full_content,
                    'source': 'Google CSE',
                    'query': query
                })
            time.sleep(1.5)
            return results
        except Exception as e:
            logger.error(f"Ошибка выполнения поиска через Google CSE: {str(e)}")
            # Не логгируем ошибку Google CSE как ошибку DuckDuckGo
            return []

    def _generate_snippet(self, full_content: str) -> str:
        """Генерация улучшенного сниппета из полного контента"""
        if not full_content or full_content == "Контент не найден" or len(full_content.strip()) == 0:
            return "Контент не найден или недоступен"
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
