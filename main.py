class BM25SearchEngine:
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
        self.bm25 = None
        self.chunks_info = []
        self.doc_index = defaultdict(list)
        self.is_index_loaded = False
        self.index_path = os.path.join(DATA_DIR, "bm25_index.pkl")

    def load_index(self) -> bool:
        """Загружает индекс из pickle файла"""
        try:
            if not os.path.exists(self.index_path):
                st.error(f"Файл индекса {self.index_path} не найден")
                return False

            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                
                if isinstance(data, tuple) and len(data) == 3:
                    self.bm25, self.chunks_info, self.doc_index = data
                elif isinstance(data, dict):
                    self.bm25 = data.get('bm25')
                    self.chunks_info = data.get('chunks_info', [])
                    self.doc_index = data.get('doc_index', defaultdict(list))
                else:
                    st.error("Неверный формат данных в файле индекса")
                    return False
                
                # Проверяем, что bm25 был успешно загружен
                if self.bm25 is None:
                    st.error("Не удалось загрузить индекс BM25 из файла")
                    return False
                
                self.is_index_loaded = True
                return True
                
        except Exception as e:
            st.error(f"Ошибка загрузки индекса: {e}")
            return False

    def search(self, query: str, top_n: int = 5) -> List[Dict]:
        """Выполняет поиск по индексу BM25"""
        if not self.is_index_loaded:
            if not self.load_index():
                st.error("Не удалось загрузить индекс для поиска")
                return []

        if self.bm25 is None:
            st.error("Индекс BM25 не инициализирован")
            return []

        tokens = self.preprocessor.preprocess(query)
        if not tokens:
            st.warning("Не удалось извлечь ключевые слова из запроса")
            return []

        try:
            scores = self.bm25.get_scores(tokens)
            best_indices = np.argsort(scores)[-top_n:][::-1]

            results = []
            for idx in best_indices:
                if idx < len(self.chunks_info):
                    result = {
                        'doc_id': self.chunks_info[idx].get('doc_id', f"doc_{idx}"),
                        'doc_name': self.chunks_info[idx].get('doc_name', "Без названия"),
                        'chunk_text': self.chunks_info[idx].get('chunk_text', ""),
                        'score': float(scores[idx])
                    }
                    results.append(result)
            return results
        except Exception as e:
            st.error(f"Ошибка при выполнении поиска: {e}")
            return []
