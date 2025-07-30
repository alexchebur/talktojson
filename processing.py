import re
import os
import numpy as np
from difflib import SequenceMatcher
from typing import List
from utils import clean_keyword
from rank_bm25 import BM25Okapi


class DataProcessor:
    def __init__(self, index_builder):
        self.index_builder = index_builder

