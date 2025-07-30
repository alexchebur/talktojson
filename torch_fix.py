# torch_fix.py
import os
os.environ['STREAMLIT_DISABLE_WATCHDOG'] = 'true'
os.environ['TORCH_DISABLE_PATH_CHECK'] = 'true'
import torch
import warnings

def apply_torch_fix():
    """Применяет все необходимые фиксы для совместимости torch со Streamlit"""
    # Фикс для __path__._path
    if hasattr(torch._classes, '__path__'):
        delattr(torch._classes, '__path__')
    
    # Отключаем ненужные предупреждения
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Применяем фикс при импорте
apply_torch_fix()
