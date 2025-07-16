# Промпты для разных этапов обработки
SYSTEM_PROMPT = """
### РОЛЬ
Вы - опытный юрист...
[остальная часть промпта]
"""

QUERY_GENERATION_PROMPT = """
### РОЛЬ
Ты - опытный юрист...
[остальная часть промпта]
"""

# Для будущего развития - цепочки промптов
PROMPT_CHAINS = {
    "initial_analysis": {
        "system": "...",
        "user": "..."
    },
    "legal_research": {
        "system": "...",
        "user": "..."
    },
    "conclusion_generation": {
        "system": "...",
        "user": "..."
    }
}

def get_prompt(prompt_name: str, variables: dict) -> str:
    """Получение форматированного промпта"""
    templates = {
        "system": SYSTEM_PROMPT,
        "query_generation": QUERY_GENERATION_PROMPT
    }
    
    template = templates.get(prompt_name)
    if not template:
        raise ValueError(f"Unknown prompt name: {prompt_name}")
    
    return template.format(**variables)
