"""
Модуль для работы с конфигурационными MD файлами репозиториев.

Формат MD файла:
```yaml
---
repository: owner/repo
rag_enabled: true
rag_threshold: 0.7
---
```

Или в формате Markdown с frontmatter:
```markdown
---
repository: owner/repo
rag_enabled: true
rag_threshold: 0.7
---

# Описание репозитория

Дополнительная информация о репозитории...
```
"""

from __future__ import annotations

import os
import re
import logging
from typing import Dict, Optional, Any
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
REPO_CONFIG_DIR = os.path.join(CONFIG_DIR, "repos")

# Создаем директорию для конфигураций репозиториев
os.makedirs(REPO_CONFIG_DIR, exist_ok=True)


def parse_md_config(file_path: str) -> Dict[str, Any]:
    """
    Парсит MD файл с конфигурацией репозитория.
    
    Args:
        file_path: Путь к MD файлу
        
    Returns:
        Словарь с настройками репозитория
    """
    config = {
        "repository": None,
        "rag_enabled": False,
        "rag_threshold": 0.7,
        "file_path": file_path,
    }
    
    if not os.path.exists(file_path):
        logger.warning(f"Конфигурационный файл не найден: {file_path}")
        return config
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Ищем frontmatter в формате YAML между ---
        frontmatter_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL | re.MULTILINE)
        
        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)
            
            # Парсим YAML-like структуру
            for line in frontmatter.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Парсим ключ: значение
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == "repository":
                        config["repository"] = value
                    elif key == "rag_enabled":
                        config["rag_enabled"] = value.lower() in ("true", "1", "yes", "on")
                    elif key == "rag_threshold":
                        try:
                            threshold = float(value)
                            config["rag_threshold"] = max(0.0, min(1.0, threshold))
                        except ValueError:
                            logger.warning(f"Некорректное значение rag_threshold: {value}")
        
        logger.info(f"Загружена конфигурация из {file_path}: repository={config['repository']}")
        
    except Exception as e:
        logger.error(f"Ошибка при парсинге конфигурационного файла {file_path}: {e}")
    
    return config


def list_repo_configs() -> list[Dict[str, Any]]:
    """
    Возвращает список всех доступных конфигурационных MD файлов.
    
    Returns:
        Список словарей с информацией о конфигурациях
    """
    configs = []
    
    if not os.path.exists(REPO_CONFIG_DIR):
        return configs
    
    for file_name in os.listdir(REPO_CONFIG_DIR):
        if file_name.endswith('.md'):
            file_path = os.path.join(REPO_CONFIG_DIR, file_name)
            config = parse_md_config(file_path)
            config["name"] = file_name[:-3]  # Убираем .md
            configs.append(config)
    
    return configs


def get_repo_config(config_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Получает конфигурацию репозитория по имени.
    
    Args:
        config_name: Имя конфигурации (без .md) или None
        
    Returns:
        Словарь с настройками или None
    """
    if not config_name:
        return None
    
    file_path = os.path.join(REPO_CONFIG_DIR, f"{config_name}.md")
    
    if not os.path.exists(file_path):
        logger.warning(f"Конфигурационный файл не найден: {file_path}")
        return None
    
    config = parse_md_config(file_path)
    config["name"] = config_name
    return config


def create_example_config() -> str:
    """
    Создает пример конфигурационного файла.
    
    Returns:
        Путь к созданному файлу
    """
    example_path = os.path.join(REPO_CONFIG_DIR, "example.md")
    
    example_content = """---
repository: owner/repo
rag_enabled: true
rag_threshold: 0.7
---

# Пример конфигурации репозитория

Это пример конфигурационного файла для настройки работы с репозиторием.

## Параметры:

- `repository`: Владелец и название репозитория (например, `octocat/Hello-World`)
- `rag_enabled`: Включить ли использование RAG (true/false)
- `rag_threshold`: Порог релевантности для RAG (0.0 - 1.0)
"""
    
    with open(example_path, "w", encoding="utf-8") as f:
        f.write(example_content)
    
    logger.info(f"Создан пример конфигурационного файла: {example_path}")
    return example_path

