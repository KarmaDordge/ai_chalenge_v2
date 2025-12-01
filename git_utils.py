"""
Утилиты для работы с Git репозиториями.
"""

from __future__ import annotations

import os
import subprocess
import logging
from typing import Optional, Dict, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_git_repo_structure(repo_path: Optional[str] = None) -> str:
    """
    Получает структуру проекта из Git репозитория.
    
    Args:
        repo_path: Путь к репозиторию (если None, используется текущая директория)
        
    Returns:
        Строка с древовидной структурой проекта
    """
    if repo_path is None:
        repo_path = os.getcwd()
    
    # Проверяем, что это git репозиторий
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.exists(git_dir):
        return "Текущая директория не является Git репозиторием."
    
    try:
        # Получаем список всех файлов в репозитории (исключая .git)
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode != 0:
            return f"Ошибка при получении структуры репозитория: {result.stderr}"
        
        files = result.stdout.strip().split('\n')
        files = [f for f in files if f.strip()]  # Убираем пустые строки
        
        if not files:
            return "Репозиторий пуст."
        
        # Строим дерево
        tree = {}
        for file_path in files:
            parts = file_path.split('/')
            current = tree
            
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        
        # Форматируем дерево в строку
        def format_tree(node: Dict[str, Any], prefix: str = "", is_last: bool = True) -> str:
            lines = []
            items = sorted(node.items())
            
            for i, (name, children) in enumerate(items):
                is_last_item = i == len(items) - 1
                current_prefix = prefix + ("└── " if is_last_item else "├── ")
                lines.append(current_prefix + name)
                
                if children:
                    next_prefix = prefix + ("    " if is_last_item else "│   ")
                    lines.append(format_tree(children, next_prefix, is_last_item))
            
            return "\n".join(lines)
        
        structure = format_tree(tree)
        
        # Добавляем информацию о репозитории
        try:
            remote_result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            remote_url = remote_result.stdout.strip() if remote_result.returncode == 0 else None
            
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
            
            header = "Структура проекта:\n\n"
            if remote_url:
                header += f"Репозиторий: {remote_url}\n"
            if current_branch:
                header += f"Ветка: {current_branch}\n"
            header += "\n"
            
            return header + structure
        except subprocess.TimeoutExpired:
            # Если не удалось получить информацию о репозитории, возвращаем только структуру
            return "Структура проекта:\n\n" + structure
        except Exception:
            # Если не удалось получить информацию о репозитории, возвращаем только структуру
            return "Структура проекта:\n\n" + structure
        
    except subprocess.TimeoutExpired:
        return "Превышено время ожидания при получении структуры репозитория."
    except Exception as e:
        logger.error(f"Ошибка при получении структуры Git репозитория: {e}")
        return f"Ошибка при получении структуры репозитория: {str(e)}"


def get_git_repo_info(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Получает информацию о Git репозитории.
    
    Args:
        repo_path: Путь к репозиторию (если None, используется текущая директория)
        
    Returns:
        Словарь с информацией о репозитории
    """
    if repo_path is None:
        repo_path = os.getcwd()
    
    info = {
        "is_git_repo": False,
        "remote_url": None,
        "current_branch": None,
        "total_files": 0,
    }
    
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.exists(git_dir):
        return info
    
    info["is_git_repo"] = True
    
    try:
        # Получаем remote URL
        remote_result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if remote_result.returncode == 0:
            info["remote_url"] = remote_result.stdout.strip()
        
        # Получаем текущую ветку
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if branch_result.returncode == 0:
            info["current_branch"] = branch_result.stdout.strip()
        
        # Получаем количество файлов
        files_result = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if files_result.returncode == 0:
            files = [f for f in files_result.stdout.strip().split('\n') if f.strip()]
            info["total_files"] = len(files)
        
    except Exception as e:
        logger.error(f"Ошибка при получении информации о Git репозитории: {e}")
    
    return info

