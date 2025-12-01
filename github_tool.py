"""
Модуль для работы с GitHub через tool для GigaChat API.

Реализует набор функций для работы с репозиториями, issues, pull requests,
файлами и другими возможностями GitHub API.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GitHub API базовый URL
GITHUB_API_BASE = "https://api.github.com"


def get_github_token() -> str:
    """
    Получает GitHub токен из переменных окружения.

    Returns:
        str: GitHub токен

    Raises:
        RuntimeError: Если токен не найден
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GITHUB_TOKEN не найден в переменных окружения. "
            "Создайте токен на https://github.com/settings/tokens"
        )
    return token


def _make_github_request(
    method: str,
    endpoint: str,
    data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Выполняет запрос к GitHub API.

    Args:
        method: HTTP метод (GET, POST, PATCH, DELETE)
        endpoint: API endpoint (например, /repos/owner/repo)
        data: Данные для POST/PATCH запросов
        params: Query параметры

    Returns:
        dict: JSON ответ от API
    """
    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    url = f"{GITHUB_API_BASE}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        # Для DELETE запросов может не быть JSON ответа
        if response.status_code == 204:
            return {"success": True, "message": "Operation completed successfully"}

        return response.json()

    except requests.exceptions.HTTPError as e:
        error_msg = f"GitHub API error: {e.response.status_code}"
        try:
            error_data = e.response.json()
            error_msg = f"{error_msg} - {error_data.get('message', str(e))}"
        except Exception:
            error_msg = f"{error_msg} - {str(e)}"

        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error making GitHub request: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


# ---------------------------------------------------------------------------
# GitHub Tool Functions
# ---------------------------------------------------------------------------

def github_get_repo(owner: str, repo: str) -> Dict[str, Any]:
    """
    Получает информацию о репозитории.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория

    Returns:
        dict: Информация о репозитории
    """
    return _make_github_request("GET", f"/repos/{owner}/{repo}")


def github_list_issues(
    owner: str,
    repo: str,
    state: str = "open",
    labels: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Получает список issues в репозитории.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        state: Статус issues (open, closed, all)
        labels: Фильтр по меткам (через запятую)
        limit: Максимальное количество issues

    Returns:
        dict: Список issues
    """
    params = {
        "state": state,
        "per_page": min(limit, 100),
    }
    if labels:
        params["labels"] = labels

    result = _make_github_request("GET", f"/repos/{owner}/{repo}/issues", params=params)

    if isinstance(result, list):
        return {"issues": result[:limit], "count": len(result)}
    return result


def github_get_issue(owner: str, repo: str, issue_number: int) -> Dict[str, Any]:
    """
    Получает детали конкретного issue.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        issue_number: Номер issue

    Returns:
        dict: Информация об issue
    """
    return _make_github_request("GET", f"/repos/{owner}/{repo}/issues/{issue_number}")


def github_create_issue(
    owner: str,
    repo: str,
    title: str,
    body: Optional[str] = None,
    labels: Optional[List[str]] = None,
    assignees: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Создает новый issue в репозитории.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        title: Заголовок issue
        body: Описание issue
        labels: Список меток
        assignees: Список исполнителей

    Returns:
        dict: Созданный issue
    """
    data: Dict[str, Any] = {"title": title}
    if body:
        data["body"] = body
    if labels:
        data["labels"] = labels
    if assignees:
        data["assignees"] = assignees

    return _make_github_request("POST", f"/repos/{owner}/{repo}/issues", data=data)


def github_update_issue(
    owner: str,
    repo: str,
    issue_number: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
    state: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Обновляет существующий issue.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        issue_number: Номер issue
        title: Новый заголовок
        body: Новое описание
        state: Новый статус (open, closed)
        labels: Новые метки

    Returns:
        dict: Обновленный issue
    """
    data: Dict[str, Any] = {}
    if title:
        data["title"] = title
    if body:
        data["body"] = body
    if state:
        data["state"] = state
    if labels:
        data["labels"] = labels

    return _make_github_request("PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", data=data)


def github_create_comment(
    owner: str,
    repo: str,
    issue_number: int,
    body: str,
) -> Dict[str, Any]:
    """
    Добавляет комментарий к issue или pull request.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        issue_number: Номер issue/PR
        body: Текст комментария

    Returns:
        dict: Созданный комментарий
    """
    data = {"body": body}
    return _make_github_request("POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments", data=data)


def github_list_repo_contents(
    owner: str,
    repo: str,
    path: str = "",
    ref: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Получает содержимое директории в репозитории (список файлов и папок).

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        path: Путь к директории (пустая строка для корня)
        ref: Ветка или коммит (по умолчанию - main/master)

    Returns:
        dict: Список файлов и директорий
    """
    params = {}
    if ref:
        params["ref"] = ref

    endpoint = f"/repos/{owner}/{repo}/contents"
    if path:
        endpoint = f"{endpoint}/{path}"

    result = _make_github_request("GET", endpoint, params=params)

    if isinstance(result, list):
        # Форматируем список для удобного отображения
        formatted_items = []
        for item in result:
            formatted_items.append({
                "name": item.get("name"),
                "type": item.get("type"),  # file или dir
                "path": item.get("path"),
                "size": item.get("size", 0),
                "url": item.get("html_url"),
            })
        return {
            "path": path or "/",
            "items": formatted_items,
            "count": len(formatted_items),
        }
    return result


def github_get_repo_tree(
    owner: str,
    repo: str,
    branch: Optional[str] = None,
    recursive: bool = False,
) -> Dict[str, Any]:
    """
    Получает полное дерево файлов репозитория (git tree).

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        branch: Ветка (по умолчанию - default branch)
        recursive: Получить всё дерево рекурсивно (может быть большим)

    Returns:
        dict: Дерево файлов репозитория
    """
    # Сначала получаем информацию о ветке
    if not branch:
        # Получаем default branch
        repo_info = _make_github_request("GET", f"/repos/{owner}/{repo}")
        if "error" in repo_info:
            return repo_info
        branch = repo_info.get("default_branch", "main")

    # Получаем SHA последнего коммита в ветке
    branch_info = _make_github_request("GET", f"/repos/{owner}/{repo}/branches/{branch}")
    if "error" in branch_info:
        return branch_info

    commit_sha = branch_info.get("commit", {}).get("sha")
    if not commit_sha:
        return {"error": "Не удалось получить SHA коммита"}

    # Получаем дерево
    params = {}
    if recursive:
        params["recursive"] = "1"

    result = _make_github_request("GET", f"/repos/{owner}/{repo}/git/trees/{commit_sha}", params=params)

    if isinstance(result, dict) and "tree" in result:
        # Форматируем дерево для удобного отображения
        formatted_tree = []
        for item in result["tree"]:
            formatted_tree.append({
                "path": item.get("path"),
                "type": item.get("type"),  # blob (file), tree (dir)
                "size": item.get("size", 0),
                "sha": item.get("sha"),
            })

        return {
            "branch": branch,
            "sha": commit_sha,
            "truncated": result.get("truncated", False),
            "tree": formatted_tree,
            "count": len(formatted_tree),
        }

    return result


def github_list_pull_requests(
    owner: str,
    repo: str,
    state: str = "open",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Получает список pull requests в репозитории.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        state: Статус PR (open, closed, all)
        limit: Максимальное количество PR

    Returns:
        dict: Список pull requests
    """
    params = {
        "state": state,
        "per_page": min(limit, 100),
    }

    result = _make_github_request("GET", f"/repos/{owner}/{repo}/pulls", params=params)

    if isinstance(result, list):
        return {"pull_requests": result[:limit], "count": len(result)}
    return result


def github_get_file_content(
    owner: str,
    repo: str,
    path: str,
    ref: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Получает содержимое файла из репозитория.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        path: Путь к файлу в репозитории
        ref: Ветка или коммит (по умолчанию - main/master)

    Returns:
        dict: Информация о файле и его содержимое
    """
    params = {}
    if ref:
        params["ref"] = ref

    result = _make_github_request("GET", f"/repos/{owner}/{repo}/contents/{path}", params=params)

    # Декодируем base64 содержимое если присутствует
    if isinstance(result, dict) and "content" in result and result.get("encoding") == "base64":
        import base64
        try:
            decoded_content = base64.b64decode(result["content"]).decode("utf-8")
            result["decoded_content"] = decoded_content
        except Exception as e:
            logger.warning(f"Failed to decode file content: {e}")

    return result


def github_search_code(
    query: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Поиск кода в GitHub.

    Args:
        query: Поисковый запрос (например, "repo:owner/repo language:python")
        limit: Максимальное количество результатов

    Returns:
        dict: Результаты поиска
    """
    params = {
        "q": query,
        "per_page": min(limit, 100),
    }

    result = _make_github_request("GET", "/search/code", params=params)

    if isinstance(result, dict) and "items" in result:
        return {
            "items": result["items"][:limit],
            "total_count": result.get("total_count", 0),
        }
    return result


def github_list_commits(
    owner: str,
    repo: str,
    sha: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Получает список коммитов в репозитории.

    Args:
        owner: Владелец репозитория
        repo: Название репозитория
        sha: Ветка или коммит (по умолчанию - main/master)
        limit: Максимальное количество коммитов

    Returns:
        dict: Список коммитов
    """
    params = {
        "per_page": min(limit, 100),
    }
    if sha:
        params["sha"] = sha

    result = _make_github_request("GET", f"/repos/{owner}/{repo}/commits", params=params)

    if isinstance(result, list):
        return {"commits": result[:limit], "count": len(result)}
    return result


def github_get_user_info(username: Optional[str] = None) -> Dict[str, Any]:
    """
    Получает информацию о пользователе GitHub.

    Args:
        username: Имя пользователя (если None - текущий аутентифицированный пользователь)

    Returns:
        dict: Информация о пользователе
    """
    if username:
        return _make_github_request("GET", f"/users/{username}")
    return _make_github_request("GET", "/user")


# ---------------------------------------------------------------------------
# Tool Definitions for GigaChat
# ---------------------------------------------------------------------------

GITHUB_TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "github_get_repo",
            "description": "Получает информацию о репозитории GitHub (описание, звезды, форки, язык программирования и т.д.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория (например, 'octocat')",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория (например, 'Hello-World')",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_issues",
            "description": "Получает список issues в репозитории с возможностью фильтрации",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Статус issues (open, closed, all)",
                    },
                    "labels": {
                        "type": "string",
                        "description": "Фильтр по меткам через запятую (например, 'bug,help wanted')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество issues",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_get_issue",
            "description": "Получает детальную информацию о конкретном issue",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "Номер issue",
                    },
                },
                "required": ["owner", "repo", "issue_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_issue",
            "description": "Создает новый issue в репозитории",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "title": {
                        "type": "string",
                        "description": "Заголовок issue",
                    },
                    "body": {
                        "type": "string",
                        "description": "Описание issue в формате Markdown",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список меток для issue",
                    },
                    "assignees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список исполнителей (usernames)",
                    },
                },
                "required": ["owner", "repo", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_create_comment",
            "description": "Добавляет комментарий к issue или pull request",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "issue_number": {
                        "type": "integer",
                        "description": "Номер issue или pull request",
                    },
                    "body": {
                        "type": "string",
                        "description": "Текст комментария в формате Markdown",
                    },
                },
                "required": ["owner", "repo", "issue_number", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_repo_contents",
            "description": "Получает список файлов и папок в директории репозитория",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "path": {
                        "type": "string",
                        "description": "Путь к директории (пустая строка для корня репозитория)",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Ветка или коммит (опционально)",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_get_repo_tree",
            "description": "Получает полное дерево файлов репозитория (все файлы и папки). Полезно для получения структуры всего репозитория.",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Ветка (по умолчанию - default branch репозитория)",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Получить полное дерево рекурсивно (все файлы и папки)",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_pull_requests",
            "description": "Получает список pull requests в репозитории",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Статус pull requests",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество PR",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_get_file_content",
            "description": "Получает содержимое файла из репозитория",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу в репозитории (например, 'src/main.py')",
                    },
                    "ref": {
                        "type": "string",
                        "description": "Ветка или коммит (опционально)",
                    },
                },
                "required": ["owner", "repo", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_search_code",
            "description": "Поиск кода в GitHub репозиториях",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос (например, 'repo:owner/repo language:python function')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество результатов",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_list_commits",
            "description": "Получает список коммитов в репозитории",
            "parameters": {
                "type": "object",
                "properties": {
                    "owner": {
                        "type": "string",
                        "description": "Владелец репозитория",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Название репозитория",
                    },
                    "sha": {
                        "type": "string",
                        "description": "Ветка или коммит (опционально)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество коммитов",
                    },
                },
                "required": ["owner", "repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_get_user_info",
            "description": "Получает информацию о пользователе GitHub",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Имя пользователя (если не указано - текущий авторизованный пользователь)",
                    },
                },
                "required": [],
            },
        },
    },
]


def get_github_tools() -> List[Dict[str, Any]]:
    """
    Возвращает список всех GitHub tools для регистрации в GigaChat.

    Returns:
        list: Список определений GitHub tools
    """
    return GITHUB_TOOLS_DEFINITIONS


def execute_github_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выполняет вызов GitHub tool на основе данных из tool_call.

    Args:
        tool_call: Словарь с данными вызова tool от GigaChat API

    Returns:
        dict: Результат выполнения функции
    """
    function_name = tool_call.get("function", {}).get("name", "")

    # Парсим аргументы
    arguments = tool_call.get("function", {}).get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            return {"error": f"Ошибка парсинга аргументов: {str(e)}"}

    # Маппинг функций
    function_map = {
        "github_get_repo": github_get_repo,
        "github_list_issues": github_list_issues,
        "github_get_issue": github_get_issue,
        "github_create_issue": github_create_issue,
        "github_update_issue": github_update_issue,
        "github_create_comment": github_create_comment,
        "github_list_repo_contents": github_list_repo_contents,
        "github_get_repo_tree": github_get_repo_tree,
        "github_list_pull_requests": github_list_pull_requests,
        "github_get_file_content": github_get_file_content,
        "github_search_code": github_search_code,
        "github_list_commits": github_list_commits,
        "github_get_user_info": github_get_user_info,
    }

    func = function_map.get(function_name)
    if not func:
        return {"error": f"Неизвестная функция: {function_name}"}

    try:
        # Вызываем функцию с распакованными аргументами
        result = func(**arguments)
        return result
    except Exception as e:
        error_msg = f"Ошибка выполнения {function_name}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def register_github_tools(tools_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Регистрирует GitHub tools в списке tools для GigaChat.

    Args:
        tools_list: Опциональный список существующих tools

    Returns:
        list: Список tools с добавленными GitHub tools
    """
    if tools_list is None:
        tools_list = []

    # Получаем все GitHub tools
    github_tools = get_github_tools()

    # Проверяем, не добавлены ли уже GitHub tools
    existing_names = {
        tool.get("function", {}).get("name")
        for tool in tools_list
    }

    # Добавляем только те tools, которых еще нет
    for tool in github_tools:
        tool_name = tool.get("function", {}).get("name")
        if tool_name not in existing_names:
            tools_list.append(tool)

    return tools_list


# ---------------------------------------------------------------------------
# Пример использования
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("Пример использования GitHub tool")
    print("=" * 80)

    # 1. Получение информации о репозитории
    print("\n1. Получение информации о репозитории:")
    print("-" * 80)
    repo_info = github_get_repo("octocat", "Hello-World")
    if "error" not in repo_info:
        print(f"Репозиторий: {repo_info.get('full_name')}")
        print(f"Описание: {repo_info.get('description')}")
        print(f"Звезды: {repo_info.get('stargazers_count')}")
        print(f"Язык: {repo_info.get('language')}")
    else:
        print(f"Ошибка: {repo_info['error']}")

    # 2. Список issues
    print("\n2. Получение списка issues:")
    print("-" * 80)
    issues = github_list_issues("octocat", "Hello-World", limit=3)
    if "error" not in issues:
        print(f"Найдено issues: {issues.get('count', 0)}")
        for issue in issues.get('issues', [])[:3]:
            print(f"  - #{issue['number']}: {issue['title']}")
    else:
        print(f"Ошибка: {issues['error']}")

    # 3. Регистрация tools
    print("\n3. Регистрация GitHub tools:")
    print("-" * 80)
    tools = register_github_tools()
    print(f"Зарегистрировано tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool['function']['name']}")

    print("\n" + "=" * 80)
    print("Пример завершен")
    print("=" * 80)
