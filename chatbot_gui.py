"""
Веб-чат с GigaChat и Hugging Face на Flask.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import re
import time
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
import markdown
from flask import (
    Flask,
    g,
    flash,
    get_flashed_messages,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from huggingface_hub import InferenceClient
from gigachat_client import ask_gigachat
from git_utils import get_git_repo_structure
from repo_config import list_repo_configs, get_repo_config
from github_tool import github_get_repo_tree

# Утилиты, вынесенные в отдельный модуль
from chat_utils import (
    slugify as cu_slugify,
    default_model_for as cu_default_model_for,
    default_chat_state as cu_default_chat_state,
    parse_temperature,
    estimate_tokens as cu_estimate_tokens,
    estimate_request_tokens as cu_estimate_request_tokens,
    calculate_gigachat_cost as cu_calculate_gigachat_cost,
    validate_history_entry as cu_validate_history_entry,
    extract_tokens_from_usage as cu_extract_tokens_from_usage,
    create_meta_dict as cu_create_meta_dict,
    handle_hf_http_error as cu_handle_hf_http_error,
    handle_hf_response_error as cu_handle_hf_response_error,
)
from db_utils import (
    get_db as db_get_db,
    close_db as db_close_db,
    db_ensure_session,
    db_add_message,
    db_clear_session,
    db_wal_checkpoint,
    db_list_sessions,
    db_get_messages,
    db_load_presets,
    db_upsert_preset,
    make_unique_preset_key,
)

# Настраиваем переменные окружения
load_dotenv()

# Настройка логирования
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")
DB_PATH = os.path.join(HISTORY_DIR, "chat_history.db")
PROMPTS_FILE_PATH = os.path.join(CONFIG_DIR, "preset_prompts.json")
PROMPT_NAMES_FILE_PATH = os.path.join(CONFIG_DIR, "preset_prompt_names.json")

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

DEFAULT_PRESET_PROMPTS = OrderedDict(
    {
        "no_settings": (
            "Ты — полезный AI-ассистент. "
            "Помогай пользователю решать его задачи максимально эффективно."
        )
    }
)

DEFAULT_PRESET_NAMES = OrderedDict({"no_settings": "Без настроек"})

DEFAULT_PRESET_KEY = "no_settings"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_PROVIDER = "gigachat"

AVAILABLE_PROVIDERS = OrderedDict(
    {
        "gigachat": {
            "title": "GigaChat API",
            "models": OrderedDict(
                {
                    "GigaChat": {
                        "label": "GigaChat Lite",
                        "task": "chat",
                        "mode": "chat-completion",
                    },
                    "GigaChat-Pro": {
                        "label": "GigaChat Pro",
                        "task": "chat",
                        "mode": "chat-completion",
                    },
                    "GigaChat-Pro-Max": {
                        "label": "GigaChat Pro Max",
                        "task": "chat",
                        "mode": "chat-completion",
                    },
                }
            ),
        },
        "huggingface": {
            "title": "Hugging Face API",
            "models": OrderedDict(
                {
                    "deepseek-ai/DeepSeek-R1": {
                        "label": "DeepSeek R1",
                        "task": "text-generation",
                        "mode": "chat-completion",
                    },
                    "katanemo/Arch-Router-1.5B": {
                        "label": "Arch Router 1.5B",
                        "task": "text-generation",
                        "mode": "chat-completion",
                    },
                    "Sao10K/L3-8B-Stheno-v3.2": {
                        "label": "L3-8B Stheno v3.2",
                        "task": "text-generation",
                        "mode": "chat-completion",
                    },
                }
            ),
        },
    }
)

SESSION_KEY = "chat_state"
SESSION_ID_KEY = "chat_session_id"  # ID сессии для файлового хранилища


# ---------------------------------------------------------------------------
# Идентификатор сессии (используется для привязки истории в БД)
# ---------------------------------------------------------------------------


def _get_session_id() -> str:
    """Получает или создает UUID сессии (используется для строк в БД)."""
    if SESSION_ID_KEY not in session:
        import uuid
        session[SESSION_ID_KEY] = str(uuid.uuid4())
        session.modified = True
    return session[SESSION_ID_KEY]

def _start_new_session() -> str:
    """Генерирует новый UUID чата и сохраняет его в cookie-сессии."""
    import uuid
    new_id = str(uuid.uuid4())
    session[SESSION_ID_KEY] = new_id
    session.modified = True
    return new_id
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-me")


# Фильтр для форматирования текста с поддержкой Markdown
@app.template_filter('format_markdown')
def format_markdown(text: str) -> str:
    """
    Форматирует текст с поддержкой Markdown и базовой обработкой LaTeX формул.
    """
    if not text:
        return ""
    
    import re
    from markupsafe import Markup
    
    # Временно заменяем формулы на плейсхолдеры перед обработкой Markdown
    formula_placeholders = {}
    placeholder_counter = 0
    
    # Блоковые формулы $$...$$
    def replace_block_formula(match):
        nonlocal placeholder_counter
        placeholder = f"__FORMULA_BLOCK_{placeholder_counter}__"
        formula_placeholders[placeholder] = f'<div class="math-block">{match.group(1).strip()}</div>'
        placeholder_counter += 1
        return placeholder
    
    text = re.sub(
        r'\$\$(.*?)\$\$',
        replace_block_formula,
        text,
        flags=re.DOTALL
    )
    
    # Инлайн формулы $...$ (но не $$)
    def replace_inline_formula(match):
        nonlocal placeholder_counter
        placeholder = f"__FORMULA_INLINE_{placeholder_counter}__"
        formula_placeholders[placeholder] = f'<span class="math-inline">{match.group(1)}</span>'
        placeholder_counter += 1
        return placeholder
    
    text = re.sub(
        r'(?<!\$)\$([^$\n]+?)\$(?!\$)',
        replace_inline_formula,
        text
    )
    
    # Конвертируем Markdown в HTML
    md = markdown.Markdown(
        extensions=[
            'codehilite',
            'fenced_code',
            'tables',
            'nl2br',  # Автоматические переносы строк
        ],
        extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': False,
            }
        }
    )
    
    html = md.convert(text)
    
    # Восстанавливаем формулы
    for placeholder, formula_html in formula_placeholders.items():
        html = html.replace(placeholder, formula_html)
    
    return Markup(html)

@app.teardown_appcontext
def close_db(exception: Optional[BaseException]) -> None:  # noqa: ARG001
    """Closes DB connection at the end of request."""
    db_close_db(exception)


def _db_ensure_session(session_uuid: str, title: Optional[str] = None) -> None:
    db_ensure_session(session_uuid, title)


def _db_add_message(session_uuid: str, role: str, content: str, meta: Optional[dict]) -> None:
    db_add_message(session_uuid, role, content, meta)


def _db_clear_session(session_uuid: str) -> None:
    db_clear_session(session_uuid)


def _db_list_sessions(limit: int = 100) -> list[dict]:
    return db_list_sessions(limit)


def _db_get_messages(session_uuid: str) -> list[dict]:
    return db_get_messages(session_uuid)


# ---------------------------------------------------------------------------
# Утилиты для загрузки и подготовки промптов (переведены на хранение в SQLite)
# ---------------------------------------------------------------------------

def _db_load_presets() -> OrderedDict[str, Dict[str, str]]:
    return db_load_presets()


def _db_upsert_preset(key: str, name: str, prompt: str) -> None:
    db_upsert_preset(key, name, prompt)


def load_presets() -> OrderedDict[str, Dict[str, str]]:
    """Загружает предустановки из БД, при отсутствии — заполняет значениями по умолчанию."""
    presets = _db_load_presets()
    if not presets:
        for key, prompt in DEFAULT_PRESET_PROMPTS.items():
            name = DEFAULT_PRESET_NAMES.get(key, key)
            _db_upsert_preset(key, name, prompt)
        presets = _db_load_presets()
    return presets


def write_presets(
    prompts: OrderedDict[str, str], names: OrderedDict[str, str]
) -> None:
    """Сохраняет промпты и названия в БД."""
    for key, prompt in prompts.items():
        name = names.get(key, key)
        _db_upsert_preset(key, name, prompt)


def slugify(title: str) -> str:
    """Преобразует строку в slug (через chat_utils)."""
    return cu_slugify(title)


def default_model_for(provider: str) -> str:
    """Возвращает модель по умолчанию для провайдера (через chat_utils)."""
    return cu_default_model_for(AVAILABLE_PROVIDERS, provider)



def default_chat_state(presets: OrderedDict[str, Dict[str, str]]) -> Dict[str, object]:
    """Создает состояние чата по умолчанию (обертка над chat_utils.default_chat_state)."""
    state = cu_default_chat_state(
        presets=presets,
        available_providers=AVAILABLE_PROVIDERS,
        default_preset_key=DEFAULT_PRESET_KEY,
        default_provider=DEFAULT_PROVIDER,
        default_temperature=DEFAULT_TEMPERATURE,
    )
    # Добавляем use_local_vectors и rag_threshold по умолчанию
    state["use_local_vectors"] = False
    state["rag_threshold"] = 0.7
    state["repo_config"] = None  # Имя выбранного MD файла конфигурации
    return state




def estimate_tokens(text: str) -> int:
    """Приблизительный подсчет токенов: ~1 токен = 4 символа (через chat_utils)."""
    return cu_estimate_tokens(text)


def estimate_request_tokens(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
) -> int:
    """Подсчитывает общее количество токенов в запросе (через chat_utils)."""
    return cu_estimate_request_tokens(system_prompt, history, user_message)


def calculate_gigachat_cost(
    prompt_tokens: int, completion_tokens: int, model: str
) -> Optional[float]:
    """Рассчитывает стоимость запроса к GigaChat в рублях (через chat_utils)."""
    return cu_calculate_gigachat_cost(prompt_tokens, completion_tokens, model)


def _validate_history_entry(entry: Dict[str, str]) -> bool:
    """Проверяет валидность записи истории (через chat_utils)."""
    return cu_validate_history_entry(entry)


def _extract_tokens_from_usage(usage: object) -> Tuple[Optional[int], ...]:
    """Извлекает токены из объекта usage (через chat_utils)."""
    return cu_extract_tokens_from_usage(usage)


def _create_meta_dict(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    elapsed: float,
    cost: Optional[float] = None,
) -> Dict[str, Optional[int | float]]:
    """Создает словарь метаданных для ответа (через chat_utils)."""
    return cu_create_meta_dict(prompt_tokens, completion_tokens, total_tokens, elapsed, cost)


def _handle_hf_http_error(http_err: requests.HTTPError) -> None:
    """Обрабатывает HTTP ошибки от Hugging Face API (через chat_utils)."""
    return cu_handle_hf_http_error(http_err)


def _handle_hf_response_error(response: requests.Response) -> None:
    """Обрабатывает ошибки ответа от Hugging Face API (через chat_utils)."""
    return cu_handle_hf_response_error(response)


# ---------------------------------------------------------------------------
# GigaChat interaction moved to gigachat_client.ask_gigachat
# ---------------------------------------------------------------------------


def ask_huggingface(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    temperature: float,
    model: str,
    task: str = "text-generation",
    mode: str = "chat-completion",
) -> Tuple[str, Dict[str, Optional[int | float]]]:
    """Запрашивает ответ от Hugging Face API.

    Returns:
        Tuple[текст ответа, метаданные]
    """
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Не найден HUGGINGFACE_API_TOKEN. "
            "Добавьте ключ авторизации в .env или переменные окружения."
        )

    client = InferenceClient(token=token)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    if mode == "chat-completion":
        return _ask_hf_chat_completion(
            client, system_prompt, history, user_message, temperature, model
        )

    return _ask_hf_text_generation(
        headers, system_prompt, history, user_message, temperature, model, task
    )


def _ask_hf_chat_completion(
    client: InferenceClient,
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    temperature: float,
    model: str,
) -> Tuple[str, Dict[str, Optional[int | float]]]:
    """Запрашивает ответ через Hugging Face chat-completion API."""
    messages = [
        {
            "role": "system",
            "content": system_prompt.strip() or "You are a helpful assistant.",
        }
    ]

    for entry in history:
        if not _validate_history_entry(entry):
            continue
        if entry["role"] in ("user", "assistant"):
            messages.append(
                {"role": entry["role"], "content": entry["content"]}
            )

    messages.append({"role": "user", "content": user_message})

    try:
        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - start
    except requests.HTTPError as http_err:
        _handle_hf_http_error(http_err)

    choices = getattr(completion, "choices", [])
    if not choices:
        raise RuntimeError(f"Пустой ответ от Hugging Face: {completion}")

    message = choices[0].message
    content = getattr(message, "content", "").strip()
    if not content:
        raise RuntimeError(f"Пустой ответ от Hugging Face: {completion}")

    usage = getattr(completion, "usage", None)
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if usage is not None:
        prompt_tokens, completion_tokens, total_tokens = (
            _extract_tokens_from_usage(usage)
        )

    meta = _create_meta_dict(
        prompt_tokens, completion_tokens, total_tokens, elapsed, cost=None
    )

    return content, meta


def _ask_hf_text_generation(
    headers: Dict[str, str],
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    temperature: float,
    model: str,
    task: str,
) -> Tuple[str, Dict[str, Optional[int | float]]]:
    """Запрашивает ответ через Hugging Face text-generation API."""
    lines = [system_prompt.strip()]
    for entry in history:
        if not _validate_history_entry(entry):
            continue
        role = "User" if entry["role"] == "user" else "Assistant"
        lines.append(f"{role}: {entry['content']}")

    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    prompt_text = "\n".join(lines)

    payload = {
        "model": model,
        "task": task,
        "inputs": prompt_text,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": 256,
            "return_full_text": True,
        },
    }

    start = time.perf_counter()
    response = requests.post(
        "https://router.huggingface.co/hf-inference/text-generation",
        headers=headers,
        params={"model": model},
        json=payload,
        timeout=60,
    )
    elapsed = time.perf_counter() - start

    _handle_hf_response_error(response)

    data = response.json()
    generated_text = _extract_text_from_hf_response(data)

    if not generated_text:
        raise RuntimeError(f"Пустой ответ от Hugging Face: {data}")

    if generated_text.startswith(prompt_text):
        generated_text = generated_text[len(prompt_text) :]

    # Приблизительный подсчет токенов для text-generation endpoint
    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(generated_text)
    total_tokens = prompt_tokens + completion_tokens

    meta = _create_meta_dict(
        prompt_tokens, completion_tokens, total_tokens, elapsed, cost=None
    )

    result_text = (
        generated_text.strip()
        or "Не удалось получить содержательный ответ от модели Hugging Face."
    )

    return result_text, meta


def _extract_text_from_hf_response(data: dict | list) -> str:
    """Извлекает текст из ответа Hugging Face API."""
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            return (
                item.get("generated_text")
                or item.get("translation_text")
                or ""
            )
        if isinstance(item, str):
            return item
    if isinstance(data, dict):
        return (
            data.get("generated_text") or data.get("translation_text") or ""
        )
    return ""


# ---------------------------------------------------------------------------
# Flask views
# ---------------------------------------------------------------------------


def _create_chat_state(
    preset_key: str,
    temperature: float,
    provider: str,
    model: str,
    history: List[Dict[str, str]] | None = None,
    use_local_vectors: bool = False,
    rag_threshold: float = 0.7,
    repo_config: Optional[str] = None,
) -> Dict[str, object]:
    """Создает словарь состояния чата."""
    return {
        "preset_key": preset_key,
        "temperature": temperature,
        "provider": provider,
        "model": model,
        "history": history or [],
        "use_local_vectors": use_local_vectors,
        "rag_threshold": rag_threshold,
        "repo_config": repo_config,
    }


def _save_session_state(state: Dict[str, object]) -> None:
    """Сохраняет состояние (только настройки) в cookie-сессию.
    
    История сообщений хранится в БД и из cookie не пишется.
    """
    session_state = {
        "preset_key": state.get("preset_key"),
        "temperature": state.get("temperature"),
        "provider": state.get("provider"),
        "model": state.get("model"),
        "use_local_vectors": state.get("use_local_vectors", False),
        "rag_threshold": state.get("rag_threshold", 0.7),
        "repo_config": state.get("repo_config"),
    }
    
    try:
        session[SESSION_KEY] = session_state
        session.modified = True
    except Exception as e:
        # Если произошла ошибка при сохранении
        flash(
            "⚠️ Не удалось сохранить настройки сессии.",
            "warning"
        )


def _load_session_state() -> Dict[str, object]:
    """Загружает состояние из cookie-сессии и историю из БД."""
    session_id = _get_session_id()
    
    # Загружаем настройки из сессии
    session_state = session.get(SESSION_KEY, {})
    
    # Загружаем историю из БД
    history = _db_get_messages(session_id)
    
    # Объединяем настройки и историю
    # Если настроек нет в сессии, возвращаем только историю (настройки будут установлены через _validate_chat_state)
    if not session_state:
        return {"history": history}
    
    state = {
        "preset_key": session_state.get("preset_key"),
        "temperature": session_state.get("temperature"),
        "provider": session_state.get("provider"),
        "model": session_state.get("model"),
        "use_local_vectors": session_state.get("use_local_vectors", False),
        "rag_threshold": session_state.get("rag_threshold", 0.7),
        "repo_config": session_state.get("repo_config"),
        "history": history,
    }
    
    # MD файл конфигурации теперь используется только для указания репозитория
    # Настройки RAG управляются только через UI
    return state


def _validate_chat_state(
    state: Dict[str, object] | None,
    presets: OrderedDict[str, Dict[str, str]],
) -> Dict[str, object]:
    """Валидирует и возвращает корректное состояние чата."""
    if (
        not state
        or state.get("preset_key") not in presets
        or state.get("provider") not in AVAILABLE_PROVIDERS
        or state.get("model")
        not in AVAILABLE_PROVIDERS.get(state.get("provider", ""), {}).get(
            "models", {}
        )
    ):
        return default_chat_state(presets)

    state.pop("custom_prompt", None)
    return state


def _parse_form_settings(
    state: Dict[str, object], presets: OrderedDict[str, Dict[str, str]]
) -> Tuple[str, float, str, str, bool, float, Optional[str]]:
    """Парсит настройки из формы запроса."""
    preset_key = request.form.get("preset") or state["preset_key"]
    if preset_key not in presets:
        preset_key = next(iter(presets))

    temperature = parse_temperature(
        request.form.get("temperature"),
        state.get("temperature", DEFAULT_TEMPERATURE),
    )

    provider = request.form.get("provider") or state.get(
        "provider", DEFAULT_PROVIDER
    )
    if provider not in AVAILABLE_PROVIDERS:
        provider = DEFAULT_PROVIDER

    provider_models = AVAILABLE_PROVIDERS[provider]["models"]
    model = request.form.get("model") or state.get("model")
    if model not in provider_models:
        model = default_model_for(provider)

    # Парсим флаг использования локальных векторов
    use_local_vectors = request.form.get("use_local_vectors") == "on"

    # Парсим порог релевантности для RAG
    rag_threshold_str = request.form.get("rag_threshold")
    if rag_threshold_str:
        try:
            rag_threshold = float(rag_threshold_str)
            rag_threshold = max(0.0, min(1.0, rag_threshold))  # Ограничиваем 0.0-1.0
        except (ValueError, TypeError):
            rag_threshold = state.get("rag_threshold", 0.7)
    else:
        rag_threshold = state.get("rag_threshold", 0.7)

    # Парсим выбранный MD файл конфигурации репозитория
    repo_config = request.form.get("repo_config") or state.get("repo_config")
    if repo_config == "":
        repo_config = None

    return preset_key, temperature, provider, model, use_local_vectors, rag_threshold, repo_config


def _handle_save_preset(
    preset_key: str,
    temperature: float,
    provider: str,
    model: str,
) -> Tuple[bool, Dict[str, object] | None]:
    """Обрабатывает сохранение предустановки.

    Returns:
        Tuple[успешно ли сохранено, новое состояние или None]
    """
    preset_title = request.form.get("preset_title", "").strip()
    preset_prompt_text = request.form.get("preset_prompt", "").strip()

    if not preset_title:
        flash("Укажите название предустановки.", "warning")
        return False, None

    if not preset_prompt_text:
        flash("Введите текст промпта для сохранения.", "warning")
        return False, None

    # Генерируем уникальный ключ (slug) относительно существующих ключей в БД
    base_slug = slugify(preset_title)
    slug = make_unique_preset_key(base_slug)

    # Сохраняем новую предустановку в БД
    _db_upsert_preset(slug, preset_title, preset_prompt_text)

    flash("Предустановка сохранена.", "info")
    new_state = _create_chat_state(slug, temperature, provider, model)
    return True, new_state


def _check_settings_changed(
    state: Dict[str, object],
    preset_key: str,
    provider: str,
    temperature: float,
    model: str,
) -> bool:
    """Проверяет, изменились ли настройки."""
    return (
        state["preset_key"] != preset_key
        or state.get("provider", DEFAULT_PROVIDER) != provider
        or abs(state.get("temperature", DEFAULT_TEMPERATURE) - temperature)
        > 1e-6
        or state.get("model") != model
    )


@app.route("/", methods=["GET", "POST"])
def index():
    """Главная страница приложения."""
    presets = load_presets()
    # Загружаем состояние из сессии и истории из файла
    state = _load_session_state()
    state = _validate_chat_state(state, presets)

    if request.method == "POST":
        action = request.form.get("action", "send")
        preset_key, temperature, provider, model, use_local_vectors, rag_threshold, repo_config = _parse_form_settings(
            state, presets
        )
        
        logger.info(f"Настройки после парсинга формы: use_local_vectors={use_local_vectors}, rag_threshold={rag_threshold}, repo_config={repo_config}")
        
        # Получаем репозиторий из MD файла конфигурации, если он выбран
        repository = None
        if repo_config:
            config = get_repo_config(repo_config)
            if config:
                logger.info(f"Загружена конфигурация {repo_config}: repository={config.get('repository')}")
                # Получаем репозиторий из конфигурации
                repository = config.get("repository")
                logger.info(f"Репозиторий из конфигурации {repo_config}: {repository}")
            else:
                logger.warning(f"Конфигурация {repo_config} не найдена")
        else:
            logger.info("Конфигурация репозитория не выбрана")

        if action == "save_preset":
            success, new_state = _handle_save_preset(
                preset_key, temperature, provider, model
            )
            if success:
                _save_session_state(new_state)
                return redirect(url_for("index"))

            # В случае ошибок продолжаем с обновленными параметрами
            state.update(
                {
                    "preset_key": preset_key,
                    "temperature": temperature,
                    "provider": provider,
                    "model": model,
                    "use_local_vectors": use_local_vectors,
                    "rag_threshold": rag_threshold,
                }
            )
            _save_session_state(state)
            return redirect(url_for("index"))

        if action == "reset":
            # Начинаем новый диалог (не удаляя историю предыдущего)
            _start_new_session()
            state = _create_chat_state(preset_key, temperature, provider, model, use_local_vectors=use_local_vectors, rag_threshold=rag_threshold, repo_config=repo_config)
            _save_session_state(state)
            flash("Начат новый диалог.", "info")
            return redirect(url_for("index"))

        message = request.form.get("message", "").strip()
        if not message:
            flash("Введите сообщение перед отправкой.", "warning")
            state.update(
                {
                    "preset_key": preset_key,
                    "temperature": temperature,
                    "provider": provider,
                    "model": model,
                    "use_local_vectors": use_local_vectors,
                    "rag_threshold": rag_threshold,
                    "repo_config": repo_config,
                }
            )
            _save_session_state(state)
            return redirect(url_for("index"))
        
        # Обработка команды /help
        if message.startswith("/help"):
            # Получаем system_prompt из presets
            system_prompt = presets[preset_key]["prompt"]
            
            # Получаем репозиторий из конфигурации MD файла
            repository = None
            if repo_config:
                config = get_repo_config(repo_config)
                if config:
                    repository = config.get("repository")
            
            # Если репозиторий указан в конфигурации, получаем структуру из GitHub
            repo_structure_text = ""
            if repository:
                try:
                    # Парсим owner/repo
                    if "/" in repository:
                        owner, repo = repository.split("/", 1)
                        # Получаем структуру репозитория из GitHub
                        tree_result = github_get_repo_tree(owner, repo, recursive=True)
                        
                        if "error" in tree_result:
                            repo_structure_text = f"Ошибка при получении структуры репозитория {repository}: {tree_result.get('error', 'Неизвестная ошибка')}\n\n"
                        else:
                            # Форматируем структуру репозитория
                            tree_items = tree_result.get("tree", [])
                            branch = tree_result.get("branch", "main")
                            
                            # Строим дерево из путей
                            tree_dict = {}
                            for item in tree_items:
                                path = item.get("path", "")
                                item_type = item.get("type", "blob")
                                if path:
                                    parts = path.split('/')
                                    current = tree_dict
                                    for idx, part in enumerate(parts):
                                        if part not in current:
                                            # Если это последний элемент пути и это файл, то это файл
                                            # Иначе это директория
                                            if idx == len(parts) - 1 and item_type == "blob":
                                                current[part] = {"type": "blob", "children": {}}
                                            else:
                                                current[part] = {"type": "tree", "children": {}}
                                        current = current[part]["children"]
                            
                            # Форматируем дерево в строку
                            def format_tree_structure(node: dict, prefix: str = "", is_last: bool = True) -> str:
                                lines = []
                                items = sorted(node.items())
                                
                                for i, (name, data) in enumerate(items):
                                    is_last_item = i == len(items) - 1
                                    item_type = data.get("type", "blob")
                                    type_marker = "/" if item_type == "tree" else ""
                                    current_prefix = prefix + ("└── " if is_last_item else "├── ")
                                    lines.append(current_prefix + name + type_marker)
                                    
                                    children = data.get("children", {})
                                    if children:
                                        next_prefix = prefix + ("    " if is_last_item else "│   ")
                                        lines.append(format_tree_structure(children, next_prefix, is_last_item))
                                
                                return "\n".join(lines)
                            
                            structure_tree = format_tree_structure(tree_dict)
                            repo_structure_text = f"Структура репозитория GitHub: {repository}\nВетка: {branch}\nВсего файлов: {len(tree_items)}\n\n{structure_tree}\n"
                    else:
                        repo_structure_text = f"Некорректный формат репозитория: {repository}. Ожидается формат owner/repo\n\n"
                except Exception as e:
                    repo_structure_text = f"Ошибка при получении структуры репозитория {repository}: {str(e)}\n\n"
            
            # Если репозиторий не указан, используем локальный Git
            if not repository or not repo_structure_text:
                git_structure = get_git_repo_structure()
                repo_structure_text = git_structure
            
            # Формируем запрос для GigaChat
            help_prompt = f"Пользователь запросил структуру проекта командой /help.\n\nПолучена следующая структура проекта:\n\n```\n{repo_structure_text}\n```\n\nПредставь эту структуру проекта в удобном и понятном формате. Опиши основные директории и файлы, их назначение (если возможно определить по названиям). Структурируй информацию так, чтобы пользователю было легко понять организацию проекта."
            
            # Отправляем запрос в GigaChat
            try:
                if provider == "gigachat":
                    assistant_text, assistant_meta = ask_gigachat(
                        system_prompt=system_prompt,
                        history=state["history"],
                        user_message=help_prompt,
                        temperature=temperature,
                        model=model,
                        use_local_vectors=False,  # Отключаем RAG для команды /help
                        rag_threshold=rag_threshold,
                        repository=repository,
                    )
                else:
                    # Для других провайдеров просто возвращаем структуру
                    assistant_text = f"```\n{repo_structure_text}\n```"
                    assistant_meta = {
                        "provider": provider,
                        "model": model,
                        "elapsed": 0.0,
                        "total_tokens": 0,
                    }
            except Exception as exc:
                # В случае ошибки возвращаем структуру напрямую
                assistant_text = f"```\n{repo_structure_text}\n```"
                assistant_meta = {
                    "provider": provider,
                    "model": model,
                    "elapsed": 0.0,
                    "total_tokens": 0,
                }
                flash(f"Не удалось обработать через GigaChat: {exc}", "warning")
            
            # Добавляем ответ в историю
            user_meta = {
                "tokens": estimate_tokens(message),
                "request_tokens": estimate_request_tokens(system_prompt, state["history"], help_prompt),
            }
            state["history"].append(
                {"role": "user", "content": message, "meta": user_meta}
            )
            
            assistant_meta["provider"] = provider
            assistant_meta["model"] = model
            state["history"].append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "meta": assistant_meta,
                }
            )
            
            # Сохраняем в БД
            session_id = _get_session_id()
            _db_ensure_session(session_id, title="Структура проекта")
            _db_add_message(session_id, "user", message, user_meta)
            _db_add_message(session_id, "assistant", assistant_text, assistant_meta)
            
            state.update(
                {
                    "preset_key": preset_key,
                    "temperature": temperature,
                    "provider": provider,
                    "model": model,
                    "use_local_vectors": use_local_vectors,
                    "rag_threshold": rag_threshold,
                    "repo_config": repo_config,
                }
            )
            _save_session_state(state)
            return redirect(url_for("index"))

        settings_changed = (
            _check_settings_changed(state, preset_key, provider, temperature, model)
            or state.get("use_local_vectors", False) != use_local_vectors
            or abs(state.get("rag_threshold", 0.7) - rag_threshold) > 1e-6
            or state.get("repo_config") != repo_config
        )

        if settings_changed:
            state = _create_chat_state(
                preset_key, temperature, provider, model, use_local_vectors=use_local_vectors, rag_threshold=rag_threshold, repo_config=repo_config
            )
        else:
            state.update(
                {
                    "preset_key": preset_key,
                    "temperature": temperature,
                    "provider": provider,
                    "model": model,
                    "use_local_vectors": use_local_vectors,
                    "rag_threshold": rag_threshold,
                    "repo_config": repo_config,
                }
            )

        system_prompt = presets[preset_key]["prompt"]
        user_message_tokens = estimate_tokens(message)
        total_request_tokens = estimate_request_tokens(
            system_prompt, state["history"], message
        )

        try:
            if provider == "gigachat":
                # Финальная проверка настроек перед отправкой
                logger.info("=" * 80)
                logger.info("ФИНАЛЬНЫЕ НАСТРОЙКИ ПЕРЕД ОТПРАВКОЙ В GIGACHAT:")
                logger.info(f"  use_local_vectors: {use_local_vectors} (тип: {type(use_local_vectors)})")
                logger.info(f"  rag_threshold: {rag_threshold}")
                logger.info(f"  repository: {repository}")
                logger.info(f"  repo_config: {repo_config}")
                logger.info("=" * 80)
                
                assistant_text, assistant_meta = ask_gigachat(
                    system_prompt=system_prompt,
                    history=state["history"],
                    user_message=message,
                    temperature=temperature,
                    model=model,
                    use_local_vectors=use_local_vectors,
                    rag_threshold=rag_threshold,
                    repository=repository,
                )
            else:
                model_meta = AVAILABLE_PROVIDERS[provider]["models"].get(
                    model, {}
                )
                task = model_meta.get("task", "text-generation")
                mode = model_meta.get("mode", "text-generation")
                assistant_text, assistant_meta = ask_huggingface(
                    system_prompt=system_prompt,
                    history=state["history"],
                    user_message=message,
                    temperature=temperature,
                    model=model,
                    task=task,
                    mode=mode,
                )
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            # Улучшаем сообщения об ошибках
            if "handshake" in error_msg.lower() or "timeout" in error_msg.lower() or "ssl" in error_msg.lower():
                flash(
                    f"Ошибка подключения к GigaChat API: таймаут SSL соединения. "
                    f"Попробуйте повторить запрос через несколько секунд. "
                    f"Ошибка: {error_msg}",
                    "error"
                )
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                flash(
                    f"Ошибка сетевого подключения к GigaChat API. "
                    f"Проверьте интернет-соединение и попробуйте снова. "
                    f"Ошибка: {error_msg}",
                    "error"
                )
            else:
                flash(f"Не удалось получить ответ: {error_msg}", "error")
            
            logger.error(f"Ошибка при запросе к GigaChat: {exc}", exc_info=True)
            _save_session_state(state)
            return redirect(url_for("index"))

        # Сохраняем сообщение пользователя с метаданными о токенах
        user_meta = {
            "tokens": user_message_tokens,
            "request_tokens": total_request_tokens,
        }
        state["history"].append(
            {"role": "user", "content": message, "meta": user_meta}
        )

        # Сохраняем ответ ассистента с полными метаданными
        assistant_meta["provider"] = provider
        assistant_meta["model"] = model
        state["history"].append(
            {
                "role": "assistant",
                "content": assistant_text,
                "meta": assistant_meta,
            }
        )

        # Сохраняем в БД: создаем сессию (заголовок = первое пользовательское сообщение)
        session_id = _get_session_id()
        first_user_title = None
        # Если в БД еще нет сессии — заголовок берем по первому USER сообщению истории
        # В качестве простого правила возьмем первые 64 символа
        for entry in state["history"]:
            if entry.get("role") == "user":
                first_user_title = (entry.get("content") or "").strip()
                break
        title_trimmed = (first_user_title or "").replace("\n", " ")[:64]
        _db_ensure_session(session_id, title=title_trimmed if title_trimmed else None)
        # Добавляем два новых сообщения
        _db_add_message(session_id, "user", message, user_meta)
        _db_add_message(session_id, "assistant", assistant_text, assistant_meta)

        _save_session_state(state)
        return redirect(url_for("index"))

    _save_session_state(state)
    # Загружаем список доступных конфигураций репозиториев
    repo_configs = list_repo_configs()
    return render_template(
        "index.html",
        presets=presets,
        providers=AVAILABLE_PROVIDERS,
        state=state,
        history=state["history"],
        models=AVAILABLE_PROVIDERS[state["provider"]]["models"],
        repo_configs=repo_configs,
    )


@app.route("/history", methods=["GET"])
def history_list():
    """Список сохраненных сессий из БД."""
    sessions = _db_list_sessions(limit=200)
    return render_template("history.html", sessions=sessions)


@app.route("/history/<session_uuid>", methods=["GET"])
def history_view(session_uuid: str):
    """Просмотр конкретной истории диалога."""
    messages = _db_get_messages(session_uuid)
    # Для удобства отображения используем тот же шаблон чата, но в read-only режиме
    presets = load_presets()
    state = default_chat_state(presets)
    state["history"] = messages
    repo_configs = list_repo_configs()
    return render_template(
        "index.html",
        presets=presets,
        providers=AVAILABLE_PROVIDERS,
        state=state,
        history=messages,
        models=AVAILABLE_PROVIDERS[state["provider"]]["models"],
        repo_configs=repo_configs,
    )


@app.errorhandler(500)
def internal_error(error):
    """Обработчик внутренних ошибок сервера."""
    traceback.print_exc()
    return f"Internal Server Error: {error}", 500


def create_app() -> Flask:
    """Фабрика приложения Flask."""
    return app


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5050)),
        debug=os.getenv("FLASK_DEBUG") == "1",
    )
