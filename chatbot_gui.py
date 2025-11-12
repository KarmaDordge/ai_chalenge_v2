"""
Веб-чат с GigaChat и Hugging Face на Flask.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from flask import (
    Flask,
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

# Настраиваем переменные окружения
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")
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
# Функции для работы с файловым хранилищем истории
# ---------------------------------------------------------------------------


def _get_session_id() -> str:
    """Получает или создает ID сессии для файлового хранилища."""
    if SESSION_ID_KEY not in session:
        import uuid
        session[SESSION_ID_KEY] = str(uuid.uuid4())
        session.modified = True
    return session[SESSION_ID_KEY]


def _get_history_file_path(session_id: str) -> str:
    """Возвращает путь к файлу истории для данной сессии."""
    return os.path.join(HISTORY_DIR, f"{session_id}.json")


def _load_history_from_file(session_id: str) -> List[Dict[str, str]]:
    """Загружает историю чата из файла."""
    file_path = _get_history_file_path(session_id)
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []


def _save_history_to_file(session_id: str, history: List[Dict[str, str]]) -> None:
    """Сохраняет историю чата в файл."""
    file_path = _get_history_file_path(session_id)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Если не удалось сохранить, логируем ошибку, но не прерываем работу
        import logging
        logging.error(f"Failed to save history to file: {e}")


def _clear_history_file(session_id: str) -> None:
    """Удаляет файл истории для данной сессии."""
    file_path = _get_history_file_path(session_id)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-me")


# ---------------------------------------------------------------------------
# Утилиты для загрузки и подготовки промптов
# ---------------------------------------------------------------------------


def _load_json_mapping(file_path: str) -> OrderedDict[str, str]:
    """Загружает JSON-файл и возвращает OrderedDict."""
    if not os.path.exists(file_path):
        return OrderedDict()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            if isinstance(data, dict):
                return OrderedDict(data)
    except Exception:
        pass
    return OrderedDict()


def load_raw_preset_files() -> Tuple[OrderedDict[str, str], OrderedDict[str, str]]:
    """Загружает файлы с промптами и названиями предустановок."""
    prompts = _load_json_mapping(PROMPTS_FILE_PATH)
    names = _load_json_mapping(PROMPT_NAMES_FILE_PATH)

    if not prompts:
        prompts = OrderedDict(DEFAULT_PRESET_PROMPTS)
    if not names:
        names = OrderedDict(DEFAULT_PRESET_NAMES)

    return prompts, names


def load_presets() -> OrderedDict[str, Dict[str, str]]:
    """Загружает и объединяет предустановки из файлов."""
    prompts, names = load_raw_preset_files()
    presets = OrderedDict()

    for key, title in names.items():
        if key in prompts:
            presets[key] = {"name": title, "prompt": prompts[key]}

    for key, prompt in prompts.items():
        if key not in presets:
            presets[key] = {"name": key, "prompt": prompt}

    if not presets:
        presets[DEFAULT_PRESET_KEY] = {
            "name": DEFAULT_PRESET_NAMES[DEFAULT_PRESET_KEY],
            "prompt": DEFAULT_PRESET_PROMPTS[DEFAULT_PRESET_KEY],
        }

    return presets


def write_presets(
    prompts: OrderedDict[str, str], names: OrderedDict[str, str]
) -> None:
    """Сохраняет промпты и названия в файлы."""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(PROMPTS_FILE_PATH, "w", encoding="utf-8") as prompts_file:
        json.dump(prompts, prompts_file, ensure_ascii=False, indent=2)
    with open(PROMPT_NAMES_FILE_PATH, "w", encoding="utf-8") as names_file:
        json.dump(names, names_file, ensure_ascii=False, indent=2)


def slugify(title: str) -> str:
    """Преобразует строку в slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    return slug or "preset"


def default_model_for(provider: str) -> str:
    """Возвращает модель по умолчанию для провайдера."""
    models = AVAILABLE_PROVIDERS[provider]["models"]
    return next(iter(models))


def default_chat_state(
    presets: OrderedDict[str, Dict[str, str]]
) -> Dict[str, object]:
    """Создает начальное состояние чата."""
    preset_key = (
        DEFAULT_PRESET_KEY
        if DEFAULT_PRESET_KEY in presets
        else next(iter(presets))
    )
    provider_key = (
        DEFAULT_PROVIDER
        if DEFAULT_PROVIDER in AVAILABLE_PROVIDERS
        else next(iter(AVAILABLE_PROVIDERS))
    )
    return {
        "preset_key": preset_key,
        "temperature": DEFAULT_TEMPERATURE,
        "provider": provider_key,
        "model": default_model_for(provider_key),
        "history": [],
    }


def parse_temperature(raw_value: str | None, fallback: float) -> float:
    """Парсит температуру из строки с валидацией."""
    if raw_value is None:
        return fallback
    try:
        temp = float(raw_value)
    except ValueError:
        return fallback
    temp = max(0.0, min(2.0, round(temp, 2)))
    return temp


def estimate_tokens(text: str) -> int:
    """Приблизительный подсчет токенов: ~1 токен = 4 символа."""
    return max(1, len(text) // 4)


def estimate_request_tokens(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
) -> int:
    """Подсчитывает общее количество токенов в запросе.
    
    Включает системный промпт, историю диалога и текущее сообщение пользователя.
    """
    total = estimate_tokens(system_prompt)
    
    for entry in history:
        if not _validate_history_entry(entry):
            continue
        total += estimate_tokens(entry["content"])
        # Добавляем небольшой штраф за роль (приблизительно 2 токена на сообщение)
        total += 2
    
    total += estimate_tokens(user_message)
    # Добавляем штраф за роль текущего сообщения пользователя
    total += 2
    
    return total


def calculate_gigachat_cost(
    prompt_tokens: int, completion_tokens: int, model: str
) -> Optional[float]:
    """Рассчитывает стоимость запроса к GigaChat в рублях.

    Примерные цены (на 1000 токенов):
    - GigaChat Lite: вход 0.01 руб, выход 0.03 руб
    - GigaChat Pro: вход 0.05 руб, выход 0.15 руб
    - GigaChat Pro Max: вход 0.10 руб, выход 0.30 руб
    """
    pricing = {
        "GigaChat": {"input": 0.01, "output": 0.03},
        "GigaChat-Pro": {"input": 0.05, "output": 0.15},
        "GigaChat-Pro-Max": {"input": 0.10, "output": 0.30},
    }

    model_pricing = pricing.get(model, pricing["GigaChat"])
    input_cost = (prompt_tokens / 1000.0) * model_pricing["input"]
    output_cost = (completion_tokens / 1000.0) * model_pricing["output"]
    return round(input_cost + output_cost, 6)


def _validate_history_entry(entry: Dict[str, str]) -> bool:
    """Проверяет валидность записи истории."""
    return (
        isinstance(entry, dict)
        and "role" in entry
        and "content" in entry
    )


def _extract_tokens_from_usage(usage: object) -> Tuple[Optional[int], ...]:
    """Извлекает токены из объекта usage.

    Returns:
        Tuple[prompt_tokens, completion_tokens, total_tokens]
    """
    # Поддерживаем как объекты с атрибутами, так и словари
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens") or usage.get("tokens")
    else:
        # Пробуем получить как атрибуты объекта
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(
            usage, "input_tokens", None
        )
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(
            usage, "output_tokens", None
        )
        total_tokens = getattr(usage, "total_tokens", None) or getattr(
            usage, "tokens", None
        )
        
        # Если это объект с методом dict() или __dict__, пробуем получить через него
        if total_tokens is None:
            try:
                # Пробуем метод dict() (для Pydantic моделей)
                if hasattr(usage, "dict"):
                    usage_dict = usage.dict()
                    if isinstance(usage_dict, dict):
                        total_tokens = (
                            usage_dict.get("total_tokens")
                            or usage_dict.get("tokens")
                            or usage_dict.get("total")
                        )
                # Пробуем __dict__ атрибут
                elif hasattr(usage, "__dict__"):
                    usage_dict = usage.__dict__
                    if isinstance(usage_dict, dict):
                        total_tokens = (
                            usage_dict.get("total_tokens")
                            or usage_dict.get("tokens")
                            or usage_dict.get("total")
                        )
                # Пробуем получить все атрибуты через dir()
                else:
                    attrs = [attr for attr in dir(usage) if not attr.startswith("_")]
                    for attr in ["total_tokens", "tokens", "total"]:
                        if attr in attrs:
                            try:
                                value = getattr(usage, attr)
                                if value is not None:
                                    total_tokens = value
                                    break
                            except Exception:
                                pass
            except Exception:
                pass

    # Если total_tokens не найден или равен 0, но есть prompt_tokens и completion_tokens, вычисляем
    # Примечание: GigaChat может возвращать total_tokens, который учитывает кэширование,
    # поэтому если total_tokens не найден, используем сумму prompt_tokens + completion_tokens
    if (total_tokens is None or total_tokens == 0) and prompt_tokens is not None and completion_tokens is not None:
        # Вычисляем total_tokens только если он не был найден или равен 0
        # Это может быть не совсем точно из-за кэширования, но лучше чем ничего
        calculated_total = prompt_tokens + completion_tokens
        if total_tokens is None:
            total_tokens = calculated_total
        elif total_tokens == 0:
            # Если total_tokens равен 0, но есть prompt_tokens и completion_tokens,
            # используем вычисленное значение
            total_tokens = calculated_total

    # Приводим к int
    try:
        if prompt_tokens is not None:
            prompt_tokens = int(prompt_tokens)
        if completion_tokens is not None:
            completion_tokens = int(completion_tokens)
        if total_tokens is not None:
            total_tokens = int(total_tokens)
    except (TypeError, ValueError):
        pass

    return prompt_tokens, completion_tokens, total_tokens


def _create_meta_dict(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    elapsed: float,
    cost: Optional[float] = None,
) -> Dict[str, Optional[int | float]]:
    """Создает словарь метаданных для ответа."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "cost": cost,
    }


def _handle_hf_http_error(http_err: requests.HTTPError) -> None:
    """Обрабатывает HTTP ошибки от Hugging Face API."""
    response = http_err.response
    status = response.status_code if response else "?"
    detail = response.text if response else str(http_err)

    if status == 404:
        raise RuntimeError(
            "Модель недоступна через Hugging Face Inference API. "
            "Проверьте, опубликована ли она для Inference и не приватна."
        ) from http_err
    if status == 429:
        raise RuntimeError(
            "Превышен лимит запросов к Hugging Face Inference API."
        ) from http_err
    raise RuntimeError(
        f"Hugging Face вернул ошибку {status}: {detail}"
    ) from http_err


def _handle_hf_response_error(response: requests.Response) -> None:
    """Обрабатывает ошибки ответа от Hugging Face API."""
    if response.status_code == 404:
        raise RuntimeError(
            "Модель недоступна через Hugging Face Inference API. "
            "Проверьте, опубликована ли она для Inference и не приватна."
        )
    if response.status_code == 429:
        raise RuntimeError(
            "Превышен лимит запросов к Hugging Face Inference API."
        )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Hugging Face вернул ошибку {response.status_code}: "
            f"{response.text}"
        )


# ---------------------------------------------------------------------------
# GigaChat interaction
# ---------------------------------------------------------------------------


def ask_gigachat(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    temperature: float,
    model: str | None = None,
) -> Tuple[str, Dict[str, Optional[int | float]]]:
    """Запрашивает ответ от GigaChat API.

    Returns:
        Tuple[текст ответа, метаданные]
    """
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise RuntimeError(
            "Не найден GIGACHAT_CREDENTIALS. "
            "Добавьте ключ авторизации в .env или переменные окружения."
        )

    if not model:
        model = default_model_for("gigachat")

    messages: List[Messages] = [
        Messages(role=MessagesRole.SYSTEM, content=system_prompt),
    ]

    for entry in history:
        if not _validate_history_entry(entry):
            continue
        role = (
            MessagesRole.USER
            if entry["role"] == "user"
            else MessagesRole.ASSISTANT
        )
        messages.append(Messages(role=role, content=entry["content"]))

    messages.append(Messages(role=MessagesRole.USER, content=user_message))

    chat = Chat(
        messages=messages,
        model=model,
        temperature=temperature,
        flags=["no_cache"],
    )

    with GigaChat(credentials=credentials, verify_ssl_certs=False) as client:
        start = time.perf_counter()
        response = client.chat(chat)
        elapsed = time.perf_counter() - start

    usage = getattr(response, "usage", None)
    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    if usage is not None:
        prompt_tokens, completion_tokens, total_tokens = (
            _extract_tokens_from_usage(usage)
        )
        
        # Отладочная информация (можно удалить после проверки)
        # print(f"DEBUG: usage type: {type(usage)}")
        # print(f"DEBUG: usage dir: {dir(usage) if hasattr(usage, '__dict__') else 'N/A'}")
        # print(f"DEBUG: Extracted - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")

    cost = None
    if prompt_tokens is not None and completion_tokens is not None:
        cost = calculate_gigachat_cost(prompt_tokens, completion_tokens, model)

    meta = _create_meta_dict(
        prompt_tokens, completion_tokens, total_tokens, elapsed, cost
    )

    return response.choices[0].message.content, meta


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
) -> Dict[str, object]:
    """Создает словарь состояния чата."""
    return {
        "preset_key": preset_key,
        "temperature": temperature,
        "provider": provider,
        "model": model,
        "history": history or [],
    }


def _save_session_state(state: Dict[str, object]) -> None:
    """Сохраняет состояние в сессию и историю в файл."""
    # Получаем ID сессии
    session_id = _get_session_id()
    
    # Извлекаем историю из состояния
    history = state.get("history", [])
    
    # Сохраняем историю в файл
    _save_history_to_file(session_id, history)
    
    # Сохраняем в сессию только настройки (без истории)
    # Это значительно уменьшает размер cookie
    session_state = {
        "preset_key": state.get("preset_key"),
        "temperature": state.get("temperature"),
        "provider": state.get("provider"),
        "model": state.get("model"),
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
    """Загружает состояние из сессии и историю из файла."""
    session_id = _get_session_id()
    
    # Загружаем настройки из сессии
    session_state = session.get(SESSION_KEY, {})
    
    # Загружаем историю из файла
    history = _load_history_from_file(session_id)
    
    # Объединяем настройки и историю
    # Если настроек нет в сессии, возвращаем только историю (настройки будут установлены через _validate_chat_state)
    if not session_state:
        return {"history": history}
    
    state = {
        "preset_key": session_state.get("preset_key"),
        "temperature": session_state.get("temperature"),
        "provider": session_state.get("provider"),
        "model": session_state.get("model"),
        "history": history,
    }
    
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
) -> Tuple[str, float, str, str]:
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

    return preset_key, temperature, provider, model


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

    raw_prompts, raw_names = load_raw_preset_files()
    base_slug = slugify(preset_title)
    slug = base_slug
    counter = 2

    while slug in raw_prompts or slug in raw_names:
        slug = f"{base_slug}-{counter}"
        counter += 1

    raw_prompts[slug] = preset_prompt_text
    raw_names[slug] = preset_title
    write_presets(raw_prompts, raw_names)

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
        preset_key, temperature, provider, model = _parse_form_settings(
            state, presets
        )

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
                }
            )
            _save_session_state(state)
            return redirect(url_for("index"))

        if action == "reset":
            # Очищаем файл истории
            session_id = _get_session_id()
            _clear_history_file(session_id)
            
            state = _create_chat_state(
                preset_key, temperature, provider, model
            )
            _save_session_state(state)
            flash("Диалог сброшен.", "info")
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
                }
            )
            _save_session_state(state)
            return redirect(url_for("index"))

        settings_changed = _check_settings_changed(
            state, preset_key, provider, temperature, model
        )

        if settings_changed:
            state = _create_chat_state(
                preset_key, temperature, provider, model
            )
        else:
            state.update(
                {
                    "preset_key": preset_key,
                    "temperature": temperature,
                    "provider": provider,
                    "model": model,
                }
            )

        system_prompt = presets[preset_key]["prompt"]
        user_message_tokens = estimate_tokens(message)
        total_request_tokens = estimate_request_tokens(
            system_prompt, state["history"], message
        )

        try:
            if provider == "gigachat":
                assistant_text, assistant_meta = ask_gigachat(
                    system_prompt=system_prompt,
                    history=state["history"],
                    user_message=message,
                    temperature=temperature,
                    model=model,
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
            flash(f"Не удалось получить ответ: {exc}", "error")
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

        _save_session_state(state)
        return redirect(url_for("index"))

    _save_session_state(state)
    return render_template(
        "index.html",
        presets=presets,
        providers=AVAILABLE_PROVIDERS,
        state=state,
        history=state["history"],
        models=AVAILABLE_PROVIDERS[state["provider"]]["models"],
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
