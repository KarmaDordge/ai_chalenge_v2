from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from chat_utils import (
    validate_history_entry as cu_validate_history_entry,
    extract_tokens_from_usage as cu_extract_tokens_from_usage,
    create_meta_dict as cu_create_meta_dict,
    calculate_gigachat_cost as cu_calculate_gigachat_cost,
)


def ask_gigachat(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    temperature: float,
    model: str | None = None,
) -> Tuple[str, Dict[str, Optional[int | float]]]:
    """Взаимодействие с GigaChat API: формирование сообщений, запрос, метаданные."""
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise RuntimeError(
            "Не найден GIGACHAT_CREDENTIALS. "
            "Добавьте ключ авторизации в .env или переменные окружения."
        )

    if not model:
        # Избегаем зависимости от AVAILABLE_PROVIDERS; берем безопасное значение по умолчанию
        model = "GigaChat"

    messages: List[Messages] = [
        Messages(role=MessagesRole.SYSTEM, content=system_prompt),
    ]

    for entry in history:
        if not cu_validate_history_entry(entry):
            continue
        role = MessagesRole.USER if entry["role"] == "user" else MessagesRole.ASSISTANT
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
        prompt_tokens, completion_tokens, total_tokens = cu_extract_tokens_from_usage(usage)

    cost = None
    if prompt_tokens is not None and completion_tokens is not None:
        cost = cu_calculate_gigachat_cost(prompt_tokens, completion_tokens, model)

    meta = cu_create_meta_dict(
        prompt_tokens, completion_tokens, total_tokens, elapsed, cost
    )

    return response.choices[0].message.content, meta


