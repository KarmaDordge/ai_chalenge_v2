from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import requests


def slugify(title: str) -> str:
    """Convert string to URL-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    return slug or "preset"


def default_model_for(available_providers: Dict[str, Dict[str, object]], provider: str) -> str:
    """Return default model id for provider."""
    models = available_providers[provider]["models"]  # type: ignore[index]
    return next(iter(models))  # type: ignore[arg-type]


def default_chat_state(
    presets: OrderedDict[str, Dict[str, str]],
    available_providers: Dict[str, Dict[str, object]],
    default_preset_key: str,
    default_provider: str,
    default_temperature: float,
) -> Dict[str, object]:
    """Build initial chat state."""
    preset_key = default_preset_key if default_preset_key in presets else next(iter(presets))
    provider_key = (
        default_provider if default_provider in available_providers else next(iter(available_providers))
    )
    return {
        "preset_key": preset_key,
        "temperature": default_temperature,
        "provider": provider_key,
        "model": default_model_for(available_providers, provider_key),
        "history": [],
    }


def parse_temperature(raw_value: str | None, fallback: float) -> float:
    """Parse temperature value with validation."""
    if raw_value is None:
        return fallback
    try:
        temp = float(raw_value)
    except ValueError:
        return fallback
    temp = max(0.0, min(2.0, round(temp, 2)))
    return temp


# -------------------------
# Token and cost utilities
# -------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~1 token = 4 chars."""
    return max(1, len(text) // 4)


def validate_history_entry(entry: Dict[str, str]) -> bool:
    """Check history entry validity."""
    return isinstance(entry, dict) and "role" in entry and "content" in entry


def estimate_request_tokens(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
) -> int:
    """Estimate total tokens in a request including roles overhead."""
    total = estimate_tokens(system_prompt)
    for entry in history:
        if not validate_history_entry(entry):
            continue
        total += estimate_tokens(entry["content"])
        total += 2  # small role overhead
    total += estimate_tokens(user_message)
    total += 2
    return total


def calculate_gigachat_cost(
    prompt_tokens: int, completion_tokens: int, model: str
) -> Optional[float]:
    """Approximate GigaChat request cost (RUB)."""
    pricing = {
        "GigaChat": {"input": 0.01, "output": 0.03},
        "GigaChat-Pro": {"input": 0.05, "output": 0.15},
        "GigaChat-Pro-Max": {"input": 0.10, "output": 0.30},
    }
    model_pricing = pricing.get(model, pricing["GigaChat"])
    input_cost = (prompt_tokens / 1000.0) * model_pricing["input"]
    output_cost = (completion_tokens / 1000.0) * model_pricing["output"]
    return round(input_cost + output_cost, 6)


def extract_tokens_from_usage(usage: object) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extract token counts from a generic usage object."""
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        total_tokens = usage.get("total_tokens") or usage.get("tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None) or getattr(usage, "tokens", None)

        if total_tokens is None:
            try:
                if hasattr(usage, "dict"):
                    usage_dict = usage.dict()
                    if isinstance(usage_dict, dict):
                        total_tokens = usage_dict.get("total_tokens") or usage_dict.get("tokens") or usage_dict.get("total")
                elif hasattr(usage, "__dict__"):
                    usage_dict = usage.__dict__
                    if isinstance(usage_dict, dict):
                        total_tokens = usage_dict.get("total_tokens") or usage_dict.get("tokens") or usage_dict.get("total")
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

    if (total_tokens is None or total_tokens == 0) and prompt_tokens is not None and completion_tokens is not None:
        calculated_total = prompt_tokens + completion_tokens
        if total_tokens is None or total_tokens == 0:
            total_tokens = calculated_total

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


def create_meta_dict(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    total_tokens: Optional[int],
    elapsed: float,
    cost: Optional[float] = None,
) -> Dict[str, Optional[int | float]]:
    """Build meta dictionary for responses."""
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "elapsed": elapsed,
        "cost": cost,
    }


# -------------------------
# HTTP error handling (HF)
# -------------------------

def handle_hf_http_error(http_err: requests.HTTPError) -> None:
    """Raise clearer messages for common HF API HTTP errors."""
    response = http_err.response
    status = response.status_code if response else "?"
    detail = response.text if response else str(http_err)

    if status == 404:
        raise RuntimeError(
            "Модель недоступна через Hugging Face Inference API. "
            "Проверьте, опубликована ли она для Inference и не приватна."
        ) from http_err
    if status == 429:
        raise RuntimeError("Превышен лимит запросов к Hugging Face Inference API.") from http_err
    raise RuntimeError(f"Hugging Face вернул ошибку {status}: {detail}") from http_err


def handle_hf_response_error(response: requests.Response) -> None:
    """Raise clearer messages for non-2xx HF API responses."""
    if response.status_code == 404:
        raise RuntimeError(
            "Модель недоступна через Hugging Face Inference API. "
            "Проверьте, опубликована ли она для Inference и не приватна."
        )
    if response.status_code == 429:
        raise RuntimeError("Превышен лимит запросов к Hugging Face Inference API.")
    if response.status_code >= 400:
        raise RuntimeError(f"Hugging Face вернул ошибку {response.status_code}: {response.text}")


