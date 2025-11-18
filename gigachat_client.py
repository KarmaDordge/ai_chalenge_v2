from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from chat_utils import (
    validate_history_entry as cu_validate_history_entry,
    extract_tokens_from_usage as cu_extract_tokens_from_usage,
    create_meta_dict as cu_create_meta_dict,
    calculate_gigachat_cost as cu_calculate_gigachat_cost,
)

# Импорт weather tool
from weather_tool import (
    execute_weather_tool,
    register_weather_tool,
    get_weather,
)


def _extract_city_name(text: str) -> str | None:
    """
    Извлекает название города из текста.
    
    Args:
        text: Текст для поиска названия города
        
    Returns:
        Название города или None если не найдено
    """
    # Простой паттерн для поиска упоминаний городов
    # Ищем фразы типа "в [город]", "погода в [город]", "город [город]"
    patterns = [
        r'(?:в|городе?|погод[аы]?\s+в)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)',
        r'([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?)\s+(?:погод|температур)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            city_name = match.group(1).strip()
            # Проверяем, что это не служебное слово
            if city_name.lower() not in ['какая', 'какой', 'текущая', 'сейчас']:
                return city_name
    
    return None


def _get_city_coordinates_from_llm(city_name: str, credentials: str) -> tuple[float, float] | None:
    """
    Получает координаты города через GigaChat API.
    
    Args:
        city_name: Название города
        credentials: Учетные данные GigaChat
        
    Returns:
        Tuple[широта, долгота] или None если не удалось получить
    """
    try:
        geo_prompt = (
            f"Назови точные географические координаты города {city_name} в формате: "
            f"широта: число, долгота: число. "
            f"Ответь только двумя числами через запятую, например: 55.7558, 37.6173"
        )
        
        with GigaChat(credentials=credentials, verify_ssl_certs=False) as client:
            messages = [
                Messages(role=MessagesRole.SYSTEM, content="Ты помощник, который дает только координаты городов в формате: широта, долгота"),
                Messages(role=MessagesRole.USER, content=geo_prompt),
            ]
            
            chat = Chat(
                messages=messages,
                model="GigaChat",
                temperature=0.3,  # Низкая температура для более точного ответа
                flags=["no_cache"],
            )
            
            response = client.chat(chat)
            content = response.choices[0].message.content.strip()
            
            logger.info(f"Ответ GigaChat для координат города {city_name}: {content}")
            
            # Пытаемся извлечь координаты из ответа
            coord_match = re.search(r'([\d.]+)\s*[,;]\s*([\d.]+)', content)
            if coord_match:
                lat = float(coord_match.group(1))
                lon = float(coord_match.group(2))
                
                # Проверяем, что координаты в разумных пределах
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    logger.info(f"Получены координаты для {city_name}: {lat}, {lon}")
                    return lat, lon
            
            # Если не нашли в формате "число, число", пытаемся найти отдельно
            lat_match = re.search(r'широт[аы]?[:\s]+([\d.]+)', content, re.IGNORECASE)
            lon_match = re.search(r'долгот[аы]?[:\s]+([\d.]+)', content, re.IGNORECASE)
            
            if lat_match and lon_match:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    logger.info(f"Получены координаты для {city_name}: {lat}, {lon}")
                    return lat, lon
            
            logger.warning(f"Не удалось извлечь координаты из ответа GigaChat: {content}")
            return None
            
    except Exception as e:
        logger.error(f"Ошибка при получении координат города {city_name} через GigaChat: {e}")
        return None


def _extract_coordinates(text: str) -> tuple[float | None, float | None]:
    """
    Извлекает координаты из текста.
    
    Args:
        text: Текст для поиска координат
        
    Returns:
        Tuple[latitude, longitude] или (None, None) если не найдено
    """
    # Ищем паттерны координат
    coord_patterns = [
        r'weather\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)',  # weather(55.7558, 37.6173)
        r'широт[аы]\s*[:\s]*([\d.]+).*?долгот[аы]\s*[:\s]*([\d.]+)',  # широта: 55.7558, долгота: 37.6173
        r'latitude\s*[:\s]*([\d.]+).*?longitude\s*[:\s]*([\d.]+)',  # latitude: 55.7558, longitude: 37.6173
        r'координат[ы]?\s*[:\s]*([\d.]+)\s*,\s*([\d.]+)',  # координаты: 55.7558, 37.6173
        r'([\d.]+)\s*,\s*([\d.]+)',  # 55.7558, 37.6173 (последний паттерн, менее точный)
    ]
    
    for pattern in coord_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                # Проверяем, что координаты в разумных пределах
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    logger.info(f"Извлечены координаты из текста: lat={lat}, lon={lon}")
                    return lat, lon
            except (ValueError, IndexError):
                continue
    
    return None, None


def _execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выполняет вызов tool на основе данных из tool_call.
    
    Args:
        tool_call: Словарь с данными вызова tool от GigaChat API
        
    Returns:
        dict: Результат выполнения tool
    """
    function_name = tool_call.get("function", {}).get("name", "")
    
    # Обработка weather tool
    if function_name == "weather":
        # Преобразуем tool_call в формат, ожидаемый execute_weather_tool
        tool_call_for_execution = {
            "name": "weather",
            "arguments": tool_call.get("function", {}).get("arguments", "{}"),
        }
        return execute_weather_tool(tool_call_for_execution)
    
    # Если tool не найден, возвращаем ошибку
    return {
        "error": f"Неизвестный tool: {function_name}",
    }


def ask_gigachat(
    system_prompt: str,
    history: List[Dict[str, str]],
    user_message: str,
    temperature: float,
    model: str | None = None,
    enable_tools: bool = True,
) -> Tuple[str, Dict[str, Optional[int | float]]]:
    """
    Взаимодействие с GigaChat API: формирование сообщений, запрос, метаданные.
    
    Поддерживает автоматическую обработку tool calls (например, weather tool).
    
    Args:
        system_prompt: Системный промпт
        history: История сообщений
        user_message: Сообщение пользователя
        temperature: Температура генерации
        model: Модель GigaChat (по умолчанию "GigaChat")
        enable_tools: Включить ли поддержку tools (по умолчанию True)
        
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
        # Избегаем зависимости от AVAILABLE_PROVIDERS; берем безопасное значение по умолчанию
        model = "GigaChat"

    # Регистрируем tools, если включены
    tools = None
    if enable_tools:
        tools = register_weather_tool()
        logger.info(f"Зарегистрированы tools: {len(tools)} tool(s)")
        logger.debug(f"Tools definition: {json.dumps(tools, ensure_ascii=False, indent=2)}")

    # Добавляем информацию о доступных tools в системный промпт, если tools включены
    enhanced_system_prompt = system_prompt
    if enable_tools and tools:
        tools_info = "\n\nДоступные инструменты:\n"
        for tool in tools:
            tool_func = tool.get("function", {})
            tools_info += f"- {tool_func.get('name', 'unknown')}: {tool_func.get('description', '')}\n"
        enhanced_system_prompt = system_prompt + tools_info
    
    messages: List[Messages] = [
        Messages(role=MessagesRole.SYSTEM, content=enhanced_system_prompt),
    ]

    for entry in history:
        if not cu_validate_history_entry(entry):
            continue
        role = MessagesRole.USER if entry["role"] == "user" else MessagesRole.ASSISTANT
        messages.append(Messages(role=role, content=entry["content"]))

    # Проверяем, не упоминает ли пользователь погоду
    # Если да, пытаемся определить координаты (из текста или по названию города через LLM)
    weather_data = None
    city_name_for_response = None
    if enable_tools and tools:
        user_lower = user_message.lower()
        if "погод" in user_lower or "weather" in user_lower:
            # Сначала пытаемся извлечь координаты напрямую из текста
            latitude, longitude = _extract_coordinates(user_message)
            
            # Если координаты не найдены, пытаемся определить город и получить координаты через LLM
            if latitude is None or longitude is None:
                city_name = _extract_city_name(user_message)
                if city_name:
                    city_name_for_response = city_name
                    logger.info(f"Определен город: {city_name}, запрашиваем координаты через GigaChat")
                    coords = _get_city_coordinates_from_llm(city_name, credentials)
                    if coords:
                        latitude, longitude = coords
                        logger.info(f"Получены координаты города {city_name}: {latitude}, {longitude}")
                    else:
                        logger.warning(f"Не удалось получить координаты для города {city_name}")
            
            # Если нашли координаты (любым способом), вызываем weather tool
            if latitude is not None and longitude is not None:
                logger.info(f"Обнаружен запрос погоды с координатами: {latitude}, {longitude}")
                weather_data = get_weather(latitude, longitude)
                logger.info(f"Результат weather tool: {json.dumps(weather_data, ensure_ascii=False)}")
                # Не добавляем данные в промпт, чтобы GigaChat не генерировал лишние объяснения

    messages.append(Messages(role=MessagesRole.USER, content=user_message))

    # Начало измерения времени
    start = time.perf_counter()
    
    # Общие счетчики токенов для всех запросов (включая tool calls)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_sum = 0
    
    with GigaChat(credentials=credentials, verify_ssl_certs=False) as client:
        # Максимальное количество итераций для обработки tool calls (защита от бесконечного цикла)
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            # Логируем структуру tools перед отправкой
            if tools:
                logger.info(f"Отправка запроса с {len(tools)} tool(s)")
                logger.debug(f"Tools structure: {json.dumps(tools, ensure_ascii=False, indent=2)}")
            
            chat = Chat(
                messages=messages,
                model=model,
                temperature=temperature,
                flags=["no_cache"],
                tools=tools if tools else None,
            )
            
            # Логируем структуру Chat объекта
            logger.debug(f"Chat object: messages={len(messages)}, model={model}, tools={tools is not None}")
            
            response = client.chat(chat)
            
            # Логируем структуру ответа
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")
            
            # Собираем статистику токенов
            usage = getattr(response, "usage", None)
            if usage is not None:
                pt, ct, tt = cu_extract_tokens_from_usage(usage)
                if pt is not None:
                    total_prompt_tokens += pt
                if ct is not None:
                    total_completion_tokens += ct
                if tt is not None:
                    total_tokens_sum += tt
            
            message = response.choices[0].message
            
            # Логируем структуру ответа для отладки
            logger.debug(f"Response message type: {type(message)}")
            logger.debug(f"Response message attributes: {dir(message)}")
            logger.debug(f"Response message content: {getattr(message, 'content', None)}")
            
            # Проверяем, есть ли tool calls в ответе
            # Проверяем разные возможные атрибуты для tool calls
            tool_calls = None
            if hasattr(message, "tool_calls"):
                tool_calls = getattr(message, "tool_calls", None)
            elif hasattr(message, "function_calls"):
                tool_calls = getattr(message, "function_calls", None)
            elif hasattr(message, "tool_calls_list"):
                tool_calls = getattr(message, "tool_calls_list", None)
            
            # Также проверяем, может быть tool_calls в виде словаря
            if tool_calls is None:
                message_dict = message.__dict__ if hasattr(message, "__dict__") else {}
                tool_calls = message_dict.get("tool_calls") or message_dict.get("function_calls")
            
            logger.info(f"Tool calls found: {tool_calls is not None}, count: {len(tool_calls) if tool_calls else 0}")
            
            if tool_calls:
                logger.debug(f"Tool calls structure: {tool_calls}")
                logger.debug(f"Tool calls type: {type(tool_calls)}")
                logger.info(f"Выполняется {len(tool_calls)} tool call(s)")
            
            if tool_calls and len(tool_calls) > 0:
                # Добавляем ответ ассистента с tool calls в историю
                messages.append(Messages(
                    role=MessagesRole.ASSISTANT,
                    content=message.content or "",
                    tool_calls=tool_calls,
                ))
                
                # Выполняем все tool calls
                for tool_call in tool_calls:
                    logger.info(f"Выполняется tool call: {json.dumps(tool_call, ensure_ascii=False, indent=2)}")
                    tool_result = _execute_tool_call(tool_call)
                    logger.info(f"Результат tool call: {json.dumps(tool_result, ensure_ascii=False, indent=2)}")
                    
                    # Преобразуем результат в строку JSON
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                    
                    # Добавляем результат tool в историю
                    # GigaChat использует MessagesRole.FUNCTION для результатов tools
                    tool_call_id = tool_call.get("id", "")
                    messages.append(Messages(
                        role=MessagesRole.FUNCTION,
                        content=tool_result_str,
                        tool_call_id=tool_call_id,
                    ))
                
                # Продолжаем цикл, чтобы получить финальный ответ от модели
                iteration += 1
                continue
            
            # Если нет tool calls, но у нас уже есть данные о погоде из исходного сообщения,
            # возвращаем только температуру
            if not tool_calls and weather_data and "error" not in weather_data:
                # Используем название города, если оно было определено ранее
                if city_name_for_response:
                    final_content = f"Температура в {city_name_for_response}: {weather_data.get('temperature', 'N/A')}°C"
                else:
                    # Пытаемся извлечь название города еще раз (на случай, если оно не было определено ранее)
                    city_name, _ = _extract_city_name(user_message)
                    if city_name:
                        final_content = f"Температура в {city_name}: {weather_data.get('temperature', 'N/A')}°C"
                    else:
                        final_content = f"Температура: {weather_data.get('temperature', 'N/A')}°C"
                
                logger.info(f"Возвращаем упрощенный ответ о погоде: {final_content}")
                
                elapsed = time.perf_counter() - start
                prompt_tokens = total_prompt_tokens if total_prompt_tokens > 0 else None
                completion_tokens = total_completion_tokens if total_completion_tokens > 0 else None
                total_tokens = total_tokens_sum if total_tokens_sum > 0 else None
                
                cost = None
                if prompt_tokens is not None and completion_tokens is not None:
                    cost = cu_calculate_gigachat_cost(prompt_tokens, completion_tokens, model)

                meta = cu_create_meta_dict(
                    prompt_tokens, completion_tokens, total_tokens, elapsed, cost
                )
                
                return final_content, meta
            
            # Если нет tool calls, проверяем, не упоминает ли модель weather в ответе
            # Если да, пытаемся извлечь координаты и вызвать tool вручную
            if not tool_calls and enable_tools and tools:
                content = message.content or ""
                # Проверяем, упоминает ли модель weather tool
                if "weather" in content.lower() or "погод" in content.lower():
                    logger.info("Обнаружено упоминание weather в ответе, пытаемся извлечь координаты")
                    latitude, longitude = _extract_coordinates(content)
                    
                    # Если не нашли в ответе, ищем в исходном сообщении пользователя
                    if latitude is None or longitude is None:
                        latitude, longitude = _extract_coordinates(user_message)
                    
                    # Если координаты не найдены в ответе, пытаемся определить город и получить координаты через LLM
                    if latitude is None or longitude is None:
                        city_name = _extract_city_name(user_message)
                        if city_name:
                            city_name_for_response = city_name
                            logger.info(f"Определен город: {city_name}, запрашиваем координаты через GigaChat")
                            coords = _get_city_coordinates_from_llm(city_name, credentials)
                            if coords:
                                latitude, longitude = coords
                                logger.info(f"Получены координаты города {city_name} из сообщения пользователя: {latitude}, {longitude}")
                    
                    # Если нашли координаты, вызываем weather tool
                    if latitude is not None and longitude is not None:
                        logger.info(f"Вызываем weather tool вручную для координат: {latitude}, {longitude}")
                        weather_result = get_weather(latitude, longitude)
                        logger.info(f"Результат weather tool: {json.dumps(weather_result, ensure_ascii=False)}")
                        
                        # Формируем простой ответ только с температурой
                        if "error" not in weather_result:
                            # Используем название города, если оно было определено
                            city_name, _ = _extract_city_name(user_message)
                            if city_name:
                                final_content = f"Температура в {city_name}: {weather_result.get('temperature', 'N/A')}°C"
                            else:
                                final_content = f"Температура: {weather_result.get('temperature', 'N/A')}°C"
                        else:
                            final_content = f"Ошибка при получении данных о погоде: {weather_result.get('error', 'Неизвестная ошибка')}"
                        
                        elapsed = time.perf_counter() - start
                        prompt_tokens = total_prompt_tokens if total_prompt_tokens > 0 else None
                        completion_tokens = total_completion_tokens if total_completion_tokens > 0 else None
                        total_tokens = total_tokens_sum if total_tokens_sum > 0 else None
                        
                        cost = None
                        if prompt_tokens is not None and completion_tokens is not None:
                            cost = cu_calculate_gigachat_cost(prompt_tokens, completion_tokens, model)

                        meta = cu_create_meta_dict(
                            prompt_tokens, completion_tokens, total_tokens, elapsed, cost
                        )
                        
                        return final_content, meta
            
            # Если нет tool calls, получаем финальный ответ
            logger.info("Tool calls не обнаружены, возвращаем финальный ответ")
            elapsed = time.perf_counter() - start
            
            # Используем накопленные токены
            prompt_tokens = total_prompt_tokens if total_prompt_tokens > 0 else None
            completion_tokens = total_completion_tokens if total_completion_tokens > 0 else None
            total_tokens = total_tokens_sum if total_tokens_sum > 0 else None
            
            cost = None
            if prompt_tokens is not None and completion_tokens is not None:
                cost = cu_calculate_gigachat_cost(prompt_tokens, completion_tokens, model)

            meta = cu_create_meta_dict(
                prompt_tokens, completion_tokens, total_tokens, elapsed, cost
            )

            return message.content or "", meta
        
        # Если превышен лимит итераций, возвращаем последний ответ
        elapsed = time.perf_counter() - start
        prompt_tokens = total_prompt_tokens if total_prompt_tokens > 0 else None
        completion_tokens = total_completion_tokens if total_completion_tokens > 0 else None
        total_tokens = total_tokens_sum if total_tokens_sum > 0 else None
        
        cost = None
        if prompt_tokens is not None and completion_tokens is not None:
            cost = cu_calculate_gigachat_cost(prompt_tokens, completion_tokens, model)

        meta = cu_create_meta_dict(
            prompt_tokens, completion_tokens, total_tokens, elapsed, cost
        )

        return message.content or "Превышен лимит итераций обработки tool calls.", meta


