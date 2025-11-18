"""
Модуль для работы с погодой через tool для GigaChat API.

Реализует функцию get_weather и регистрацию tool "weather" для использования
в GigaChat агенте. Использует стандартную библиотеку requests.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import requests


# URL сервера погоды
WEATHER_SERVER_URL = "http://localhost:8080"


def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Получает данные о погоде по координатам.
    
    Делает GET запрос к http://localhost:8080/?latitude={latitude}&longitude={longitude}
    и возвращает JSON с данными о погоде.
    
    Args:
        latitude: Широта (от -90 до 90)
        longitude: Долгота (от -180 до 180)
        
    Returns:
        dict: JSON с данными о погоде (температура, условие, скорость ветра, влажность)
              или словарь с ошибкой, если запрос не удался
              
    Example:
        >>> result = get_weather(66.7558, 37.6173)
        >>> print(result)
        {'temperature': 15, 'condition': 'clear', 'wind_speed': 5, 'humidity': 60}
    """
    try:
        # Формируем URL с параметрами
        url = f"{WEATHER_SERVER_URL}/"
        params = {
            "latitude": latitude,
            "longitude": longitude,
        }
        
        # Выполняем GET запрос
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        # Возвращаем JSON ответ
        return response.json()
        
    except requests.exceptions.RequestException as e:
        # В случае ошибки возвращаем словарь с описанием ошибки
        return {
            "error": f"Ошибка при запросе погоды: {str(e)}",
            "latitude": latitude,
            "longitude": longitude,
        }


# Определение tool для GigaChat API (формат OpenAI-style function calling)
WEATHER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": (
            "Получает актуальную информацию о погоде по географическим координатам. "
            "Возвращает температуру, условия, скорость ветра и влажность."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "description": "Широта в градусах (от -90 до 90)",
                },
                "longitude": {
                    "type": "number",
                    "description": "Долгота в градусах (от -180 до 180)",
                },
            },
            "required": ["latitude", "longitude"],
        },
    },
}


def get_weather_tool() -> Dict[str, Any]:
    """
    Возвращает определение tool "weather" для регистрации в GigaChat агенте.
    
    Returns:
        dict: Определение tool в формате GigaChat/OpenAI function calling
        
    Example:
        >>> tool = get_weather_tool()
        >>> # Использование в GigaChat:
        >>> # chat = Chat(messages=messages, tools=[tool])
    """
    return WEATHER_TOOL_DEFINITION


def execute_weather_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выполняет вызов tool "weather" на основе данных из tool_call.
    
    Args:
        tool_call: Словарь с данными вызова tool от GigaChat API.
                  Ожидается структура: {"name": "weather", "arguments": {...}}
                  
    Returns:
        dict: Результат выполнения функции get_weather
        
    Example:
        >>> tool_call = {
        ...     "name": "weather",
        ...     "arguments": '{"latitude": 66.7558, "longitude": 37.6173}'
        ... }
        >>> result = execute_weather_tool(tool_call)
        >>> print(result)
        {'temperature': 15, 'condition': 'clear', 'wind_speed': 5, 'humidity': 60}
    """
    # Извлекаем имя функции
    function_name = tool_call.get("name", "")
    
    if function_name != "weather":
        return {
            "error": f"Неизвестная функция: {function_name}",
        }
    
    # Парсим аргументы (могут быть строкой JSON или уже словарем)
    arguments = tool_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            return {
                "error": f"Ошибка парсинга аргументов: {str(e)}",
            }
    
    # Извлекаем координаты
    latitude = arguments.get("latitude")
    longitude = arguments.get("longitude")
    
    if latitude is None or longitude is None:
        return {
            "error": "Не указаны координаты latitude и/или longitude",
        }
    
    # Вызываем функцию получения погоды
    return get_weather(latitude, longitude)


def register_weather_tool(tools_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Регистрирует tool "weather" в списке tools для GigaChat.
    
    Если передан существующий список tools, добавляет weather tool к нему.
    Если список не передан, создает новый список с одним weather tool.
    
    Args:
        tools_list: Опциональный список существующих tools
        
    Returns:
        list: Список tools с добавленным weather tool
        
    Example:
        >>> # Создание нового списка tools
        >>> tools = register_weather_tool()
        >>> 
        >>> # Добавление к существующему списку
        >>> existing_tools = [other_tool]
        >>> all_tools = register_weather_tool(existing_tools)
    """
    if tools_list is None:
        tools_list = []
    
    # Проверяем, не добавлен ли уже weather tool
    weather_tool = get_weather_tool()
    for tool in tools_list:
        if tool.get("function", {}).get("name") == "weather":
            # Tool уже есть, возвращаем список без изменений
            return tools_list
    
    # Добавляем weather tool
    tools_list.append(weather_tool)
    return tools_list


# ---------------------------------------------------------------------------
# Пример использования
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Пример использования weather tool")
    print("=" * 60)
    
    # 1. Пример прямого вызова функции get_weather
    print("\n1. Прямой вызов get_weather:")
    print("-" * 60)
    coordinates = (66.7558, 37.6173)  # Пример координат (Москва)
    result = get_weather(coordinates[0], coordinates[1])
    print(f"Координаты: {coordinates}")
    print(f"Результат: {json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 2. Пример получения определения tool
    print("\n2. Определение tool для GigaChat:")
    print("-" * 60)
    tool_def = get_weather_tool()
    print(json.dumps(tool_def, ensure_ascii=False, indent=2))
    
    # 3. Пример регистрации tool
    print("\n3. Регистрация tool:")
    print("-" * 60)
    tools = register_weather_tool()
    print(f"Зарегистрировано tools: {len(tools)}")
    print(f"Имя tool: {tools[0]['function']['name']}")
    
    # 4. Пример выполнения tool_call (симуляция вызова от GigaChat)
    print("\n4. Симуляция вызова tool от GigaChat:")
    print("-" * 60)
    simulated_tool_call = {
        "name": "weather",
        "arguments": json.dumps({
            "latitude": coordinates[0],
            "longitude": coordinates[1],
        }),
    }
    tool_result = execute_weather_tool(simulated_tool_call)
    print(f"Tool call: {json.dumps(simulated_tool_call, ensure_ascii=False, indent=2)}")
    print(f"Результат выполнения: {json.dumps(tool_result, ensure_ascii=False, indent=2)}")
    
    print("\n" + "=" * 60)
    print("Пример завершен")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Пример интеграции в gigachat_client.py:
# ---------------------------------------------------------------------------
#
# from weather_tool import get_weather_tool, execute_weather_tool, register_weather_tool
# from gigachat.models import Chat, Messages, MessagesRole
#
# # При создании Chat объекта добавить tools:
# tools = register_weather_tool()  # или register_weather_tool(existing_tools)
# chat = Chat(
#     messages=messages,
#     model=model,
#     temperature=temperature,
#     tools=tools,  # Добавить tools параметр
#     flags=["no_cache"],
# )
#
# # При обработке ответа от GigaChat, если есть tool_calls:
# response = client.chat(chat)
# if hasattr(response, 'choices') and response.choices:
#     message = response.choices[0].message
#     if hasattr(message, 'tool_calls') and message.tool_calls:
#         # Выполнить вызовы tools
#         for tool_call in message.tool_calls:
#             if tool_call.get('function', {}).get('name') == 'weather':
#                 result = execute_weather_tool(tool_call)
#                 # Добавить результат обратно в историю сообщений
#
# ---------------------------------------------------------------------------

