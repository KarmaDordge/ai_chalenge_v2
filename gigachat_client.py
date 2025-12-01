from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters

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

# Импорт GitHub tool
from github_tool import (
    execute_github_tool,
    register_github_tools,
)

# Импорт vector store для RAG
VECTOR_STORE_AVAILABLE = False
embed_query = None
search_chunks = None

try:
    from vector_store import embed_query, search_chunks
    VECTOR_STORE_AVAILABLE = True
    logger.info("vector_store успешно импортирован, RAG доступен")
except ImportError as e:
    VECTOR_STORE_AVAILABLE = False
    logger.warning(f"vector_store не доступен, RAG будет отключен: {e}")
except Exception as e:
    VECTOR_STORE_AVAILABLE = False
    logger.error(f"Ошибка при импорте vector_store: {e}", exc_info=True)


def convert_openai_tools_to_gigachat_functions(tools: List[Dict[str, Any]]) -> List[Function]:
    """
    Конвертирует OpenAI-style tool definitions в GigaChat Function objects.

    Args:
        tools: Список определений tools в формате OpenAI

    Returns:
        Список объектов Function для GigaChat
    """
    functions = []

    for tool in tools:
        if tool.get("type") != "function":
            logger.warning(f"Пропускаем tool типа {tool.get('type')}, ожидается 'function'")
            continue

        func_def = tool.get("function", {})
        name = func_def.get("name")
        description = func_def.get("description", "")
        parameters = func_def.get("parameters", {})

        if not name:
            logger.warning(f"Tool без имени, пропускаем: {tool}")
            continue

        # Создаем FunctionParameters из OpenAI parameters
        func_params = FunctionParameters(
            type=parameters.get("type", "object"),
            properties=parameters.get("properties", {}),
            required=parameters.get("required", []),
        )

        # Создаем Function объект
        giga_function = Function(
            name=name,
            description=description,
            parameters=func_params,
        )

        functions.append(giga_function)
        logger.debug(f"Сконвертирована функция: {name}")

    return functions


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


def build_messages_with_context(
    question: str,
    chunks: List[Dict[str, Any]],
    system_prompt: str = "",
    history: List[Dict[str, str]] | None = None,
) -> List[Messages]:
    """
    Собирает messages для GigaChat с контекстом из найденных чанков.
    
    Args:
        question: Вопрос пользователя
        chunks: Список найденных релевантных чанков (уже отфильтрованных по порогу)
        system_prompt: Исходный системный промпт
        history: История сообщений
        
    Returns:
        Список сообщений для GigaChat API
    """
    # Ограничиваем длину контекста (максимум ~4000 символов)
    max_context_length = 4000
    context_parts = []
    current_length = 0
    
    # Формируем структурированный контекст с нумерацией и источниками
    for idx, chunk in enumerate(chunks, 1):
        chunk_text = chunk.get("text", "").strip()
        chunk_path = chunk.get("path") or chunk.get("source_file", "")
        chunk_headings = chunk.get("headings", "")
        chunk_similarity = chunk.get("similarity")
        chunk_index = chunk.get("chunk_index")
        
        # Формируем заголовок чанка с метаданными
        chunk_header_parts = [f"--- Фрагмент {idx} ---"]
        
        if chunk_path:
            chunk_header_parts.append(f"Источник: {chunk_path}")
        if chunk_index is not None:
            chunk_header_parts.append(f"Индекс: {chunk_index}")
        if chunk_headings:
            chunk_header_parts.append(f"Заголовок: {chunk_headings}")
        if chunk_similarity is not None:
            chunk_header_parts.append(f"Релевантность: {chunk_similarity:.3f}")
        
        chunk_header = "\n".join(chunk_header_parts)
        chunk_str = f"{chunk_header}\n{chunk_text}\n\n"
        
        if current_length + len(chunk_str) > max_context_length:
            logger.warning(f"Достигнут лимит длины контекста, добавлено {idx-1} из {len(chunks)} чанков")
            break
        
        context_parts.append(chunk_str)
        current_length += len(chunk_str)
    
    # Склеиваем контекст
    context_text = "\n".join(context_parts).strip()
    
    # Формируем расширенный системный промпт
    enhanced_system_prompt = (
        f"{system_prompt}\n\n"
        "=== РЕЛЕВАНТНЫЙ КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===\n"
        "Ниже предоставлены релевантные фрагменты из базы знаний, которые могут быть полезны для ответа на вопрос пользователя.\n\n"
        f"{context_text}\n"
        "=== КОНЕЦ КОНТЕКСТА ===\n\n"
        "ИНСТРУКЦИИ ПО ИСПОЛЬЗОВАНИЮ КОНТЕКСТА:\n"
        "1. Используй предоставленный контекст как основу для ответа, если он релевантен вопросу.\n"
        "2. Если контекст содержит точную информацию по вопросу - опирайся на неё и ссылайся на источники.\n"
        "3. Если контекст частично релевантен - используй его как отправную точку и дополняй своими знаниями для полного ответа.\n"
        "4. Если контекст не релевантен или недостаточен - можешь использовать свои знания, но упомяни об этом.\n"
        "5. При сопоставлении вопроса с контекстом учитывай возможные синонимы, связанные понятия и контекстный смысл.\n"
        "6. Стремись дать наиболее полный и точный ответ, комбинируя информацию из контекста и свои знания при необходимости.\n\n"
        "При ответе указывай источники, когда используешь информацию из предоставленного контекста "
        "(например: 'Согласно фрагменту 1 из источника X...' или 'На основе данных из базы знаний...')."
    )
    
    # Собираем messages
    messages: List[Messages] = [
        Messages(role=MessagesRole.SYSTEM, content=enhanced_system_prompt),
    ]
    
    # Добавляем историю
    if history:
        for entry in history:
            if not cu_validate_history_entry(entry):
                continue
            role = MessagesRole.USER if entry["role"] == "user" else MessagesRole.ASSISTANT
            messages.append(Messages(role=role, content=entry["content"]))
    
    # Добавляем текущий вопрос
    messages.append(Messages(role=MessagesRole.USER, content=question))
    
    return messages


def _execute_function_call(function_call) -> Dict[str, Any]:
    """
    Выполняет вызов функции на основе данных из function_call (GigaChat format).

    Args:
        function_call: Объект FunctionCall от GigaChat API с атрибутами name и arguments

    Returns:
        dict: Результат выполнения функции
    """
    function_name = getattr(function_call, "name", "")
    arguments_str = getattr(function_call, "arguments", "{}")

    logger.info(f"Executing function: {function_name}")
    logger.debug(f"Function arguments: {arguments_str}")

    # Обработка weather tool
    if function_name == "weather":
        tool_call_for_execution = {
            "name": "weather",
            "arguments": arguments_str,
        }
        return execute_weather_tool(tool_call_for_execution)

    # Обработка GitHub tools
    if function_name.startswith("github_"):
        # Преобразуем в формат, ожидаемый execute_github_tool
        tool_call_dict = {
            "function": {
                "name": function_name,
                "arguments": arguments_str,
            }
        }
        return execute_github_tool(tool_call_dict)

    # Если функция не найдена, возвращаем ошибку
    return {
        "error": f"Неизвестная функция: {function_name}",
    }


def _execute_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEPRECATED: Использовалась для старого формата tool_calls.
    Оставлена для обратной совместимости.

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

    # Обработка GitHub tools
    if function_name.startswith("github_"):
        return execute_github_tool(tool_call)

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
    use_local_vectors: bool = False,
    rag_threshold: float = 0.7,
    repository: Optional[str] = None,
) -> Tuple[str, Dict[str, Optional[int | float]]]:
    """
    Взаимодействие с GigaChat API: формирование сообщений, запрос, метаданные.
    
    Поддерживает автоматическую обработку tool calls (например, weather tool).
    Поддерживает RAG через локальную векторную базу данных с фильтрацией по порогу релевантности.
    
    Args:
        system_prompt: Системный промпт
        history: История сообщений
        user_message: Сообщение пользователя
        temperature: Температура генерации
        model: Модель GigaChat (по умолчанию "GigaChat")
        enable_tools: Включить ли поддержку tools (по умолчанию True)
        use_local_vectors: Использовать ли локальную векторную базу для RAG (по умолчанию False)
        rag_threshold: Порог релевантности для фильтрации чанков (0.0-1.0, по умолчанию 0.7)
                      Для cosine similarity: чем меньше score, тем лучше (score < threshold)
        repository: Репозиторий GitHub в формате owner/repo (опционально)
        
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

    # Обработка RAG: поиск релевантных чанков, если включен флаг
    logger.info(f"RAG проверка: use_local_vectors={use_local_vectors}, VECTOR_STORE_AVAILABLE={VECTOR_STORE_AVAILABLE}")
    logger.info(f"embed_query доступна: {embed_query is not None}, search_chunks доступна: {search_chunks is not None}")
    context_chunks = []
    if use_local_vectors:
        logger.info("RAG запрошен пользователем")
        if not VECTOR_STORE_AVAILABLE:
            logger.warning("RAG запрошен, но vector_store недоступен. Продолжаем без RAG.")
        elif embed_query is None or search_chunks is None:
            logger.error("Функции embed_query или search_chunks не импортированы")
        else:
            try:
                logger.info(f"Поиск релевантных чанков для RAG (порог: {rag_threshold})...")
                logger.info(f"Запрос пользователя: {user_message[:100]}...")
                
                # Получаем эмбеддинг запроса
                try:
                    query_embedding = embed_query(user_message)
                    logger.info(f"Создан эмбеддинг запроса, размерность: {len(query_embedding)}")
                except Exception as embed_error:
                    logger.error(f"Ошибка при создании эмбеддинга: {embed_error}")
                    logger.error("Проверьте, что Ollama запущен и доступен по адресу из OLLAMA_URL")
                    logger.error("Или проверьте, что модель эмбеддинга указана в OLLAMA_EMBED_MODEL")
                    raise RuntimeError(f"Не удалось создать эмбеддинг для RAG: {embed_error}") from embed_error
                
                # Ищем релевантные чанки
                try:
                    all_chunks = search_chunks(query_embedding)
                    logger.info(f"Найдено всего чанков: {len(all_chunks)}")
                except Exception as search_error:
                    logger.error(f"Ошибка при поиске чанков: {search_error}")
                    logger.error("Проверьте подключение к PostgreSQL и наличие таблицы chunk_embeddings")
                    raise RuntimeError(f"Не удалось найти чанки для RAG: {search_error}") from search_error
                
                if len(all_chunks) > 0:
                    logger.info(f"Первый чанк: score={all_chunks[0].get('score', 'N/A')}, path={all_chunks[0].get('path', 'N/A')}")
                
                # Фильтруем по порогу релевантности
                # Для cosine similarity: score < threshold (меньше = лучше)
                # Преобразуем score в similarity: similarity = 1 - score (для cosine)
                filtered_chunks = []
                for idx, chunk in enumerate(all_chunks):
                    score = chunk.get("score", 1.0)
                    # Для cosine: similarity = 1 - score
                    similarity = 1.0 - score if score <= 1.0 else 0.0
                    if idx < 3:  # Логируем первые 3 чанка
                        logger.info(f"Чанк {idx}: score={score:.4f}, similarity={similarity:.4f}, threshold={rag_threshold}, path={chunk.get('path', 'N/A')}")
                    if similarity >= rag_threshold:
                        chunk["similarity"] = similarity  # Добавляем similarity для отображения
                        filtered_chunks.append(chunk)
                
                context_chunks = filtered_chunks
                logger.info(f"Найдено {len(all_chunks)} чанков, после фильтрации (threshold={rag_threshold}): {len(context_chunks)}")
                if len(context_chunks) == 0:
                    logger.warning(f"Все чанки отфильтрованы порогом {rag_threshold}. Попробуйте снизить порог.")
                    logger.info(f"Примеры score найденных чанков: {[chunk.get('score', 0) for chunk in all_chunks[:5]]}")
            except RuntimeError as e:
                # RuntimeError означает критическую ошибку RAG - продолжаем без RAG
                logger.error(f"Критическая ошибка RAG: {e}. Продолжаем без RAG.")
                context_chunks = []
            except Exception as e:
                logger.error(f"Неожиданная ошибка при поиске чанков для RAG: {e}", exc_info=True)
                # Продолжаем без RAG в случае ошибки
                context_chunks = []
    else:
        logger.info("RAG не запрошен (use_local_vectors=False)")

    # Регистрируем tools, если включены
    tools = None
    functions = None
    if enable_tools:
        tools = register_weather_tool()
        tools = register_github_tools(tools)
        logger.info(f"Зарегистрированы tools: {len(tools)} tool(s)")
        logger.debug(f"Tools definition: {json.dumps(tools, ensure_ascii=False, indent=2)}")

        # Конвертируем OpenAI-style tools в GigaChat Function objects
        functions = convert_openai_tools_to_gigachat_functions(tools)
        logger.info(f"Сконвертировано в GigaChat functions: {len(functions)} function(s)")

    # Формируем messages: если включен RAG, используем build_messages_with_context
    if use_local_vectors and context_chunks:
        # Добавляем информацию о доступных tools в системный промпт, если tools включены
        enhanced_system_prompt = system_prompt
        if enable_tools and tools:
            tools_info = "\n\nДоступные инструменты:\n"
            for tool in tools:
                tool_func = tool.get("function", {})
                tools_info += f"- {tool_func.get('name', 'unknown')}: {tool_func.get('description', '')}\n"
            enhanced_system_prompt = system_prompt + tools_info
        
        # Добавляем информацию о репозитории, если указана
        if repository:
            repo_info = f"\n\nРабота с репозиторием GitHub: {repository}\n"
            repo_info += "При использовании GitHub инструментов используй этот репозиторий по умолчанию, если пользователь не указал другой.\n"
            enhanced_system_prompt = enhanced_system_prompt + repo_info
        
        messages = build_messages_with_context(
            question=user_message,
            chunks=context_chunks,
            system_prompt=enhanced_system_prompt,
            history=history,
        )
    else:
        # Обычный режим без RAG
        # Добавляем информацию о доступных tools в системный промпт, если tools включены
        enhanced_system_prompt = system_prompt
        if enable_tools and tools:
            tools_info = "\n\nДоступные инструменты:\n"
            for tool in tools:
                tool_func = tool.get("function", {})
                tools_info += f"- {tool_func.get('name', 'unknown')}: {tool_func.get('description', '')}\n"
            enhanced_system_prompt = system_prompt + tools_info
        
        # Добавляем информацию о репозитории, если указана
        if repository:
            repo_info = f"\n\nРабота с репозиторием GitHub: {repository}\n"
            repo_info += "При использовании GitHub инструментов используй этот репозиторий по умолчанию, если пользователь не указал другой.\n"
            enhanced_system_prompt = enhanced_system_prompt + repo_info
        
        messages: List[Messages] = [
            Messages(role=MessagesRole.SYSTEM, content=enhanced_system_prompt),
        ]

        for entry in history:
            if not cu_validate_history_entry(entry):
                continue
            role = MessagesRole.USER if entry["role"] == "user" else MessagesRole.ASSISTANT
            messages.append(Messages(role=role, content=entry["content"]))

        messages.append(Messages(role=MessagesRole.USER, content=user_message))

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
            # Логируем структуру functions перед отправкой
            if functions:
                logger.info(f"Отправка запроса с {len(functions)} function(s)")
                logger.debug(f"Functions: {[f.name for f in functions]}")

            # Создаем Chat с functions вместо tools
            chat_params = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "flags": ["no_cache"],
            }

            if functions:
                chat_params["functions"] = functions
                chat_params["function_call"] = "auto"

            chat = Chat(**chat_params)
            
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
            
            choice = response.choices[0]
            message = choice.message

            # Логируем структуру ответа для отладки
            logger.debug(f"Response message type: {type(message)}")
            logger.debug(f"Response message attributes: {dir(message)}")
            logger.debug(f"Response message content: {getattr(message, 'content', None)}")

            # GigaChat использует finish_reason == "function_call" для определения вызова функции
            finish_reason = getattr(choice, "finish_reason", None)
            logger.debug(f"Finish reason: {finish_reason}")

            # Проверяем наличие function_call в ответе
            function_call = None
            if finish_reason == "function_call":
                function_call = getattr(message, "function_call", None)
                logger.info(f"Function call detected: {function_call}")
                if function_call:
                    logger.debug(f"Function call type: {type(function_call)}")
                    logger.debug(f"Function call attributes: {dir(function_call)}")

            # Если есть function_call, выполняем функцию
            if function_call:
                # Добавляем сообщение ассистента с function_call в историю
                # ВАЖНО: передаем function_call в Messages, чтобы GigaChat знал о вызове функции
                messages.append(Messages(
                    role=MessagesRole.ASSISTANT,
                    content=message.content or "",
                    function_call=function_call,
                ))

                # Выполняем функцию
                logger.info(f"Выполняется function call: {function_call.name}")
                function_result = _execute_function_call(function_call)
                logger.info(f"Результат function call: {json.dumps(function_result, ensure_ascii=False, indent=2)}")

                # Преобразуем результат в строку JSON
                function_result_str = json.dumps(function_result, ensure_ascii=False)

                # Добавляем результат функции в историю
                # GigaChat использует MessagesRole.FUNCTION для результатов функций
                messages.append(Messages(
                    role=MessagesRole.FUNCTION,
                    content=function_result_str,
                ))

                # Продолжаем цикл, чтобы получить финальный ответ от модели
                iteration += 1
                continue
            
            # Если нет function_call, но у нас уже есть данные о погоде из исходного сообщения,
            # возвращаем только температуру
            if not function_call and weather_data and "error" not in weather_data:
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
            
            # Если нет function_call, проверяем, не упоминает ли модель weather в ответе
            # Если да, пытаемся извлечь координаты и вызвать tool вручную
            if not function_call and enable_tools and tools:
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
            
            # Добавляем информацию об источниках, если использовался RAG
            if use_local_vectors and context_chunks:
                # Берем все использованные чанки (уже отсортированы по релевантности)
                sources = []
                for chunk in context_chunks:
                    source_info = {
                        "path": chunk.get("path") or chunk.get("source_file", ""),
                        "headings": chunk.get("headings", ""),
                        "similarity": chunk.get("similarity"),
                        "score": chunk.get("score", 0.0),
                        "chunk_index": chunk.get("chunk_index"),
                    }
                    sources.append(source_info)
                meta["sources"] = sources
                meta["rag_threshold"] = rag_threshold
                meta["chunks_found"] = len(context_chunks)

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


