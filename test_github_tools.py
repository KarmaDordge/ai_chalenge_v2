#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы GitHub tools с GigaChat.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.WARNING,  # Убираем DEBUG для читаемости
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загружаем переменные окружения
load_dotenv()

# Проверяем наличие токенов
gigachat_creds = os.getenv("GIGACHAT_CREDENTIALS")
github_token = os.getenv("GITHUB_TOKEN")

if not gigachat_creds:
    print("❌ GIGACHAT_CREDENTIALS не найден в .env")
    sys.exit(1)

if not github_token:
    print("⚠️  GITHUB_TOKEN не найден - GitHub tools могут не работать")

print("=" * 80)
print("🧪 Тест GitHub Tools с GigaChat")
print("=" * 80)

# Импортируем функции
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters
from github_tool import register_github_tools
from gigachat_client import convert_openai_tools_to_gigachat_functions

# Регистрируем GitHub tools
tools = register_github_tools()
print(f"\n✅ Зарегистрировано {len(tools)} GitHub tools")

# Конвертируем в GigaChat Function format
functions = convert_openai_tools_to_gigachat_functions(tools)
print(f"✅ Сконвертировано в {len(functions)} GigaChat functions")

# Подготавливаем тестовый запрос
test_messages = [
    Messages(
        role=MessagesRole.SYSTEM,
        content="Ты — AI-ассистент с доступом к GitHub API через tools. "
                "Когда пользователь просит информацию о GitHub репозитории, "
                "ты ОБЯЗАТЕЛЬНО используешь доступные tools, а не просто описываешь их. "
                "ВАЖНО: Вызывай tools автоматически, не спрашивай разрешения!"
    ),
    Messages(
        role=MessagesRole.USER,
        content="Покажи список файлов в корне репозитория octocat/Hello-World"
    ),
]

# Пробуем разные модели
models_to_test = ["GigaChat-Pro", "GigaChat"]

for model_name in models_to_test:
    print(f"\n{'=' * 80}")
    print(f"🧪 Тестирование модели: {model_name}")
    print("=" * 80)

    try:
        with GigaChat(credentials=gigachat_creds, verify_ssl_certs=False) as client:
            chat = Chat(
                messages=test_messages,
                model=model_name,
                temperature=0.7,
                functions=functions,  # Используем functions вместо tools
                function_call="auto",  # Включаем автоматический вызов функций
            )

            print("\n🔄 Ожидание ответа от GigaChat...")
            response = client.chat(chat)

            print("\n📥 Ответ получен!")
            print("=" * 80)

            if not response.choices:
                print("❌ Нет choices в ответе")
                continue

            choice = response.choices[0]
            message = choice.message
            finish_reason = getattr(choice, "finish_reason", None)

            print(f"\n🔍 Finish reason: {finish_reason}")

            # Проверяем наличие function_call
            if finish_reason == "function_call":
                function_call = getattr(message, "function_call", None)
                if function_call:
                    print(f"\n✅ УСПЕХ! GigaChat вызвал функцию!")
                    print(f"\nДетали function call:")
                    print(f"  Function: {function_call.name}")
                    print(f"  Arguments: {function_call.arguments}")
                else:
                    print("\n❌ finish_reason='function_call', но function_call отсутствует")
            else:
                print("\n❌ GigaChat НЕ вызвал функции!")
                print("\nВместо этого вернул текст:")
                print("-" * 80)
                content = message.content if hasattr(message, 'content') else str(message)
                # Обрезаем до 500 символов для читаемости
                print(content[:500] + ("..." if len(content) > 500 else ""))
                print("-" * 80)

            # Выводим usage если есть
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                print(f"\n📊 Использование токенов:")
                print(f"   Prompt: {usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 'N/A'}")
                print(f"   Completion: {usage.completion_tokens if hasattr(usage, 'completion_tokens') else 'N/A'}")
                print(f"   Total: {usage.total_tokens if hasattr(usage, 'total_tokens') else 'N/A'}")

    except Exception as e:
        print(f"\n❌ Ошибка при тестировании {model_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "=" * 80)
print("✅ Тест завершен")
print("=" * 80)

print("\n💡 Выводы:")
print("   - Если хотя бы одна модель вызвала tools - функционал работает!")
print("   - Если обе модели НЕ вызвали tools - проблема в GigaChat API или формате")
print("   - Рекомендация: используйте GigaChat-Pro или GigaChat-Pro-Max для function calling")
