#!/usr/bin/env python3
"""
Полный тест function calling - вызов функции и получение финального ответа.
"""

import os
import sys
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Проверяем наличие токенов
gigachat_creds = os.getenv("GIGACHAT_CREDENTIALS")

if not gigachat_creds:
    print("❌ GIGACHAT_CREDENTIALS не найден в .env")
    sys.exit(1)

print("=" * 80)
print("🧪 Полный тест Function Calling с GigaChat")
print("=" * 80)

# Импортируем gigachat_client
from gigachat_client import ask_gigachat

# Тестовый запрос
user_message = "Покажи список файлов в корне репозитория octocat/Hello-World"
system_prompt = "Ты — AI-ассистент с доступом к GitHub API."

print(f"\n📝 Запрос: {user_message}")
print(f"🤖 Модель: GigaChat-Pro")
print("\n🔄 Отправка запроса...")

try:
    response, meta = ask_gigachat(
        system_prompt=system_prompt,
        history=[],
        user_message=user_message,
        temperature=0.7,
        model="GigaChat-Pro",
        enable_tools=True,
        use_local_vectors=False,
    )

    print("\n" + "=" * 80)
    print("✅ Получен финальный ответ!")
    print("=" * 80)
    print(f"\n{response}")

    print("\n" + "=" * 80)
    print("📊 Метаданные:")
    print("=" * 80)
    for key, value in meta.items():
        print(f"  {key}: {value}")

except Exception as e:
    print(f"\n❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ Тест завершен")
print("=" * 80)
