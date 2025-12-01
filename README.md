## GigaChat Assistant — веб-чат на Flask с GigaChat и Hugging Face

Веб-приложение предоставляет удобный интерфейс для общения с моделями **GigaChat** и **Hugging Face Inference API**, хранит историю диалогов в SQLite и позволяет работать с предустановленными системными промптами.

### Основные возможности

- **Веб-чат**: удобный интерфейс чата с современным UI (`templates/index.html`, `static/styles.css`).
- **Два провайдера**: переключение между GigaChat и моделями Hugging Face.
- **RAG (Retrieval-Augmented Generation)**: поддержка локальной векторной базы данных PostgreSQL с pgvector для контекстного поиска релевантных чанков.
- **Function Calling (Tools)**: интеграция внешних инструментов через GigaChat tools API.
  - **Weather Tool**: получение данных о погоде по координатам
  - **GitHub Tool**: работа с GitHub API (репозитории, issues, PR, файлы, поиск)
- **История сессий**: сохранение диалогов в SQLite (`chat_history/chat_history.db`) и просмотр истории через отдельную страницу.
- **Предустановки промптов**: хранение системных промптов (presets) в базе данных, создание новых через UI.
- **Метаданные запросов**: отображение токенов, времени ответа и примерной стоимости запросов GigaChat.

---

## Быстрый запуск

### Требования

- Python 3.13 (или совместимый 3.10+)
- Установленные зависимости из `requirements.txt`
- Зарегистрированные ключи:
  - **GigaChat**: переменная окружения `GIGACHAT_CREDENTIALS`
  - **Hugging Face**: переменная окружения `HUGGINGFACE_API_TOKEN` (опционально)
- **Для RAG (опционально)**:
  - PostgreSQL с расширением pgvector
  - Таблица `public.chunk_embeddings` с колонкой `embedding VECTOR(dim)`
  - Ollama для создания эмбеддингов (модель `nomic-embed-text`)

### Установка и запуск

```bash
cd /Users/aizen/gigachat_api
python -m venv venv
source venv/bin/activate  # в Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Настройка переменных окружения

Создайте файл `.env` в корне проекта (или скопируйте из `.env.example`, если есть) и добавьте:

```env
# Обязательные
FLASK_SECRET_KEY=change-me
GIGACHAT_CREDENTIALS=your_credentials_here

# Опциональные
HUGGINGFACE_API_TOKEN=your_hf_token_here

# Для GitHub Tool (опционально)
GITHUB_TOKEN=your_github_personal_access_token_here

# Для RAG (опционально)
DATABASE_URL=postgresql://user:pass@host:port/db
PGVECTOR_METRIC=cosine
TOP_K=8
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
```

**Альтернатива**: можно установить переменные через `export` в терминале перед запуском:

```bash
export GIGACHAT_CREDENTIALS="..."
export DATABASE_URL="postgresql://user:pass@host:port/db"
# и т.д.
```

### Запуск

```bash
python chatbot_gui.py
```

Приложение по умолчанию поднимается на `http://0.0.0.0:5050`.

---

## Архитектура проекта

### Обзор модулей

- **`chatbot_gui.py`**  
  Главный модуль Flask-приложения:
  - создаёт объект `app` и фабрику `create_app()`;
  - определяет маршруты:
    - `/` — основная страница чата;
    - `/history` — список сохранённых сессий;
    - `/history/<session_uuid>` — просмотр конкретного диалога;
  - управляет состоянием чата в cookie-сессии и историей сообщений через БД;
  - вызывает клиентов **GigaChat** (`ask_gigachat`) и **Hugging Face** (`ask_huggingface`).

- **`chat_utils.py`**  
  Набор чистых утилит:
  - `slugify()` — генерация ключа для предустановки;
  - `default_chat_state()` — базовое состояние чата;
  - утилиты токенов и стоимости: `estimate_tokens()`, `estimate_request_tokens()`, `calculate_gigachat_cost()`;
  - работа с метаданными: `extract_tokens_from_usage()`, `create_meta_dict()`;
  - обработка ошибок Hugging Face: `handle_hf_http_error()`, `handle_hf_response_error()`.

- **`db_utils.py`**  
  Слой доступа к SQLite:
  - инициализация и миграции структуры (`_init_db`, `_ensure_migrations`);
  - управление соединением в контексте Flask (`get_db`, `close_db`);
  - работа с сессиями: `db_ensure_session()`, `db_list_sessions()`, `db_clear_session()`;
  - работа с сообщениями: `db_add_message()`, `db_get_messages()`;
  - работа с предустановками: `db_load_presets()`, `db_upsert_preset()`, `make_unique_preset_key()`;
  - обслуживание WAL: `db_wal_checkpoint()`.

- **`gigachat_client.py`**
  Обёртка над SDK GigaChat:
  - `ask_gigachat()` формирует список сообщений (`system + history + user`),
    отправляет запрос через `GigaChat`, извлекает usage и считает стоимость.
  - Поддержка RAG: при `use_local_vectors=True` ищет релевантные чанки через `vector_store`
    и формирует контекст через `build_messages_with_context()`.
  - Поддержка Function Calling: регистрация и выполнение tools (weather, GitHub).

- **`vector_store.py`**
  Модуль для работы с векторной базой данных PostgreSQL (pgvector):
  - `init_pg()` — создание соединения с БД;
  - `embed_query()` — создание эмбеддинга запроса через Ollama API;
  - `search_chunks()` — поиск top-k релевантных чанков по cosine similarity.

- **`weather_tool.py`**
  Инструмент для получения данных о погоде:
  - `get_weather()` — запрос погоды по координатам;
  - `register_weather_tool()` — регистрация tool для GigaChat;
  - `execute_weather_tool()` — выполнение вызова tool;
  - интеграция с локальным MCP сервером (http://localhost:8080).

- **`github_tool.py`**
  Инструмент для работы с GitHub API (12 функций):
  - **Репозитории**: получение информации о репозитории
  - **Issues**: список, создание, обновление, комментарии
  - **Pull Requests**: список, детали PR
  - **Файлы и структура**:
    - `github_list_repo_contents()` — список файлов в директории
    - `github_get_repo_tree()` — полное дерево репозитория
    - `github_get_file_content()` — содержимое конкретного файла
  - **Поиск**: поиск кода в GitHub
  - **Коммиты**: список коммитов в репозитории
  - **Пользователи**: информация о пользователях GitHub
  - `register_github_tools()` — регистрация всех GitHub tools для GigaChat;
  - `execute_github_tool()` — выполнение вызовов GitHub tools.

- **`templates/index.html`**  
  Основной шаблон чата:
  - вывод истории (`history`) с разделением ролей `user/assistant`;
  - отображение метаданных (время, токены, стоимость);
  - формы ввода сообщения и настройки (провайдер, модель, температура, предустановка);
  - модальное окно для сохранения новых предустановок.

- **`templates/history.html`**  
  Список сохранённых сессий с ссылками на просмотр диалога.

- **`static/styles.css`**  
  Современный адаптивный UI (layout, чат‑bubbles, формы, модальные окна).

- **`config/`**  
  Сейчас папка используется как базовая директория для preset‑файлов, но фактическое хранение предустановок перенесено в SQLite.

- **`chat_history/`**  
  Каталог с SQLite‑базой `chat_history.db` и служебными файлами WAL.

---

## Поток данных и жизненный цикл запроса

### 1. Загрузка главной страницы (`GET /`)

1. Flask вызывает `index()` в `chatbot_gui.py`.
2. Загружаются предустановки: `load_presets()` → `db_load_presets()` (из SQLite).  
   При первом запуске при отсутствии записей в БД заполняются предустановки по умолчанию.
3. Состояние чата (настройки) берётся из cookie-сессии (`_load_session_state`), история — из БД (`_db_get_messages`).
4. Состояние валидируется (`_validate_chat_state`) и при необходимости восстанавливается через `default_chat_state()`.
5. Рендерится `templates/index.html` c:
   - `presets`, `providers`, `models`, `state`, `history`.

### 2. Отправка сообщения (`POST /`, action=`send`)

1. Форма чата отправляет:
   - `message` (текст пользователя),
   - текущие настройки: `preset`, `provider`, `model`, `temperature`.
2. В `index()`:
   - настройки парсятся (`_parse_form_settings`, `parse_temperature`);
   - состояние чата обновляется или пересоздаётся, если настройки изменились (`_check_settings_changed`, `_create_chat_state`);
   - определяется `system_prompt` из выбранной предустановки.
3. Подсчитываются токены:
   - `estimate_tokens(message)` — для текущего сообщения;
   - `estimate_request_tokens(system_prompt, history, message)` — для всего запроса.
4. Выбор провайдера:
   - если `provider == "gigachat"` → вызывается `ask_gigachat()` из `gigachat_client.py`:
     - если включен флаг `use_local_vectors` (через checkbox в UI):
       - создаётся эмбеддинг запроса через `embed_query()` (Ollama);
       - выполняется поиск релевантных чанков через `search_chunks()` (PostgreSQL/pgvector);
       - формируется контекст через `build_messages_with_context()` с найденными чанками;
       - в метаданные ответа добавляется информация об источниках (первые 3 чанка);
     - иначе — обычный режим без RAG;
   - иначе — `ask_huggingface()` в `chatbot_gui.py`, который:
     - при `mode="chat-completion"` обращается к `InferenceClient.chat.completions.create`;
     - при `mode="text-generation"` — к REST‑endpoint Hugging Face (`requests.post`).
5. Ответ провайдера возвращается как `(assistant_text, assistant_meta)`:
   - `assistant_meta` содержит токены, время, стоимость (для GigaChat) и др.
6. В `state["history"]` добавляются:
   - сообщение пользователя с `meta` (токены и `request_tokens`);
   - ответ ассистента с `meta` (и полем `provider`/`model`).
7. История сохраняется в БД:
   - генерируется/получается `session_id` (`_get_session_id`);
   - сессия создаётся/обновляется `db_ensure_session()` (заголовок — первое пользовательское сообщение);
   - в `chat_messages` пишутся две записи: `db_add_message()` для `user` и `assistant`.
8. Обновлённые настройки сохраняются в cookie‑сессии (`_save_session_state`), выполняется `redirect` на `/`.

### 3. Управление предустановками промптов

- **Сохранение новой предустановки**:
  - на форме настроек открывается модальное окно;
  - при `action="save_preset"` вызывается `_handle_save_preset()`:
    - `slugify()` + `make_unique_preset_key()` формируют уникальный ключ;
    - `db_upsert_preset()` сохраняет название и текст промпта в SQLite;
  - состояние чата переключается на новую предустановку.

### 4. История диалогов

- **Просмотр списка**:  
  `/history` → `history_list()` → `db_list_sessions()` → `templates/history.html`.
- **Просмотр конкретной сессии**:  
  `/history/<session_uuid>` → `history_view()` → `db_get_messages()` и рендер `index.html` в read‑only режиме (без отправки новых сообщений).

---

## Состояние, сессии и хранилище

- **Cookie‑сессия Flask**:
  - хранит только настройки (`preset_key`, `temperature`, `provider`, `model`) в `SESSION_KEY`;
  - хранит UUID текущего чата в `SESSION_ID_KEY`, который используется как первичный ключ для БД.

- **SQLite‑база (`chat_history/chat_history.db`)**:
  - `chat_sessions` — заголовок диалога, таймстемпы, UUID;
  - `chat_messages` — последовательность сообщений (роль, текст, `meta_json`, время);
  - `presets` — ключ, название, текст системного промпта, время создания.

---

## Расширение и модификации

- **Добавить нового провайдера модели**:
  - расширить словарь `AVAILABLE_PROVIDERS` в `chatbot_gui.py` (название, модели, режимы `task`/`mode`);
  - добавить новую ветку обработки в `index()` (по аналогии с GigaChat/Hugging Face).

- **Настроить RAG**:
  - убедиться, что PostgreSQL с pgvector настроен и таблица `public.chunk_embeddings` существует;
  - настроить переменные окружения: `DATABASE_URL`, `PGVECTOR_METRIC`, `TOP_K`, `OLLAMA_URL`, `OLLAMA_EMBED_MODEL`;
  - включить checkbox "Использовать локальную базу векторов (RAG)" в UI;
  - при включенном флаге ответы будут содержать контекст из найденных чанков и список источников.

- **Настроить GitHub Tool**:
  - создать Personal Access Token на https://github.com/settings/tokens;
  - добавить `GITHUB_TOKEN` в `.env` файл;
  - GitHub tool автоматически доступен в GigaChat через function calling;
  - примеры использования: "Покажи информацию о репозитории octocat/Hello-World", "Создай issue в моем репозитории", "Найди файл README.md в репозитории".

- **Изменить UI**:
  - править разметку в `templates/index.html` и стили в `static/styles.css`.

- **Добавить новые типы метаданных**:
  - расширить `create_meta_dict()` в `chat_utils.py`;
  - убедиться, что шаблон `index.html` корректно отображает новые поля `meta`.

- **Добавить новые Tools**:
  - создать новый файл `your_tool.py` по аналогии с `weather_tool.py` или `github_tool.py`;
  - определить функции и tool definitions в формате OpenAI function calling;
  - зарегистрировать tool в `gigachat_client.py` через `register_your_tool()`;
  - добавить обработку в `_execute_tool_call()` функцию.

---

## Принципы архитектуры

- **Минимализм**: только необходимые зависимости (Flask, gigachat SDK, huggingface_hub, SQLite через стандартную библиотеку).
- **Разделение ответственности**:
  - Flask‑роуты и HTTP‑логика — в `chatbot_gui.py`;
  - работа с БД — в `db_utils.py`;
  - общие утилиты и расчёты — в `chat_utils.py`;
  - интеграция с GigaChat — в `gigachat_client.py`;
  - tools (weather, GitHub) — в отдельных модулях.
- **Прозрачность данных**: история и настройки легко читаются/расширяются в SQLite и cookie‑сессии.
- **Готовность к расширению**: добавление новых моделей, типов провайдеров, tools и UI‑функций с минимальными изменениями в существующем коде.

---

## GitHub Tool — работа с GitHub API

### Возможности

GitHub Tool предоставляет интеграцию с GitHub API через GigaChat function calling. Все операции выполняются автоматически при упоминании GitHub в диалоге.

#### Поддерживаемые операции:

1. **Репозитории** (`github_get_repo`)
   - Получение информации о репозитории (описание, звезды, форки, язык)
   - Пример: "Покажи информацию о репозитории octocat/Hello-World"

2. **Issues** (`github_list_issues`, `github_get_issue`, `github_create_issue`, `github_update_issue`)
   - Список issues с фильтрацией по статусу и меткам
   - Просмотр детальной информации об issue
   - Создание новых issues
   - Обновление существующих issues (заголовок, описание, статус)
   - Примеры:
     - "Покажи открытые issues в репозитории username/repo"
     - "Создай issue 'Bug in login' в репозитории myrepo с описанием 'Users can't login'"

3. **Комментарии** (`github_create_comment`)
   - Добавление комментариев к issues и pull requests
   - Пример: "Добавь комментарий 'Fixed in PR #123' к issue #45 в репозитории username/repo"

4. **Структура репозитория** (НОВОЕ!)
   - `github_list_repo_contents` — список файлов и папок в директории
     - Пример: "Покажи файлы в корне репозитория username/repo"
     - Пример: "Что находится в папке src/ репозитория username/repo"
   - `github_get_repo_tree` — полное дерево репозитория
     - Получение всей структуры файлов рекурсивно
     - Пример: "Покажи структуру репозитория facebook/react"
     - Пример: "Какие файлы есть в репозитории torvalds/linux"

5. **Pull Requests** (`github_list_pull_requests`)
   - Список pull requests с фильтрацией по статусу
   - Пример: "Покажи открытые PR в репозитории username/repo"

6. **Файлы** (`github_get_file_content`)
   - Получение содержимого файлов из репозитория
   - Поддержка разных веток
   - Автоматическое декодирование base64
   - Пример: "Покажи содержимое файла README.md из репозитория octocat/Hello-World"

7. **Поиск кода** (`github_search_code`)
   - Глобальный поиск кода в GitHub
   - Поддержка фильтрации по языку, репозиторию
   - Пример: "Найди код 'async function' в репозитории username/repo на языке TypeScript"

8. **Коммиты** (`github_list_commits`)
   - Список коммитов в репозитории
   - Поддержка фильтрации по ветке
   - Пример: "Покажи последние 10 коммитов в репозитории username/repo"

9. **Пользователи** (`github_get_user_info`)
   - Информация о пользователях GitHub
   - Информация о текущем авторизованном пользователе
   - Пример: "Покажи информацию о пользователе octocat"

### Настройка

1. **Создание Personal Access Token**:
   - Перейдите на https://github.com/settings/tokens
   - Нажмите "Generate new token" → "Generate new token (classic)"
   - Выберите scopes (права доступа):
     - `repo` — полный доступ к репозиториям
     - `read:user` — чтение информации о пользователе
     - `user:email` — доступ к email
   - Скопируйте сгенерированный токен

2. **Добавление токена в проект**:
   ```bash
   # В файле .env добавьте:
   GITHUB_TOKEN=ghp_your_personal_access_token_here
   ```

3. **Проверка работы**:
   ```bash
   # Запустите тестовый скрипт
   python github_tool.py
   ```

### Примеры использования в чате

```
Пользователь: Покажи мне информацию о репозитории torvalds/linux

GigaChat: [автоматически вызывает github_get_repo("torvalds", "linux")]
         Репозиторий Linux kernel от Linus Torvalds:
         - Звезды: 180K+
         - Форки: 50K+
         - Язык: C
         - Описание: Linux kernel source tree
```

```
Пользователь: Создай issue в моем репозитории username/myproject с заголовком "Add tests" и описанием "Need unit tests for auth module"

GigaChat: [автоматически вызывает github_create_issue(...)]
         Issue успешно создан! #42: "Add tests"
         URL: https://github.com/username/myproject/issues/42
```

```
Пользователь: Найди все файлы с функциями async в репозитории microsoft/TypeScript

GigaChat: [автоматически вызывает github_search_code("repo:microsoft/TypeScript async function")]
         Найдено 150+ результатов:
         1. src/compiler/checker.ts - async function checkAsync()
         2. src/server/session.ts - async function executeCommand()
         ...
```

### Безопасность

- **Токен хранится только локально** в `.env` файле
- Токен **не логируется** и не отображается в UI
- Используется **HTTPS** для всех запросов к GitHub API
- Поддержка **rate limiting** GitHub API (5000 запросов/час для авторизованных пользователей)

### Ограничения

- Требуется валидный Personal Access Token с соответствующими правами
- Rate limits GitHub API: 5000 запросов/час (авторизованные), 60 запросов/час (неавторизованные)
- Некоторые операции требуют прав записи в репозиторий (создание issues, PR)
- Search API имеет дополнительные ограничения: 30 запросов/минуту

### Расширение функционала

Для добавления новых операций GitHub API:

1. Добавьте новую функцию в [github_tool.py](github_tool.py):
   ```python
   def github_new_operation(param1: str, param2: int) -> Dict[str, Any]:
       """Описание операции."""
       return _make_github_request("GET", f"/endpoint/{param1}/{param2}")
   ```

2. Добавьте tool definition в `GITHUB_TOOLS_DEFINITIONS`:
   ```python
   {
       "type": "function",
       "function": {
           "name": "github_new_operation",
           "description": "Описание для GigaChat",
           "parameters": {
               "type": "object",
               "properties": {
                   "param1": {"type": "string", "description": "..."},
                   "param2": {"type": "integer", "description": "..."}
               },
               "required": ["param1", "param2"]
           }
       }
   }
   ```

3. Добавьте обработку в `execute_github_tool()` (автоматически через `function_map`)
