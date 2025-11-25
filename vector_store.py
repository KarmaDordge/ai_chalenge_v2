"""
Модуль для работы с векторной базой данных PostgreSQL (pgvector).
Поддерживает поиск релевантных чанков по эмбеддингам.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

# Проверяем, какая версия psycopg доступна
try:
    import psycopg2
    PSYCOPG3 = False
except ImportError:
    try:
        import psycopg
        PSYCOPG3 = True
    except ImportError:
        raise ImportError(
            "Не установлен psycopg2-binary или psycopg. "
            "Установите: pip install psycopg2-binary"
        )

if not PSYCOPG3:
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        from psycopg2.pool import SimpleConnectionPool
    except ImportError:
        raise ImportError(
            "Не установлен psycopg2-binary. "
            "Установите: pip install psycopg2-binary"
        )
else:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        raise ImportError(
            "Не установлен psycopg. "
            "Установите: pip install psycopg"
        )

from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальный пул соединений
_connection_pool: Optional[Any] = None


def init_pg() -> Any:
    """
    Создает соединение с PostgreSQL по DATABASE_URL.
    
    Returns:
        Соединение с базой данных
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "Не найден DATABASE_URL. "
            "Добавьте переменную в .env или переменные окружения."
        )
    
    try:
        if PSYCOPG3:
            conn = psycopg.connect(database_url)
        else:
            conn = psycopg2.connect(database_url)
        logger.info("Успешное подключение к PostgreSQL")
        return conn
    except Exception as e:
        logger.error(f"Ошибка подключения к PostgreSQL: {e}")
        raise


def embed_query(query_text: str) -> List[float]:
    """
    Создает эмбеддинг для запроса используя Ollama nomic-embed-text.
    
    Args:
        query_text: Текст запроса для эмбеддинга
        
    Returns:
        Список чисел (вектор эмбеддинга)
    """
    import requests
    
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    
    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={
                "model": model,
                "prompt": query_text,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding", [])
        
        if not embedding:
            raise RuntimeError(f"Пустой эмбеддинг от Ollama: {data}")
        
        logger.info(f"Создан эмбеддинг размерности {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"Ошибка при создании эмбеддинга: {e}")
        raise RuntimeError(f"Не удалось создать эмбеддинг: {e}")


def search_chunks(
    embedding: List[float],
    k: int | None = None,
    filters: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Ищет top-k релевантных чанков в базе данных по эмбеддингу.
    
    Args:
        embedding: Вектор эмбеддинга запроса
        k: Количество чанков для возврата (по умолчанию из TOP_K)
        filters: Опциональные фильтры для поиска (например, по tags, path)
        
    Returns:
        Список словарей с полями: id, text, path, headings, tags, score
    """
    if k is None:
        k = int(os.getenv("TOP_K", 8))
    
    metric = os.getenv("PGVECTOR_METRIC", "cosine")
    
    # Проверяем метрику
    if metric not in ("cosine", "l2", "inner_product"):
        logger.warning(f"Неизвестная метрика {metric}, используем cosine")
        metric = "cosine"
    
    # Определяем оператор для метрики
    if metric == "cosine":
        operator = "<=>"
    elif metric == "l2":
        operator = "<->"
    else:  # inner_product
        operator = "<#>"
    
    conn = None
    results = []  # Инициализируем пустой список на случай ошибки
    try:
        conn = init_pg()
        
        # Опционально настраиваем параметры индекса
        ef_search = os.getenv("HNSW_EF_SEARCH")
        ivfflat_probes = os.getenv("IVFFLAT_PROBES")
        
        cur = conn.cursor()
        try:
            # Устанавливаем параметры индекса, если указаны
            if ef_search:
                cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
            if ivfflat_probes:
                cur.execute("SET LOCAL ivfflat.probes = %s", (ivfflat_probes,))
            
            # Формируем SQL запрос
            base_query = f"""
                SELECT 
                    id,
                    text,
                    source_file,
                    chunk_index,
                    metadata,
                    (embedding {operator} %s::vector) AS score
                FROM public.chunk_embeddings
            """
            
            # Добавляем фильтры, если есть
            where_clauses = []
            query_params = [embedding]  # Первый параметр - вектор эмбеддинга
            
            if filters:
                if "source_file" in filters or "path" in filters:
                    # Поддерживаем оба названия для обратной совместимости
                    file_filter = filters.get("source_file") or filters.get("path")
                    where_clauses.append("source_file = %s")
                    query_params.append(file_filter)
                if "metadata" in filters:
                    # Если metadata - JSON, можно фильтровать по содержимому
                    where_clauses.append("metadata @> %s::jsonb")
                    query_params.append(json.dumps(filters["metadata"]))
            
            if where_clauses:
                base_query += " WHERE " + " AND ".join(where_clauses)
            
            base_query += f" ORDER BY embedding {operator} %s::vector LIMIT %s"
            query_params.append(embedding)  # Второй параметр для ORDER BY
            query_params.append(k)  # Третий параметр для LIMIT
            
            # Выполняем запрос
            if PSYCOPG3:
                cur.execute(base_query, query_params)
                rows = cur.fetchall()
                # Преобразуем в список словарей
                results = []
                for row in rows:
                    # Обрабатываем metadata (может быть dict, str или None)
                    metadata = row[4]
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    elif metadata is None:
                        metadata = {}
                    
                    results.append({
                        "id": row[0],
                        "text": row[1],
                        "path": row[2],  # source_file -> path для обратной совместимости
                        "source_file": row[2],
                        "chunk_index": row[3],
                        "headings": metadata.get("headings", "") if isinstance(metadata, dict) else "",
                        "tags": metadata.get("tags", []) if isinstance(metadata, dict) else [],
                        "metadata": metadata,
                        "score": float(row[5]),
                    })
            else:
                # Для psycopg2 используем позиционные параметры
                cur.execute(base_query, query_params)
                rows = cur.fetchall()
                colnames = [desc[0] for desc in cur.description]
                results = []
                for row in rows:
                    row_dict = dict(zip(colnames, row))
                    # Обрабатываем metadata (может быть dict, str или None)
                    metadata = row_dict.get("metadata")
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                    elif metadata is None:
                        metadata = {}
                    
                    # Преобразуем для обратной совместимости
                    result = {
                        "id": row_dict.get("id"),
                        "text": row_dict.get("text"),
                        "path": row_dict.get("source_file"),  # source_file -> path
                        "source_file": row_dict.get("source_file"),
                        "chunk_index": row_dict.get("chunk_index"),
                        "metadata": metadata,
                        "score": float(row_dict.get("score", 0.0)),
                    }
                    # Извлекаем headings и tags из metadata
                    if isinstance(metadata, dict):
                        result["headings"] = metadata.get("headings", "")
                        result["tags"] = metadata.get("tags", [])
                    else:
                        result["headings"] = ""
                        result["tags"] = []
                    results.append(result)
        finally:
            cur.close()
        
        logger.info(f"Найдено {len(results)} релевантных чанков")
        return results
            
    except Exception as e:
        logger.error(f"Ошибка при поиске чанков: {e}")
        raise
    finally:
        if conn:
            conn.close()

