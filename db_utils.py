from __future__ import annotations

import json
import os
import sqlite3
import time
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from flask import g

# Paths (match structure from chatbot_gui.py)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "chat_history")
DB_PATH = os.path.join(HISTORY_DIR, "chat_history.db")


def get_db() -> sqlite3.Connection:
    """Returns a SQLite connection bound to the current app context."""
    db: Optional[sqlite3.Connection] = getattr(g, "_db_conn", None)
    if db is None:
        need_init = not os.path.exists(DB_PATH)
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        g._db_conn = db
        if need_init:
            _init_db(db)
        _ensure_migrations(db)
    return db


def _init_db(db: sqlite3.Connection) -> None:
    """Initializes database schema if not exists."""
    db.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uuid TEXT NOT NULL UNIQUE,
            title TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uuid TEXT NOT NULL,
            role TEXT NOT NULL, -- 'user' | 'assistant'
            content TEXT NOT NULL,
            meta_json TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(session_uuid) REFERENCES chat_sessions(session_uuid) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chat_messages_session_time
            ON chat_messages(session_uuid, created_at);

        -- Presets storage (replaces JSON files)
        CREATE TABLE IF NOT EXISTS presets (
            key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    )


def _ensure_migrations(db: sqlite3.Connection) -> None:
    """Apply idempotent migrations for databases created before new features."""
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS presets (
            key TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            prompt TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    )
    db.commit()


def close_db(exception: Optional[BaseException]) -> None:  # noqa: ARG001
    """Closes DB connection at the end of request."""
    db: Optional[sqlite3.Connection] = getattr(g, "_db_conn", None)
    if db is not None:
        db.close()
        g._db_conn = None


def db_ensure_session(session_uuid: str, title: Optional[str] = None) -> None:
    """Creates session row if it doesn't exist; updates title if provided."""
    now = int(time.time())
    db = get_db()
    cur = db.execute(
        "SELECT session_uuid FROM chat_sessions WHERE session_uuid = ?",
        (session_uuid,),
    )
    exists = cur.fetchone() is not None
    if not exists:
        db.execute(
            "INSERT INTO chat_sessions(session_uuid, title, created_at, updated_at) VALUES(?, ?, ?, ?)",
            (session_uuid, title or "", now, now),
        )
    elif title:
        db.execute(
            "UPDATE chat_sessions SET title = COALESCE(NULLIF(title, ''), ?), updated_at = ? WHERE session_uuid = ?",
            (title, now, session_uuid),
        )
    else:
        db.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE session_uuid = ?",
            (now, session_uuid),
        )
    db.commit()


def db_add_message(session_uuid: str, role: str, content: str, meta: Optional[dict]) -> None:
    """Appends a message into DB for the given session."""
    now = int(time.time())
    db = get_db()
    meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
    db.execute(
        "INSERT INTO chat_messages(session_uuid, role, content, meta_json, created_at) VALUES(?, ?, ?, ?, ?)",
        (session_uuid, role, content, meta_json, now),
    )
    db.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE session_uuid = ?",
        (now, session_uuid),
    )
    db.commit()


def db_clear_session(session_uuid: str) -> None:
    """Deletes all messages for a session and resets its title."""
    db = get_db()
    db.execute("DELETE FROM chat_messages WHERE session_uuid = ?", (session_uuid,))
    db.execute(
        "UPDATE chat_sessions SET title = '', updated_at = ? WHERE session_uuid = ?",
        (int(time.time()), session_uuid),
    )
    db.commit()


def db_list_sessions(limit: int = 100) -> list[dict]:
    """Returns recent chat sessions with basic info."""
    db = get_db()
    rows = db.execute(
        """
        SELECT s.session_uuid,
               COALESCE(NULLIF(s.title, ''), '(без названия)') AS title,
               s.created_at,
               s.updated_at,
               COUNT(m.id) AS messages_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON m.session_uuid = s.session_uuid
        GROUP BY s.session_uuid
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def db_get_messages(session_uuid: str) -> list[dict]:
    """Loads all messages for a session."""
    db = get_db()
    rows = db.execute(
        """
        SELECT role, content, meta_json, created_at
        FROM chat_messages
        WHERE session_uuid = ?
        ORDER BY created_at ASC, id ASC
        """,
        (session_uuid,),
    ).fetchall()
    result: list[dict] = []
    for r in rows:
        meta = None
        if r["meta_json"]:
            try:
                meta = json.loads(r["meta_json"])
            except Exception:
                meta = None
        result.append(
            {"role": r["role"], "content": r["content"], "meta": meta, "created_at": r["created_at"]}
        )
    return result


def db_load_presets() -> OrderedDict[str, Dict[str, str]]:
    """Loads presets from DB."""
    db = get_db()
    rows = db.execute("SELECT key, name, prompt FROM presets ORDER BY created_at ASC").fetchall()
    presets: OrderedDict[str, Dict[str, str]] = OrderedDict()
    for r in rows:
        presets[r["key"]] = {"name": r["name"], "prompt": r["prompt"]}
    return presets


def db_upsert_preset(key: str, name: str, prompt: str) -> None:
    """Creates or updates a preset in DB."""
    db = get_db()
    now = int(time.time())
    db.execute(
        """
        INSERT INTO presets(key, name, prompt, created_at)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            name=excluded.name,
            prompt=excluded.prompt
        """,
        (key, name, prompt, now),
    )
    db.commit()


def make_unique_preset_key(base_slug: str) -> str:
    """Return a unique preset key by appending -N if needed (queries DB)."""
    db = get_db()
    # Load existing keys that match base or base-N
    like_pattern = f"{base_slug}-%"
    rows = db.execute(
        "SELECT key FROM presets WHERE key = ? OR key LIKE ?",
        (base_slug, like_pattern),
    ).fetchall()
    existing: set[str] = {r["key"] for r in rows}
    if base_slug not in existing:
        return base_slug
    counter = 2
    while True:
        candidate = f"{base_slug}-{counter}"
        if candidate not in existing:
            return candidate
        counter += 1


def db_wal_checkpoint(truncate: bool = True) -> None:
    """Perform a WAL checkpoint to flush WAL into the main database file.

    Args:
        truncate: If True, uses TRUNCATE mode to also shrink the WAL file.
                  If False, uses PASSIVE mode (no blocking, may leave WAL data).
    """
    db = get_db()
    mode = "TRUNCATE" if truncate else "PASSIVE"
    # PRAGMA wal_checkpoint returns a result row; fetch it to ensure execution
    db.execute(f"PRAGMA wal_checkpoint({mode});").fetchall()

