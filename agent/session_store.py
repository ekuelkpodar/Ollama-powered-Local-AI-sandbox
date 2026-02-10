"""SQLite session persistence for conversations."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class SessionRecord:
    """Session record container."""
    session_id: str
    created_at: str
    updated_at: str
    title: str | None
    model: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SessionStore:
    """SQLite-backed store for conversation sessions."""

    def __init__(self, db_path: str, max_sessions: int = 100):
        self.db_path = db_path
        self.max_sessions = max_sessions
        self._ensure_directory()
        self._init_db()

    def ensure_session(self, session_id: str, model: str | None) -> None:
        """Ensure a session exists and update its model if needed."""
        now = self._now()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, created_at, updated_at, model, title,
                        prompt_tokens, completion_tokens, total_tokens
                    )
                    VALUES (?, ?, ?, ?, NULL, 0, 0, 0)
                    """,
                    (session_id, now, now, model),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ?, model = COALESCE(?, model) WHERE session_id = ?",
                    (now, model, session_id),
                )
        self._prune_sessions()

    def save_message(self, session_id: str, role: str, content: str, created_at: str | None = None) -> None:
        """Persist a message to the session."""
        timestamp = created_at or self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, timestamp),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
            if role == "user":
                title = self._derive_title(content)
                if title:
                    conn.execute(
                        """
                        UPDATE sessions
                        SET title = ?
                        WHERE session_id = ? AND (title IS NULL OR title = '')
                        """,
                        (title, session_id),
                    )
        self._prune_sessions()

    def set_title(self, session_id: str, title: str) -> None:
        """Set the session title."""
        clean = self._derive_title(title) or title.strip()
        if not clean:
            return
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (clean, self._now(), session_id),
            )

    def save_tool_call(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: str | None,
        created_at: str | None = None,
    ) -> None:
        """Persist a tool call entry."""
        timestamp = created_at or self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls (session_id, tool_name, args_json, result, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, tool_name, json.dumps(args), result, timestamp),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
        self._prune_sessions()

    def update_token_counts(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> None:
        """Update token counts for a session."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET prompt_tokens = ?, completion_tokens = ?, total_tokens = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (prompt_tokens, completion_tokens, total_tokens, self._now(), session_id),
            )

    def increment_token_counts(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Increment token counters for a session."""
        total_tokens = prompt_tokens + completion_tokens
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET prompt_tokens = prompt_tokens + ?,
                    completion_tokens = completion_tokens + ?,
                    total_tokens = total_tokens + ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (prompt_tokens, completion_tokens, total_tokens, self._now(), session_id),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a full session with messages and tool calls."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None

            messages = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

            tool_calls = conn.execute(
                """
                SELECT tool_name, args_json, result, created_at
                FROM tool_calls WHERE session_id = ? ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "title": row["title"],
            "model": row["model"],
            "prompt_tokens": row["prompt_tokens"],
            "completion_tokens": row["completion_tokens"],
            "total_tokens": row["total_tokens"],
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "created_at": m["created_at"],
                }
                for m in messages
            ],
            "tool_calls": [
                {
                    "tool_name": t["tool_name"],
                    "args": json.loads(t["args_json"]) if t["args_json"] else {},
                    "result": t["result"],
                    "created_at": t["created_at"],
                }
                for t in tool_calls
            ],
        }

    def list_sessions(self, limit: int | None = None) -> list[dict[str, Any]]:
        """List sessions with summary info."""
        query = """
            SELECT
                s.session_id,
                s.created_at,
                s.updated_at,
                s.title,
                s.model,
                s.prompt_tokens,
                s.completion_tokens,
                s.total_tokens,
                (
                    SELECT COUNT(1)
                    FROM messages m
                    WHERE m.session_id = s.session_id
                ) AS message_count
            FROM sessions s
            ORDER BY s.updated_at DESC
        """
        if limit is not None:
            query += " LIMIT ?"

        with self._connect() as conn:
            rows = conn.execute(query, (limit,) if limit is not None else ()).fetchall()

        return [
            {
                "session_id": r["session_id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "title": r["title"],
                "model": r["model"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "total_tokens": r["total_tokens"],
                "message_count": r["message_count"],
            }
            for r in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and related records."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            return cur.rowcount > 0

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model TEXT,
                    title TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
                """
            )
            self._ensure_column(conn, "sessions", "title", "TEXT")

    def _prune_sessions(self) -> None:
        if self.max_sessions <= 0:
            return
        with self._connect() as conn:
            count_row = conn.execute("SELECT COUNT(1) AS cnt FROM sessions").fetchone()
            total = count_row["cnt"] if count_row else 0
            if total <= self.max_sessions:
                return

            excess = total - self.max_sessions
            rows = conn.execute(
                """
                SELECT session_id FROM sessions
                ORDER BY updated_at ASC, created_at ASC
                LIMIT ?
                """,
                (excess,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (row["session_id"],),
                )

    def _ensure_directory(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(directory, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _derive_title(content: str, max_len: int = 60) -> str:
        cleaned = " ".join(content.strip().split())
        if not cleaned:
            return ""
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[:max_len].rstrip() + "..."

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {c["name"] for c in cols}
        if column in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
