from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
_FEEDBACK_TABLE = "proactive_feedback_events"
_FEEDBACK_COLUMNS = frozenset(
    {
        "id",
        "created_at",
        "session_key",
        "user_message_id",
        "assistant_message_id",
        "proactive_message_id",
        "feedback_type",
        "confidence",
        "pa_score",
        "pua_score",
        "lag_seconds",
        "candidate_count",
        "matched_by",
        "reason",
    }
)


@dataclass(frozen=True)
class FeedbackEvent:
    session_key: str
    user_message_id: str
    assistant_message_id: str
    proactive_message_id: str | None
    feedback_type: str
    confidence: str
    pa_score: float | None
    pua_score: float | None
    lag_seconds: int | None
    candidate_count: int
    matched_by: str
    reason: str


def open_db(path: Path) -> sqlite3.Connection:
    """打开反馈库并确保当前版本的 schema 已就绪。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ = conn.execute("PRAGMA journal_mode = WAL")
    _ = conn.execute("PRAGMA synchronous = NORMAL")
    _ = conn.executescript("""
        CREATE TABLE IF NOT EXISTS proactive_feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            session_key TEXT NOT NULL,
            user_message_id TEXT NOT NULL,
            assistant_message_id TEXT NOT NULL,
            proactive_message_id TEXT,
            feedback_type TEXT NOT NULL,
            confidence TEXT NOT NULL,
            pa_score REAL,
            pua_score REAL,
            lag_seconds INTEGER,
            candidate_count INTEGER NOT NULL,
            matched_by TEXT NOT NULL,
            reason TEXT NOT NULL,
            UNIQUE(user_message_id, proactive_message_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pfe_session_created
        ON proactive_feedback_events(session_key, created_at);

        CREATE INDEX IF NOT EXISTS idx_pfe_proactive
        ON proactive_feedback_events(proactive_message_id);

        CREATE UNIQUE INDEX IF NOT EXISTS idx_pfe_one_user_per_proactive
        ON proactive_feedback_events(proactive_message_id)
        WHERE proactive_message_id IS NOT NULL;
        """)
    current_version = schema_version(conn)
    if current_version not in {0, SCHEMA_VERSION}:
        conn.close()
        raise RuntimeError(
            f"proactive_feedback schema version {current_version} unsupported; "
            f"expected {SCHEMA_VERSION}"
        )
    if current_version == 0:
        _ = conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    try:
        validate_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    """返回 SQLite user_version，供启动健康检查和 schema 断言使用。"""

    row = conn.execute("PRAGMA user_version").fetchone()
    if row is None:
        raise RuntimeError("proactive_feedback schema version unavailable")
    return int(row[0])


def validate_schema(conn: sqlite3.Connection) -> None:
    """验证反馈表的必需列，避免坏库被当成空库继续消费。"""

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (_FEEDBACK_TABLE,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"proactive_feedback table missing: {_FEEDBACK_TABLE}")
    columns = {
        str(item[1])
        for item in conn.execute(f"PRAGMA table_info({_FEEDBACK_TABLE})").fetchall()
    }
    missing = sorted(_FEEDBACK_COLUMNS - columns)
    if missing:
        raise RuntimeError(
            "proactive_feedback schema missing columns: " + ", ".join(missing)
        )


def insert_feedback(conn: sqlite3.Connection, event: FeedbackEvent) -> int | None:
    if event.proactive_message_id is not None:
        existing = conn.execute(
            """
            SELECT id
            FROM proactive_feedback_events
            WHERE proactive_message_id = ?
              AND user_message_id <> ?
            LIMIT 1
            """,
            (event.proactive_message_id, event.user_message_id),
        ).fetchone()
        if existing is not None:
            return None
    if conn.execute("SELECT 1 FROM proactive_feedback_events WHERE user_message_id=?", (event.user_message_id,)).fetchone() is not None:
        return None
    cursor = conn.execute(
        """
        INSERT INTO proactive_feedback_events (
            session_key,
            user_message_id,
            assistant_message_id,
            proactive_message_id,
            feedback_type,
            confidence,
            pa_score,
            pua_score,
            lag_seconds,
            candidate_count,
            matched_by,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.session_key,
            event.user_message_id,
            event.assistant_message_id,
            event.proactive_message_id,
            event.feedback_type,
            event.confidence,
            event.pa_score,
            event.pua_score,
            event.lag_seconds,
            event.candidate_count,
            event.matched_by,
            event.reason,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("feedback insert failed")
    return int(row_id)
