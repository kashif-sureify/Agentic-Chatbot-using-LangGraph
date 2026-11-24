from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

# Anchored to the project root so the database is the same no matter where the app is launched from
DB_PATH = Path(__file__).resolve().parent.parent.parent / "chatbot.db"

conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)

checkpointer = SqliteSaver(conn)


def _init_schema():
    # Tracked separately from the LangGraph-managed `checkpoints` table, which has no
    # per-thread row to hang a creation timestamp on
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    # Databases created before soft deletes still have the two-column table
    if "is_deleted" not in {row[1] for row in conn.execute("PRAGMA table_info(threads)")}:
        conn.execute(
            "ALTER TABLE threads ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0"
        )

    conn.commit()


_init_schema()


def record_thread_created(thread_id: str) -> None:
    """Stamp a thread's creation time the first time it is seen; later calls are no-ops."""
    conn.execute(
        "INSERT OR IGNORE INTO threads (thread_id, created_at) VALUES (?, ?)",
        (thread_id, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def delete_thread(thread_id: str) -> None:
    """Soft delete: hide the conversation from the UI but keep its history on disk."""
    conn.execute(
        "UPDATE threads SET is_deleted = 1 WHERE thread_id = ?", (thread_id,)
    )
    conn.commit()


def get_all_threads():
    """Return non-deleted thread IDs, oldest conversation first."""
    rows = conn.execute(
        """
        SELECT thread_id FROM threads
        WHERE is_deleted = 0
        ORDER BY created_at ASC, thread_id ASC
        """
    )
    return [row[0] for row in rows]
