import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_sqlite_checkpointer(database_path: Path) -> SqliteSaver:
    """Create a synchronous SQLite checkpointer for the local application."""

    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, check_same_thread=False)

    return SqliteSaver(connection)


@contextmanager
def open_sqlite_checkpointer(database_path: Path) -> Iterator[SqliteSaver]:
    """Open a SQLite checkpointer and always close its connection."""

    checkpointer = create_sqlite_checkpointer(database_path)

    try:
        yield checkpointer
    finally:
        checkpointer.conn.close()
