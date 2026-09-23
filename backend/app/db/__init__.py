"""SQLite persistence layer (Phase 2, Step 10)."""

from app.db.database import database_stats, init_database

__all__ = ["init_database", "database_stats"]
