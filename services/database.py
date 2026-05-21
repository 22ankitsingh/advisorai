"""
services/database.py
─────────────────────
Backward-compatible re-export facade.

All existing code that imports from this module continues to work unchanged:
    from services.database import get_db_connection, init_database

The real implementation now lives in:
    database/connection.py   → get_db_connection
    database/schema.py       → init_database

DO NOT add new logic here. Use the database/ package directly.
"""

# Re-export everything that was previously defined here
from database.connection import get_db_connection, DB_PATH  # noqa: F401
from database.schema import init_database                    # noqa: F401

__all__ = ["get_db_connection", "DB_PATH", "init_database"]
