"""
database/
─────────
Centralised database layer for Advisor AI.

Exports the primary connection helper so pages can do:
    from database import get_db_connection
or continue using the legacy path:
    from services.database import get_db_connection
"""

from database.connection import get_db_connection, DB_PATH  # noqa: F401
from database.schema import init_database                    # noqa: F401
