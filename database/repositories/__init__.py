"""
database/repositories/
────────────────────────
Repository layer — one file per domain entity.

Each repository wraps raw SQLite queries into clean, typed functions.
No business logic lives here — only data access.

Import pattern:
    from database.repositories.client_repository import ClientRepository
    repo = ClientRepository()
    clients = repo.get_all()
"""
