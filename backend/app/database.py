"""Database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables and apply lightweight SQLite column patches."""
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()


def _ensure_sqlite_columns():
    """Add columns introduced after first create_all (SQLite has no Alembic here)."""
    if "sqlite" not in settings.database_url:
        return
    from sqlalchemy import text

    with engine.begin() as conn:
        # project_predictions
        rows = conn.execute(text("PRAGMA table_info(project_predictions)")).fetchall()
        names = {r[1] for r in rows}
        if "low_confidence" not in names:
            conn.execute(
                text(
                    "ALTER TABLE project_predictions "
                    "ADD COLUMN low_confidence BOOLEAN DEFAULT 0"
                )
            )

        # decisions — assignee / committer
        drows = conn.execute(text("PRAGMA table_info(decisions)")).fetchall()
        dnames = {r[1] for r in drows}
        if "assignee_user_id" not in dnames:
            conn.execute(
                text("ALTER TABLE decisions ADD COLUMN assignee_user_id VARCHAR(36)")
            )
        if "committed_by_user_id" not in dnames:
            conn.execute(
                text("ALTER TABLE decisions ADD COLUMN committed_by_user_id VARCHAR(36)")
            )

