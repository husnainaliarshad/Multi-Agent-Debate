import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

# Get the directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'debate_app.db')}"

Base = declarative_base()

class DebateSession(Base):
    __tablename__ = "debate_sessions"
    
    id = Column(String, primary_key=True)
    topic = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    events = Column(JSON, default=[])
    result = Column(JSON, nullable=True)

# Create engine and session factory
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

def save_debate_session(session_id: str, topic: str, events: list, result: dict = None):
    """Save or update a debate session in the database."""
    db = SessionLocal()
    try:
        session = db.query(DebateSession).filter(DebateSession.id == session_id).first()
        if not session:
            session = DebateSession(id=session_id, topic=topic, events=events, result=result)
            db.add(session)
        else:
            session.events = events
            session.topic = topic  # Update topic on subsequent saves
            if result:
                session.result = result
        db.commit()
    except Exception as e:
        print(f"Error saving to database: {e}")
        db.rollback()
    finally:
        db.close()

def get_debate_events(session_id: str):
    """Retrieve events for a session from the database."""
    db = SessionLocal()
    try:
        session = db.query(DebateSession).filter(DebateSession.id == session_id).first()
        return session.events if session else None
    finally:
        db.close()

def get_recent_debates(limit: int = 10):
    """Get most recent debate sessions."""
    db = SessionLocal()
    try:
        sessions = db.query(DebateSession).order_by(DebateSession.created_at.desc()).limit(limit).all()
        return [
            {
                "session_id": s.id,
                "topic": s.topic,
                "timestamp": s.created_at.timestamp()
            }
            for s in sessions
        ]
    finally:
        db.close()

def delete_debate_session(session_id: str):
    """Delete a debate session from the database."""
    db = SessionLocal()
    try:
        session = db.query(DebateSession).filter(DebateSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
    except Exception as e:
        print(f"Error deleting session: {e}")
        db.rollback()
        raise
    finally:
        db.close()
