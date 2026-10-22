from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, JSON
from app.db.database import Base


class ChatSessionMessage(Base):
    """Stores individual message turns, source citations, and evaluation metrics per session."""
    __tablename__ = "chat_session_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(64), index=True, nullable=False)
    user_question = Column(Text, nullable=False)
    assistant_answer = Column(Text, nullable=False)
    faithfulness_score = Column(Float, nullable=True)
    citations = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)