from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, description="Question asked by user")


class SourceCitation(BaseModel):
    source: str
    page: int
    section: str
    type: str
    layout: Optional[str] = None


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceCitation]
    faithfulness_score: Optional[float] = None