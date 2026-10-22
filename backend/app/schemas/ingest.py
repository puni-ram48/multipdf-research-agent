from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PageLayoutInfo(BaseModel):
    page: int
    layout: str
    gutter_x: Optional[float] = None
    transition_y: Optional[float] = None


class DocumentSummary(BaseModel):
    filename: str
    total_pages: int
    abstract_found: bool
    tables_found: int
    layouts: List[PageLayoutInfo]


class IngestResponse(BaseModel):
    session_id: str
    message: str
    total_documents: int
    total_chunks: int
    abstracts_indexed: int
    tables_indexed: int
    embedding_model: str
    file_summaries: List[DocumentSummary]