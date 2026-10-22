from fastapi import APIRouter, HTTPException, status
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_engine import rag_engine_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_papers(request: ChatRequest):
    """
    Executes RAG Q&A query against session vector index:
    Retrieves context -> injects memory -> executes Gemini 2.5 Flash -> scores faithfulness.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    response = rag_engine_service.answer_question(
        session_id=request.session_id,
        question=request.question.strip()
    )
    return response


@router.delete("/chat/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Clears conversation memory for a session."""
    rag_engine_service.clear_memory(session_id)
    return {"message": f"Conversation memory cleared for session '{session_id}'."}