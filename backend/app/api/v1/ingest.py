import os
import tempfile
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.schemas.ingest import IngestResponse, DocumentSummary
from app.services.pdf_parser import pdf_parser_service
from app.services.chunker import chunker_service
from app.services.embedding import embedding_service
from app.services.vector_store import vector_store_service
from app.services.rag_engine import rag_engine_service

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_documents(
    files: List[UploadFile] = File(...),
    session_id: str = Form(None)
):
    """
    Ingests one or more PDF research papers:
    1. Generates or reuses session_id.
    2. Parses PDFs with Docling & regex fallbacks.
    3. Splits documents into structure-aware Markdown/priority chunks.
    4. Embeds and updates session FAISS vector store index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Generate a new session_id if none provided
    active_session_id = session_id if session_id else str(uuid.uuid4())[:8]

    # Reset previous memory on new ingestion pass for session
    rag_engine_service.clear_memory(active_session_id)

    all_raw_documents = []
    file_summaries: List[DocumentSummary] = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a PDF."
            )

        file_bytes = await file.read()

        # Save to temporary file so Docling can access it directly from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            docs, summary = pdf_parser_service.process_pdf_file(
                file_bytes=file_bytes,
                filename=file.filename,
                temp_filepath=tmp_path
            )
            all_raw_documents.extend(docs)
            file_summaries.append(summary)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not all_raw_documents:
        raise HTTPException(
            status_code=422,
            detail="Failed to extract readable content from uploaded PDFs."
        )

    # Chunk documents
    chunks = chunker_service.split_documents(all_raw_documents)

    # Build vector index
    vector_store_service.build_index(active_session_id, chunks)

    abstracts_count = sum(1 for c in chunks if c.metadata.get("type") == "abstract")
    tables_count = sum(s.tables_found for s in file_summaries)
    embed_model_used = embedding_service.active_model_name or "BAAI/bge-large-en-v1.5"

    return IngestResponse(
        session_id=active_session_id,
        message="PDFs processed and vector store index generated successfully.",
        total_documents=len(files),
        total_chunks=len(chunks),
        abstracts_indexed=abstracts_count,
        tables_indexed=tables_count,
        embedding_model=embed_model_used,
        file_summaries=file_summaries
    )