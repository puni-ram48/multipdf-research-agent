from fastapi import APIRouter
from app.api.v1 import ingest, chat

api_router = APIRouter()

api_router.include_router(ingest.router, tags=["Ingest"])
api_router.include_router(chat.router, tags=["Chat"])