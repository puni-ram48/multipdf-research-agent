from fastapi import Header, HTTPException, status
from app.core.config import settings


async def verify_api_key(x_api_key: str = Header(None)):
    """
    Optional API Key check for production security.
    If no GEMINI_API_KEY or strict API key setting is enforced, allows traffic through.
    """
    # Extend this function when deploying to cloud environments requiring client authorization keys
    return True