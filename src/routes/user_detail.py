from fastapi import APIRouter, HTTPException
from typing import Dict
from scripts.chat_persistence_service import get_user_details  # Import your existing function

user_router = APIRouter(prefix="/users", tags=["users"])

@user_router.get("/{user_id}/details", response_model=Dict[str, str])
async def fetch_user_details(user_id: str):
    """
    Get stored details for a user (name, job, etc.).
    """
    details = get_user_details(user_id)
    if not details:
        raise HTTPException(status_code=404, detail="User not found")
    return details