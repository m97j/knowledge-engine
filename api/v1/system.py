# api/v1/system.py

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health Check"])

# ---------------------------
# Debug endpoint (optional)
# ---------------------------
@router.get("/ping")
def ping():
    return {"message": "pong"}