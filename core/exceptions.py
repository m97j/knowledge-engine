# core/exceptions.py

from fastapi import Request, status
from fastapi.responses import JSONResponse

from core.logger import setup_logger

logger = setup_logger("exception_handler")

# ---------------------------------------------------
# Base Exception (Parent class of all custom errors)
# ---------------------------------------------------
class KnowledgeEngineException(Exception):
    """Base custom exception class for the Knowledge Engine application"""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# ---------------------------------------------------
# Domain Specific Exceptions (Hierarchical error)
# ---------------------------------------------------
class ModelLoadError(KnowledgeEngineException):
    """models/ layer where model (Embedder/Reranker) loading fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

class DatabaseError(KnowledgeEngineException):
    """storage/ layer where Qdrant or SQLite integration fails"""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

class SearchExecutionError(KnowledgeEngineException):
    """services/ layer where the search pipeline (Hybrid Search) encounters a logical error"""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class InvalidQueryError(KnowledgeEngineException):
    """api/ layer where user input is invalid (e.g., empty query, unsupported parameters)"""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)

# -----------------------------------
# FastAPI Exception Handler 
# -----------------------------------
async def custom_exception_handler(request: Request, exc: KnowledgeEngineException):
    """
    When a custom exception occurs in a FastAPI app,  
    catch it and convert it into a consistent JSON error response.
    """
    logger.error(f"[{exc.status_code}] {request.method} {request.url} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "path": str(request.url.path)
        }
    )
async def global_exception_handler(request: Request, exc: Exception):
    """Catch any unhandled exceptions that are not instances of KnowledgeEngineException,  
    log them, and return a generic error response."""
    logger.critical(f"Unhandled Exception: {str(exc)}", exc_info=True) # Log stack trace for debugging
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "message": "An unexpected error occurred."}
    )

def setup_exception_handlers(app):
    """Register custom exception handlers to the FastAPI app."""
    app.add_exception_handler(KnowledgeEngineException, custom_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)