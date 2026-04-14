# api/dependencies.py

from fastapi import Request

from services.search_service import HybridSearchService


def get_search_service(request: Request) -> HybridSearchService:
    """
    Dependency injection function for FastAPI Depends().
    """
    return request.app.state.search_service