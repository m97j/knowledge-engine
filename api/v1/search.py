# api/v1/search.py

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.dependencies import get_search_service
from api.schemas.search import SearchRequest, SearchResponse
from core.logger import setup_logger
from services.search_service import HybridSearchService

logger = setup_logger("search_api")

router = APIRouter(prefix="/search", tags=["Search"])
templates = Jinja2Templates(directory="templates")

# -------------------------------------
# Json API Endpoint for Hybrid Search
# -------------------------------------
@router.post("/", response_model=SearchResponse, summary="Execute Hybrid Search (JSON)")
async def execute_search(
    request_data: SearchRequest,
    search_service: HybridSearchService = Depends(get_search_service)
):
    """
    Execute a hybrid search using the provided query and parameters.
    """
    try:
        search_output = search_service.search(
            query=request_data.query, 
            top_k=request_data.top_k
        )
        return SearchResponse(
            query=search_output["query"], 
            results=search_output["results"], 
            latency_ms=search_output["latency_ms"]
        )
        
    except ValueError as ve:
        logger.warning(f"Invalid search request: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Search Execution Failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during search.")

# -------------------------------------
# HTML Demo Endpoint for Manual Testing
# -------------------------------------
@router.get("/demo", response_class=HTMLResponse, summary="Search Demo UI (GET)")
async def demo_page_get(request: Request):
    """
    Render a simple HTML page with a search form for manual testing of the hybrid search functionality.
    """
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "results": None, "query": ""}
    )

@router.post("/demo", response_class=HTMLResponse, summary="Search Demo UI (POST)")
async def demo_page_post(
    request: Request,
    query: str = Form(...),
    search_service: HybridSearchService = Depends(get_search_service)
):
    """
    Handle form submission from the demo page, execute the search, and render results in the same template.
    """
    try:
        search_output = search_service.search(query=query, top_k=5)
        
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": search_output["results"],
                "query": query,
                "latency_ms": search_output["latency_ms"]
            }
        )
    except Exception as e:
        logger.error(f"Demo Search Failed: {e}", exc_info=True)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "query": query,
                "error_message": "An error occurred while processing your search. Please try again."
            },
            status_code=500
        )