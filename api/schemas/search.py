# api/schemas/search.py

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------
# Request
# ---------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=50)

    # optional
    use_reranker: Optional[bool] = True

# ---------------------------
# Document metadata
# ---------------------------
class DocumentMetadata(BaseModel):
    doc_id: int
    title: str
    lang: str
    url: Optional[str] = None
    date_modified: Optional[str] = None

# ---------------------------
# Result item (LLM-friendly)
# ---------------------------
class SearchResultItem(BaseModel):
    chunk_id: int
    text: str
    score: float = Field(..., description="Reranking score (0.0 to 1.0)")
    metadata: DocumentMetadata
    scoring_details: Optional[Dict[str, Any]] = None     # optional

# ---------------------------
# Response
# ---------------------------
class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    latency_ms: int
