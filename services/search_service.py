# services/search_service.py

import time
from typing import Any, Dict, List

from api.schemas.search import DocumentMetadata, SearchResultItem
from core.exceptions import SearchExecutionError
from core.logger import setup_logger
from models.embedder import TextEmbedder
from models.reranker import TextReranker
from storage.qdrant_client import QdrantStorage
from storage.sqlite_client import SQLiteStorage

logger = setup_logger("search_service")

class HybridSearchService:
    """
    It is a business logic service that derives final search results by integrating  
    Qdrant (Vector DB), SQLite (RDBMS), Embedder, and Reranker.
    """
    def __init__(self, qdrant: QdrantStorage, sqlite: SQLiteStorage, embedder: TextEmbedder, reranker: TextReranker):
        self.qdrant = qdrant
        self.sqlite = sqlite
        self.embedder = embedder
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5, limit: int = 50) -> Dict[str, Any]:
        """
        Receives user queries and performs hybrid search and reranking.
        
        :param query: User search query
        :param top_k: Number of documents to return (after reranking)
        :param limit: Number of candidate documents to fetch from Qdrant (after RRF fusion, before reranking)
        """
        start_time = time.time()
        logger.info(f"🔍 Starting search pipeline for query: '{query}'")

        try:
            # 1. Query Embedding (Dense, Sparse Extraction)
            encoded_query = self.embedder.encode_query(query)

            # 2. Qdrant Hybrid Search (Extract limit of candidates using RRF method)
            qdrant_results = self.qdrant.hybrid_search(
                dense_vector=encoded_query.dense_vector,
                sparse_indices=encoded_query.sparse_indices,
                sparse_values=encoded_query.sparse_values,
                limit=limit
            )
            
            if not qdrant_results:
                logger.warning("No results found in Vector DB.")
                return self._build_empty_response(query, start_time)

            chunk_ids = [res.id for res in qdrant_results]

            # 3. Get Dict in SQLite for O(1) Mapping of Source Text and Metadata
            sqlite_data_map = self.sqlite.get_enriched_chunks_dict(chunk_ids)

            # 4. Data Preparation for Reranking (Merging Qdrant and SQLite Data)
            chunks_for_reranking = []
            for rank, res in enumerate(qdrant_results, start=1):
                # Defense Logic: Skip data inconsistencies (Desync) in Vector DB but not in SQLite
                chunk_info = sqlite_data_map.get(res.id)
                if not chunk_info:
                    logger.warning(f"Data Desync: chunk_id {res.id} found in Qdrant but missing in SQLite.")
                    continue
                
                chunks_for_reranking.append({
                    "chunk_id": res.id,
                    "text": chunk_info["text"],
                    "metadata": chunk_info["metadata"],
                    "rrf_score": res.score,
                    "rrf_rank": rank
                })

            if not chunks_for_reranking:
                return self._build_empty_response(query, start_time)

            # 5. Perform Cross-Encoder Reranking
            # Return a list sorted in descending order after recalculating context-based precise scores
            reranked_docs = self.reranker.rerank(
                query=query, 
                documents=chunks_for_reranking, 
                text_key="text"
            )

            # 6. Top-K Truncation and Mapping to Pydantic Schema (SearchResultItem) Specification
            final_results = []
            for doc in reranked_docs[:top_k]:
                final_results.append(SearchResultItem(
                    chunk_id=doc["chunk_id"],
                    text=doc["text"],
                    score=round(doc["rerank_score"], 4), # Neatly rounded to 4 decimal places
                    metadata=DocumentMetadata(**doc["metadata"])
                ).model_dump()) # Convert to dict for FastAPI compatibility

            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"✅ Search completed in {latency_ms}ms. Found {len(final_results)} final chunks.")

            return {
                "query": query,
                "results": final_results,
                "latency_ms": latency_ms
            }

        except Exception as e:
            # Wrap unexpected errors in custom errors and throw them to the router
            logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
            raise SearchExecutionError(f"Search pipeline failed: {str(e)}")

    def _build_empty_response(self, query: str, start_time: float) -> Dict[str, Any]:
        """Build a standard response format when no search results are found"""
        return {
            "query": query,
            "results": [],
            "latency_ms": int((time.time() - start_time) * 1000)
        }

    # ---------------------------------------------------------
    # LLM-Friendly Prompt Formatter 
    # (Utility used when injecting into Agents or VLMs)
    # ---------------------------------------------------------
    def format_for_llm(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Converts the retrieved JSON results into a Markdown/XML mixed format best understood by LLM.  
        (This method can be optionally called by API routers or other Agent systems)
        """
        if not search_results:
            return "No relevant knowledge (documents) available."

        context_blocks = []
        for i, res in enumerate(search_results, start=1):
            meta = res["metadata"]
            source = meta.get("title", f"Document_{meta.get('doc_id')}")
            
            # LLM recognizes text enclosed in XML tags (<doc>) as the clearest 'referencing context'.
            block = (
                f"<doc id=\"{i}\" source=\"{source}\" "
                f"url=\"{meta.get('url', 'N/A')}\" "
                f"relevance_score=\"{res['score']}\">\n"
                f"{res['text']}\n"
                f"</doc>"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)