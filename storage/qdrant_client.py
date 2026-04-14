# storage/qdrant_client.py

from typing import List

from qdrant_client import QdrantClient, models

from core.exceptions import DatabaseError
from core.logger import setup_logger

logger = setup_logger("qdrant_client")

class QdrantStorage:
    """
    Qdrant client performing hybrid search based on dense and sparse vectors
    """
    def __init__(self, path: str, collection_name: str = "knowledge_base"):
        self.path = path
        self.collection_name = collection_name
        try:
            # Local file system-based Qdrant connection (v1.10+)
            self.client = QdrantClient(path=self.path)
            logger.info(f"✅ Connected to local Qdrant at {self.path} (Collection: {self.collection_name})")
        except Exception as e:
            logger.critical(f"❌ Qdrant connection failed: {e}")
            raise e

    def hybrid_search(
        self, 
        dense_vector: List[float], 
        sparse_indices: List[int], 
        sparse_values: List[float], 
        limit: int = 100
    ) -> List[models.ScoredPoint]:
        """
        Qdrant's Native Fusion API to perform hybrid search with dense and sparse vectors.
        Calculates RRF (Reciprocal Rank Fusion) at the database level and returns the results.
        """
        try:
            # Qdrant v1.10+ Latest Syntax: Fusion processing after multiple searches using Prefetch
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    # 1. Sparse search query
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_indices, 
                            values=sparse_values
                        ),
                        using="sparse",
                        limit=limit,
                    ),
                    # 2. Dense search query
                    models.Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=limit,
                    ),
                ],
                # 3. Score merging (Fusion) of the two results above using the RRF method
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True
            )
            return results.points
            
        except Exception as e:
            logger.error(f"❌ Hybrid search failed: {e}", exc_info=True)
            raise DatabaseError(f"Qdrant Hybrid search execution failed: {e}")

    def close(self):
        """Qdrant client connection cleanup (if applicable)"""
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("🛑 Qdrant client connection closed.")
