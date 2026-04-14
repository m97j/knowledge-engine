# storage/sqlite_client.py

import sqlite3
from typing import Any, Dict, List

from core.exceptions import DatabaseError
from core.logger import setup_logger

logger = setup_logger("sqlite_client")

class SQLiteStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row 
            logger.info(f"✅ Connected to SQLite at {self.db_path}")
        except sqlite3.Error as e:
            logger.critical(f"❌ SQLite connection failed: {e}")
            raise DatabaseError(f"Database connection failed: {e}")

    def get_enriched_chunks_dict(self, chunk_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Given a list of chunk_ids, retrieves the corresponding text and metadata from the SQLite database.
         - This is designed for O(1) access in the search service, where we need to quickly map chunk_ids from Qdrant results to their full text and metadata for reranking and final response construction.
         - The returned dictionary is structured as { chunk_id: { "text": "...", "metadata": {...} } }, allowing for efficient lookups during the search pipeline.
         - The SQL query uses a JOIN to combine data from the chunks and documents tables, ensuring we get all necessary information in a single query for performance optimization.
         - If the list of chunk_ids is empty, it returns an empty dictionary immediately to avoid unnecessary database queries.
         - Error handling is included to catch and log any database issues that arise during query execution.
        """
        if not chunk_ids:
            return {}

        placeholders = ",".join("?" * len(chunk_ids))
        
        query = f"""
            SELECT 
                c.chunk_id, c.text AS chunk_text,
                d.doc_id, d.title, d.lang, d.url, d.date_modified
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN ({placeholders})
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(query, chunk_ids)
            rows = cur.fetchall()
            
            # Transform the result into a dictionary for O(1) access: { chunk_id: { "text": "...", "metadata": {...} } }
            result_dict = {}
            for row in rows:
                result_dict[row["chunk_id"]] = {
                    "text": row["chunk_text"],
                    "metadata": {
                        "doc_id": row["doc_id"],
                        "title": row["title"],
                        "lang": row["lang"],
                        "url": row["url"],
                        "date_modified": row["date_modified"]
                    }
                }
            return result_dict
            
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch enriched chunks: {e}")
            raise DatabaseError(f"Query execution failed: {e}")

    def close(self):
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
            logger.info("🛑 SQLite connection closed.")