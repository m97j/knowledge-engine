# models/reranker.py

from typing import Any, Dict, List

import torch
from FlagEmbedding import FlagReranker

from core.exceptions import ModelLoadError
from core.logger import setup_logger

logger = setup_logger("reranker")

class TextReranker:
    """
    Using the BGE-Reranker model, the documents retrieved in the first search are reordered (Cross-Encoding) by comparing them with the query.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = False):
        self.model_name = model_name
        self.device = self._get_device()
        self._warmup()
        
        try:
            logger.info(f"⏳ Loading Reranker Model: {self.model_name} on {self.device}")
            self.reranker = FlagReranker(
                self.model_name, 
                use_fp16=(use_fp16 and self.device.startswith("cuda"))
            )
            logger.info("✅ Reranker Model loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to load Reranker Model: {e}", exc_info=True)
            raise ModelLoadError(f"Reranker initialization failed: {e}")

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _warmup(self):
        logger.info("Warming up reranker model with a dummy input.")
        self.rerank(query="Hello world", documents=[{"text": "Hello world"}])

    def rerank(self, query: str, documents: List[Dict[str, Any]], text_key: str = "text") -> List[Dict[str, Any]]:
        """
        Takes a list of documents as input, recalculates their similarity to the query, and returns the results sorted by score.  

        :param query: The original search query string
        :param documents: A list of dictionaries in the form [{'chunk_id': 1, 'text': '...'}, ...]
        :param text_key: The key name in the document dictionary containing the body text
        """
        if not documents:
            return []

        # Generate pairs for Cross-Encoder input: [[query, doc1], [query, doc2], ...]
        sentence_pairs = [[query, doc[text_key]] for doc in documents]

        try:
            # 1. Batch score calculation
            scores = self.reranker.compute_score(sentence_pairs, normalize=True)
            
            # Wrap in a list because compute_score can return a float when there is only one input document
            if isinstance(scores, float):
                scores = [scores]

            # 2. Inject rerank_score into source document dictionarys
            for i, doc in enumerate(documents):
                doc["rerank_score"] = float(scores[i])

            # 3. Sort by score (descending)
            reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed for query '{query}': {e}")
            raise RuntimeError(f"Reranking process failed: {e}")