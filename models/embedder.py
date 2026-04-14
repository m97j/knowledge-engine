# models/embedder.py

from typing import Any, Dict, List

import torch
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel

from core.exceptions import ModelLoadError
from core.logger import setup_logger

logger = setup_logger("embedder")

# Data structure for return (Type Hinting)
class EmbedderResult(BaseModel):
    dense_vector: List[float]
    sparse_indices: List[int]
    sparse_values: List[float]

class TextEmbedder:
    """
    Converts the input text into Dense Vectors and Sparse Vectors (Lexical Weights) using the BGE-M3 model.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = False):
        self.model_name = model_name
        self.device = self._get_device()
        
        try:
            logger.info(f"⏳ Loading Embedder Model: {self.model_name} on {self.device}")
            self.model = BGEM3FlagModel(
                self.model_name, 
                use_fp16=(use_fp16 and self.device.startswith("cuda"))
            )
            self._warmup()
            logger.info("✅ Embedder Model loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ Failed to load Embedder Model: {e}", exc_info=True)
            raise ModelLoadError(f"Embedder initialization failed: {e}")

    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps" # Apple Silicon
        return "cpu"
    
    def _warmup(self):
        logger.info("Warming up embedder model with a dummy input.")
        self.encode_query("Hello world")

    def encode_query(self, text: str) -> EmbedderResult:
        """
        Converts a single query text into Qdrant hybrid search format.
        """
        try:
            # 1. model inference to get dense vector and sparse lexical weights
            output = self.model.encode(
                text, 
                return_dense=True, 
                return_sparse=True, 
                return_colbert_vecs=False
            )
            
            dense_vec = output['dense_vecs'].tolist()
            lexical_weights: Dict[str, float] = output['lexical_weights']
            
            # 2. Sparse Vector Transformation (Qdrant specifications: token_id array, weight array)
            sparse_indices = []
            sparse_values = []
            
            # Convert text tokens into unique IDs (integers) using the BGE-M3 tokenizer
            for token_str, weight in lexical_weights.items():
                # Get the ID of the string token through the tokenizer (vocab index)
                token_id = self.model.tokenizer.convert_tokens_to_ids(token_str)
                if token_id is not None:
                    sparse_indices.append(token_id)
                    sparse_values.append(float(weight))
                    
            return EmbedderResult(
                dense_vector=dense_vec,
                sparse_indices=sparse_indices,
                sparse_values=sparse_values
            )
            
        except Exception as e:
            logger.error(f"Failed to encode query '{text}': {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    # options for batch encoding if needed in the future
    def encode_documents(self, texts: List[str], batch_size: int = 12) -> Dict[str, Any]:
        return self.model.encode(
            texts, 
            batch_size=batch_size, 
            max_length=8192, # BGE-M3's max token length
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=False
        )