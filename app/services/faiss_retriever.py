import numpy as np
import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer # Swapped OpenAI for SentenceTransformer

from app.data_client.faiss_client import faiss_client
from app.data_client.data_loader import dataLoader

logger = logging.getLogger(__name__)

def faiss_retrieve(
    query: str,
    top_k: int = 5,
    index=None,
    corpus: List[str] = None,
    model: SentenceTransformer = None, # Update type hint
) -> List[Dict[str, Any]]:
    
    try:
        # Initialization
        if index is None:
            index = faiss_client()
        if corpus is None:
            corpus = dataLoader()
        if model is None:
            model = SentenceTransformer('all-MiniLM-L6-v2') # Fallback to local model
            
        # 1. Generate embedding using local model
        # We pass the query as a list and enforce float32 for FAISS compatibility
        query_vector = model.encode([query], normalize_embeddings=True).astype('float32')
        
        # Search top-k nearest neighbours
        scores, indices = index.search(query_vector, top_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            results.append({
                "rank": rank,
                "doc_id": int(idx),
                "score": round(float(score), 4),
                "content": corpus[idx]
            })

        return results
        
    except Exception as e:
        logger.error(f"FAISS retrieval failed for query '{query}': {e}", exc_info=True)
        return []