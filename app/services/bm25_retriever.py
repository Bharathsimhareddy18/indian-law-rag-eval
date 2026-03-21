import numpy as np
import logging
from typing import List, Dict, Any
from app.data_client.bm25_client import bm25_client
from app.data_client.data_loader import dataLoader

logger = logging.getLogger(__name__)

def bm25_retrieve(
    query: str,
    top_k: int = 5,
    bm25=None,
    corpus: List[str] = None
) -> List[Dict[str, Any]]:
    
    try:
        # Initialization
        if bm25 is None:
            bm25 = bm25_client()
        if corpus is None:
            corpus = dataLoader()
            
        # BM25 retrieve logic
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # Storing top k data
        results = []
        for rank, idx in enumerate(top_k_indices, start=1):
            results.append({
                "rank": rank,
                "doc_id": int(idx),
                "score": round(float(scores[idx]), 4),
                "content": corpus[idx]
            })

        return results
        
    except Exception as e:
        logger.error(f"BM25 retrieval failed for query '{query}': {e}", exc_info=True)
        return [] # Return empty list so the pipeline doesn't crash