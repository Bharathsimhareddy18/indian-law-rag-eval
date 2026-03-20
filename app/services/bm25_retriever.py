import numpy as np
from typing import List, Dict, Any

from app.data_client.bm25_client import bm25_client
from app.data_client.data_loader import dataLoader

def bm25_retrieve(
    query: str,
    top_k: int = 5,
    bm25=None,
    corpus: List[str] = None
) -> List[Dict[str, Any]]:
    
    
    #initilization
    if bm25 is None:
        bm25 = bm25_client()

    if corpus is None:
        corpus = dataLoader()
        
        
    # bm25 retrive logic of top 5 data    
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_k_indices = np.argsort(scores)[::-1][:top_k]
    
    #stoing top k data into results to return it
    results = []
    for rank, idx in enumerate(top_k_indices, start=1):
        results.append({
            "rank": rank,
            "doc_id": int(idx),
            "score": round(float(scores[idx]), 4),
            "content": corpus[idx]
        })

    return results