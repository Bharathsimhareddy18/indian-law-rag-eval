import numpy as np
from typing import List, Dict, Any

from app.data_client.faiss_client import faiss_client
from app.data_client.data_loader import dataLoader
from sentence_transformers import SentenceTransformer


def faiss_retrieve(
    query: str,
    top_k: int = 5,
    index=None,
    corpus: List[str] = None,
    model: SentenceTransformer = None
) -> List[Dict[str, Any]]:
    
    # initilization
    if index is None:
        index = faiss_client()

    if corpus is None:
        corpus = dataLoader()

    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
    
    # encode query to vector, same model used during index build
    query_vector = model.encode([query], normalize_embeddings=True).astype('float32')
    
    # search top-k nearest neighbours
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

