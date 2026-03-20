from typing import List, Dict, Any

from app.services.bm25_retriever import bm25_retrieve
from app.services.faiss_retriever import faiss_retrieve

def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    bm25=None,
    index=None,
    corpus: List[str] = None,
    model=None
) -> List[Dict[str, Any]]:

    bm25_results  = bm25_retrieve(query, top_k, bm25, corpus)
    faiss_results = faiss_retrieve(query, top_k, index, corpus, model)

    # merge by doc_id, avoid duplicates
    seen = {}

    for result in bm25_results:
        doc_id = result["doc_id"]
        seen[doc_id] = {
            "doc_id": doc_id,
            "bm25_score": result["score"],
            "faiss_score": 0.0,
            "content": result["content"]
        }

    for result in faiss_results:
        doc_id = result["doc_id"]
        if doc_id in seen:
            seen[doc_id]["faiss_score"] = result["score"]
        else:
            seen[doc_id] = {
                "doc_id": doc_id,
                "bm25_score": 0.0,
                "faiss_score": result["score"],
                "content": result["content"]
            }

    # assign rank
    results = []
    for rank, doc in enumerate(seen.values(), start=1):
        results.append({
            "rank": rank,
            "doc_id": doc["doc_id"],
            "bm25_score": doc["bm25_score"],
            "faiss_score": doc["faiss_score"],
            "content": doc["content"]
        })

    return results