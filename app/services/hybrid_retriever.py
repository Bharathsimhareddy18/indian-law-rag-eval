import logging
from typing import List, Dict, Any
from app.services.bm25_retriever import bm25_retrieve
from app.services.faiss_retriever import faiss_retrieve

logger = logging.getLogger(__name__)

K = 60  # Standard RRF constant

def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    bm25=None,
    index=None,
    corpus: List[str] = None,
    model=None
) -> List[Dict[str, Any]]:

    try:
        bm25_results  = bm25_retrieve(query, top_k, bm25, corpus)
        faiss_results = faiss_retrieve(query, top_k, index, corpus, model)

        seen = {}

        # Store BM25 rank and content for each doc
        for result in bm25_results:
            doc_id = result["doc_id"]
            seen[doc_id] = {
                "doc_id": doc_id,
                "bm25_rank": result["rank"],   # use rank, not score
                "faiss_rank": None,            # unknown yet
                "content": result["content"]
            }

        # Store FAISS rank, merge if already seen
        for result in faiss_results:
            doc_id = result["doc_id"]
            if doc_id in seen:
                seen[doc_id]["faiss_rank"] = result["rank"]
            else:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "bm25_rank": None,         # not in BM25 results
                    "faiss_rank": result["rank"],
                    "content": result["content"]
                }

        # Compute RRF score for each doc
        # If a doc is missing from one retriever, penalize it with top_k + 1
        results = []
        for doc in seen.values():
            bm25_rank  = doc["bm25_rank"]  if doc["bm25_rank"]  is not None else (top_k + 1)
            faiss_rank = doc["faiss_rank"] if doc["faiss_rank"] is not None else (top_k + 1)

            rrf_score = (1 / (K + bm25_rank)) + (1 / (K + faiss_rank))

            results.append({
                "doc_id": doc["doc_id"],
                "rrf_score": round(rrf_score, 6),
                "bm25_rank": doc["bm25_rank"],
                "faiss_rank": doc["faiss_rank"],
                "content": doc["content"]
            })

        # Sort by RRF score descending, take top_k
        results = sorted(results, key=lambda x: x["rrf_score"], reverse=True)[:top_k]

        # Assign final rank after sorting
        for rank, doc in enumerate(results, start=1):
            doc["rank"] = rank

        return results

    except Exception as e:
        logger.error(f"Hybrid retrieval merging failed for query '{query}': {e}", exc_info=True)
        return []