from rank_bm25 import BM25Okapi
from app.data_client.data_loader import dataLoader
from pathlib import Path
import pickle 
from app.core.config import settings


BM25_CACHE = settings.BM25_INDEX_PATH

def bm25_client():
    
    # load if already exists
    if BM25_CACHE.exists():
        print("Loading BM25 index from cache...")
        with open(BM25_CACHE, "rb") as f:
            return pickle.load(f)
    
    # building the BM25 index
    print("creating the BM25 index")
    content = dataLoader()
    tokenized_data = [doc.lower().split() for doc in content]
    bm25 = BM25Okapi(tokenized_data)
    
    #storing the BM25 index
    BM25_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CACHE, "wb") as f:
        pickle.dump(bm25, f)
    
    return bm25 
