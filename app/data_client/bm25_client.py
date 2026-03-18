from rank_bm25 import BM25Okapi
from app.data_client.data_loader import dataLoader

def bm25_client():
    
    content = dataLoader()
    tokenized_data = [doc.lower().split() for doc in content]
    bm25 = BM25Okapi(tokenized_data)
    
    return bm25 
