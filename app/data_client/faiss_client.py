import faiss
from sentence_transformers import SentenceTransformer
from app.data_client.data_loader import dataLoader
from pathlib import Path
from app.core.config import settings

FAISS_CACHE = settings.FAISS_INDEX_PATH

def faiss_client():
    
    # loading if already exists
    if FAISS_CACHE.exists():
        print("Loading FAISS index from cache...")
        return faiss.read_index(str(FAISS_CACHE))
    
    # building the faiss index
    model = SentenceTransformer('all-MiniLM-L6-v2') #dim 384
    content = dataLoader()
    embeddings = model.encode(content,batch_size=32, show_progress_bar=True)
    
    #index
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    #saving the index
    FAISS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_CACHE))

    print(f"FAISS index cached to {FAISS_CACHE}")
    
    return index 