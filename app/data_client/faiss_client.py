import faiss
from sentence_transformers import SentenceTransformer
from app.data_client.data_loader import dataLoader

def faiss_client():
    
    model = SentenceTransformer('all-MiniLM-L6-v2') #dim 384
    content = dataLoader()
    embeddings = model.encode(content,batch_size=32, show_progress_bar=True)
    
    #index
    dimension = 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    
    return index 