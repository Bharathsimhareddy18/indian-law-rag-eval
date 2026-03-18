import sys
import os
from pathlib import Path
from fastapi import FastAPI
from fastapi import APIRouter
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from app.data_client.bm25_client import bm25_client
from app.data_client.faiss_client import faiss_client

bm25_global = None
faiss_global = None
model_global = None 




@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_global, faiss_global, model_global
    

    
    print("Loding clinets...")
    bm25_global =  bm25_client()
    faiss_global =  faiss_client()
    model_global = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("clients loaded successfully...")
    yield
    print("closing clinets...")
    
    
app = FastAPI(lifespan=lifespan)

@app.get("/")
def home():
    return {
        "Rag-Eval": "V1",
        "clients":"loaded",
        "server": "Healty"
    }