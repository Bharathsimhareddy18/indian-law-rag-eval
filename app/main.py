import sys
import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi import APIRouter
from fastapi import Body
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from app.data_client.bm25_client import bm25_client
from app.data_client.faiss_client import faiss_client
from app.services.bm25_retriever import bm25_retrieve
from app.services.faiss_retriever import faiss_retrieve
from app.services.hybrid_retirever import hybrid_retrieve
from app.utils.pre_processing import pre_process
from openai import AsyncClient
from dotenv import load_dotenv

load_dotenv()

bm25_global = None
faiss_global = None
model_global = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_global, faiss_global, model_global, AI_client
    

    
    print("Loding clinets...")
    bm25_global =  bm25_client()
    faiss_global =  faiss_client()
    model_global = SentenceTransformer('all-MiniLM-L6-v2')
    AI_client = AsyncClient()
    
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
    
@app.post("/chat")
async def chat(query: str, history: List[Dict[str, Any]] = Body(default=[])):
    
    pre_processed = await pre_process(history, query, AI_client)
    
    
    return {"results": pre_processed}