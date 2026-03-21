import sys
import os
import time
import json
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi import APIRouter
from fastapi import Body
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.data_client.bm25_client import bm25_client
from app.data_client.faiss_client import faiss_client
from app.data_client.data_loader import dataLoader
from app.utils.pre_processing import pre_process
from app.services.bm25_retriever import bm25_retrieve
from app.services.faiss_retriever import faiss_retrieve
from app.services.hybrid_retriever import hybrid_retrieve
from app.services.llm_eval import evaluate_store_by_llm
from app.services.llm_observability_kpi import retrive_data_logs
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.models.models import ChatRequest
from app.utils.chat import chat
from openai import AsyncOpenAI,OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from pydantic import BaseModel
from app.models.models import MetricsQuery
from app.utils.guardrail import is_safe_sql
from app.services.sql_service import process_text_to_sql

load_dotenv()

# --- Configure Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

bm25_global = None
faiss_global = None
model_global = None 
corpus_global = None
supabase_global = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bm25_global, faiss_global, model_global, AI_client, corpus_global, supabase_global
    
    SUPABASE_URL = settings.SUPABASE_URL
    SUPABASE_KEY = settings.SUPABASE_KEY
    
    logger.info("Loading clients and models...")
    supabase_global = create_client(SUPABASE_URL, SUPABASE_KEY)
    bm25_global =  bm25_client()
    faiss_global =  faiss_client()
    model_global = SentenceTransformer('all-MiniLM-L6-v2')
    AI_client = AsyncOpenAI()
    corpus_global = dataLoader()

    
    logger.info("All clients loaded successfully. App is ready to receive traffic.")
    yield
    logger.info("Shutting down app, closing clients...")
   
    
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    logger.info("Health check endpoint hit.")
    return {
        "Rag-Eval": "V1",
        "clients":"loaded",
        "server": "Healthy"
    }

# --- Safe Wrapper for Background Tasks ---
async def safe_evaluate(*args, **kwargs):
    """Wraps the evaluation task to catch and log silent background failures."""
    try:
        await evaluate_store_by_llm(*args, **kwargs)
        logger.info("Background evaluation completed and stored.")
    except Exception as e:
        logger.error(f"FATAL: Background evaluation task failed completely: {e}", exc_info=True)
    
@app.post("/chatbot")
async def chatbot(request: ChatRequest):
    
    logger.info(f"Incoming chat request. Query length: {len(request.query)}")
    start_time = time.time()
        
    history_dicts = [msg.model_dump() for msg in request.history]
        
    context, tool_selection, response_stream = await chat(
        history=history_dicts,
        query=request.query, 
        client=AI_client, 
        bm25_global=bm25_global, 
        faiss_global=faiss_global, 
        corpus_global=corpus_global,
        model_global=model_global
    )
    
    async def response_generator():
        full_answer = ""
        first_token_received = False
        ttft_ms = 0
        try:
            async for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    if not first_token_received:
                        ttft_ms = int((time.time() - start_time) * 1000)
                        logger.info(f"First token generated. TTFT: {ttft_ms}ms")
                        first_token_received = True
                    text = chunk.choices[0].delta.content
                    full_answer += text
                    yield text
          
        finally:
            logger.info("Response stream finished. Triggering background evaluation.")
            # Fire the safe background task
            asyncio.create_task(
                safe_evaluate(
                    query=request.query, # Fixed: was `query=query`
                    context=context, 
                    answer=full_answer, 
                    tool_selection=tool_selection,
                    latency_ms=ttft_ms,
                    client=AI_client,
                    supabase_client=supabase_global
                    
                )
            )
            
    return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/observability")
async def observability():
    logger.info("Fetching observability data logs.")
    data_logs = await retrive_data_logs(supabase_global)
    return data_logs

@app.post("/text_to_sql")
async def text_to_sql(request: MetricsQuery):
    # Delegate everything to the service layer
    return await process_text_to_sql(
        user_prompt=request.user_prompt,
        ai_client=AI_client,
        supabase_client=supabase_global
    )