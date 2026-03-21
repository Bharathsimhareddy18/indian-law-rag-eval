import os
import logging
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from app.services.evaluator import evaluate_retriever
from app.core.config import settings

load_dotenv()

logger = logging.getLogger(__name__)

# Retry logic strictly for the DB insert
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)

async def evaluate_store_by_llm(query: str, answer: str, context: str, client: AsyncOpenAI, tool_selection: str, latency_ms: int,supabase_client: Client):
    
    try:
        # LLM Evaluation (This has its own retry logic inside evaluator.py)
        scores = await evaluate_retriever(query, answer, context, client)
        
        payload = {
            "user_query": query,
            "retrieved_context": context,
            "llm_response": answer,
            "goal_completion": scores.get("goal_completion", 0),
            "correctness_score": scores.get("correctness_score", 0),
            "faithfulness_score": scores.get("faithfulness_score", 0),
            "struggle_metric": scores.get("struggle_metric", 0),
            "tool_used": tool_selection,
            "context_relevance_score": scores.get("context_relevance_score", 0),
            "latency_ms": latency_ms
        }
        
        # Store in DB with retries
        supabase_client.table("llm_observability").insert(payload).execute()
        logger.info("Evaluation metrics successfully inserted into Supabase.")
        
        return {"message": "Evaluation stored successfully"}
        
    except Exception as e:
        logger.error(f"Failed to complete evaluation background task: {e}", exc_info=True)
        return {"message": "Failed to store evaluation", "error": str(e)}