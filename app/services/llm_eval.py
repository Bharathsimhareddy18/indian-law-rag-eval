import json
import os
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from app.services.evaluator import evaluate_retriever

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def evaluate_store_by_llm(query: str, answer: str, context: str, client: AsyncOpenAI, tool_selection: str):
    
    
    scores = await evaluate_retriever(query, answer, context, client)
    
    data = supabase.table("llm_observability").insert({
            "user_query": query,
            "retrieved_context": context,
            "llm_response": answer,
            "goal_completion": scores.get("goal_completion", 0),
            "correctness_score": scores.get("correctness_score", 0),
            "faithfulness_score": scores.get("faithfulness_score", 0),
            "struggle_metric": scores.get("struggle_metric", 0),
            "tool_used": tool_selection
        }).execute()
    
    
    
    return {"message": "Evaluation stored successfully", "data": data.data}