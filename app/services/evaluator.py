import json
from openai import AsyncOpenAI
from app.models.models import EvaluationResult
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import openai
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.APIStatusError
    )),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def evaluate_retriever(query: str, answer: str, context: str, client: AsyncOpenAI):
    
    logger.info(f"Starting LLM evaluation for query: '{query}'")
    
    eval_prompt = f"""
    You are an impartial AI judge evaluating a RAG system's response.
    
    User Query: {query}
    Retrieved Context: {context}
    System Answer: {answer}
    
    Evaluate the response on a scale of 1 to 5 based on the provided schema.
    """
    
    try:
        logger.info("Sending evaluation prompt to OpenAI...")
        # Use .parse() for native Pydantic support
        response = await client.beta.chat.completions.parse(
            model=settings.EVAL_MODEL_NAME, # Note: use a model that supports structured outputs
            messages=[{"role": "system", "content": eval_prompt}],
            response_format=EvaluationResult,
        )
        
        # .parsed is already a validated Pydantic object; convert to dict
        scores = response.choices[0].message.parsed.model_dump()
        logger.info(f"Evaluation successful. Scores: {scores}")
        
        return scores
        
    except openai.OpenAIError as e:
        logger.error(f"Critical OpenAI API failure during evaluation after retries: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation parsing: {e}", exc_info=True)
        
    # --- The Fallback Safety Net ---
    # If the LLM completely fails, return 0s so Supabase still logs the interaction
    # without crashing the background task.
    logger.warning("Returning default fallback scores (0) due to evaluation failure.")
    return {
        "goal_completion": 0,
        "correctness_score": 0,
        "faithfulness_score": 0,
        "struggle_metric": 0,
        "context_relevance_score": 0
    }