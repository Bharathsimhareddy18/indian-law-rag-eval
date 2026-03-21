import logging
import openai
from openai import AsyncOpenAI
from app.core.config import settings
from app.utils.guardrail import is_safe_sql
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

# --- 1. Retry-Protected Helper for LLM Call ---
@retry(
    wait=wait_random_exponential(min=1, max=10),
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
async def _generate_sql_with_retry(user_prompt: str, ai_client: AsyncOpenAI, schema_definition: str) -> str:
    logger.info("Generating SQL via LLM...")
    sql_response = await ai_client.chat.completions.create(
        model=settings.CHAT_MODEL_NAME, 
        messages=[
            {"role": "system", "content": f"You are a PostgreSQL expert. Convert the user query into a valid read-only SELECT statement. Output ONLY the raw sql string without any markdown formatting, backticks, explanations, or trailing semicolons.\n\nSchema:\n{schema_definition}"},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    # Strip whitespace AND any trailing semicolons to prevent subquery crashes
    return sql_response.choices[0].message.content.strip().rstrip(';')

# --- 2. Main Service Flow ---
async def process_text_to_sql(user_prompt: str, ai_client: AsyncOpenAI, supabase_client):
    """Handles the conversion of natural language to SQL and executes it."""
    
    schema_definition = """
    Table: llm_observability
    Columns: id (uuid), created_at (timestamp), user_query (text), retrieved_context (text), llm_response (text), goal_completion (float), correctness_score (float), faithfulness_score (float), struggle_metric (float), context_relevance_score (float), tool_used (text), latency_ms (integer).
    """
    
    # Step 1: Generate SQL (Protected by Tenacity)
    try:
        generated_sql = await _generate_sql_with_retry(user_prompt, ai_client, schema_definition)
    except Exception as e:
        logger.error(f"LLM SQL Generation failed after all retries: {e}", exc_info=True)
        return {"error": "Failed to generate SQL due to API issues.", "details": str(e)}

    # Step 2: Guardrail validation
    if not is_safe_sql(generated_sql):
        logger.warning(f"Blocked unsafe SQL attempt: {generated_sql}")
        return {"error": "Query blocked by security guardrails. Only SELECT operations are permitted."}

    # Step 3: Execution via Supabase RPC (No retries on DB execution)
    try:
        logger.info(f"Executing SQL via RPC: {generated_sql}")
        response = supabase_client.rpc("run_raw_sql", {"query": generated_sql}).execute()
        raw_data = response.data
    except Exception as e:
        logger.error(f"Database RPC execution failed: {e}", exc_info=True)
        return {"error": "Failed to execute query on database. The SQL might be invalid.", "generated_sql": generated_sql}

    return {
        "generated_sql": generated_sql,
        "results": raw_data
    }