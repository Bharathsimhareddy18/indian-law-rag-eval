from app.services.bm25_retriever import bm25_retrieve
from app.services.faiss_retriever import faiss_retrieve
from app.services.hybrid_retriever import hybrid_retrieve
from typing import List, Dict, Any
from openai import AsyncOpenAI
from app.core.config import settings
import openai
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@retry(
    wait=wait_random_exponential(min=1, max=60),
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
async def chat(history: List[Dict[str, Any]], query: str, client: AsyncOpenAI, bm25_global, faiss_global, corpus_global, model_global):

    logger.info(f"Routing query: '{query}'")
    
    # --- 1. Routing Phase ---
    try:
        tool_decision = await client.chat.completions.create(
            model=settings.CHAT_MODEL_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": """You are an intelligent legal router and query expander. 
                    
                    Follow these strict steps:
                    1. Read the conversation history to resolve any pronouns or vague references in the user's latest query.
                    2. Transform the user's query into a 'Hypothetical Document'. Rewrite the intent so it reads exactly like a formal Indian legal statute.
                    3. Select the best retrieval tool and pass this hypothetical legal document text into the tool's 'query' parameter.
                    4. If the user's query is just a greeting or general wish, DO NOT use any tool."""
                },
                {"role": "user", "content": f"History:\n{history}\n\nLatest query: {query}"}
            ],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "bm25_retrieve",
                        "description": "Use this tool for keyword-based search. Best for exact terminology, legal section numbers, or specific names.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The specific search query."
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "The number of top results to return. Default is 5."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "faiss_retrieve",
                        "description": "Use this tool for semantic vector search. Best for conceptual similarity, situational queries, or general legal principles.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The specific search query."
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "The number of top results to return. Default is 5."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "hybrid_retrieve",
                        "description": "Use this tool for a combination of keyword and semantic search. Best for complex queries needing both exact matches and conceptual context.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The specific search query."
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "The number of top results to return. Default is 5."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }
            ],
            tool_choice="auto"
        )
    except Exception as e:
        logger.error(f"Critical failure during LLM tool routing: {e}", exc_info=True)
        raise # Let Tenacity retry this if it's an OpenAI error

    context = " "
    tool_selection = "none"
    
    # --- 2. Tool Parsing & Execution Phase ---
    if tool_decision.choices[0].message.tool_calls:
        tool_call = tool_decision.choices[0].message.tool_calls[0]
        tool_selection = tool_call.function.name
        
        # Guard against LLM outputting malformed JSON
        try:
            arguments = json.loads(tool_call.function.arguments)
            search_query = arguments.get("query", query)
            logger.info(f"Tool '{tool_selection}' selected. Executing retrieval with query: '{search_query}'")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM tool arguments: {e}. Raw output: {tool_call.function.arguments}")
            arguments = {}
            search_query = query
    
        # Guard against Vector DB crashes
        try:
            if tool_selection == 'bm25_retrieve':
                context = bm25_retrieve(search_query, bm25=bm25_global, corpus=corpus_global)
            elif tool_selection == 'faiss_retrieve':
                context = faiss_retrieve(search_query, index=faiss_global, corpus=corpus_global, model=model_global)
            elif tool_selection == 'hybrid_retrieve':
                context = hybrid_retrieve(search_query, bm25=bm25_global, index=faiss_global, corpus=corpus_global, model=model_global)
            logger.info(f"Successfully retrieved context using {tool_selection}.")
        except Exception as e:
            logger.error(f"Vector DB retrieval failed during {tool_selection}: {e}", exc_info=True)
            context = " " # Force empty context so generation doesn't break
    else:
        logger.info("No tool selected by router. Bypassing retrieval.")

    # --- 3. Generation Phase ---
    logger.info("Initiating LLM response stream.")
    try:
        response_stream = await client.chat.completions.create(
            model=settings.CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": """You are an expert Indian Legal Assistant. 
        Use the provided context to give a professional, detailed answer.
        
        Structure your response as follows:
        1. **Direct Answer**: Start with a clear 1-2 sentence summary.
        2. **Key Sections**: Use a bulleted list to highlight relevant IPC sections. **Bold** the section numbers.
        3. **Legal Context**: A paragraph explaining how these apply or any procedural notes.
        
        If the user's query is just a greeting or wish, ignore the structure above. Simply introduce yourself as an Indian Legal Assistant and ask how you can help."""},
                {"role": "user", "content": query},
                {"role": "system", "content": f"History:\n{history}"},
                {"role": "system", "content": f"Context:\n{context}"}
            ],
            stream=True
        )
    except Exception as e:
        logger.error(f"Critical failure during LLM stream generation: {e}", exc_info=True)
        raise

    return context, tool_selection, response_stream