from openai import AsyncOpenAI
from pydantic import BaseModel
import json


async def pre_process(messages: list, query: str, client: AsyncOpenAI):

    # no history, return first message as-is
    response = await client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "system",
                "content": """You are a query pre-processor for a legal RAG system.
                Given a conversation history and the latest user message, return a JSON with:
                1. standalone_query: rephrase the latest message into a self-contained search query
                2. history_summary: 1-2 sentence summary of the conversation so far

                Return only valid JSON:
                 {
                 "standalone_query": "...",
                 "history_summary": "..."
                }"""
            },
            {
                "role": "user",
                "content": f"History:\n{messages[:-1]}\n\nLatest message: {query}"
            }
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content) 

    return {
        "standalone_query": data.get("standalone_query", query),
        "history_summary": data.get("history_summary", "")
    }