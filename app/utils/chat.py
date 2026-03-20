from openai import AsyncOpenAI
from pydantic import BaseModel
import json


async def chat(history: str, query: str, client: AsyncOpenAI, bm25_retrieve, faiss_retrieve, hybrid_retrieve):

    tool_decision = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"History:\n{history}\n\nYou are an intelligent router. Based on the user's latest query, select the appropriate retrieval tool."},
            {"role": "user", "content": query}
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
    
    if tool_decision.choices[0].message.tool_calls:
        tool_call = tool_decision.choices[0].message.tool_calls[0]
        tool_selection = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        search_query = arguments.get("query", query)
    
        if tool_selection == 'bm25_retrieve':
            context=bm25_retrieve(search_query)
        elif tool_selection == 'faiss_retrieve':
            context=faiss_retrieve(search_query)
        elif tool_selection == 'hybrid_retrieve':
            context=hybrid_retrieve(search_query)
    
    
    reponse = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query},
                  {"role": "system", "content": f"History:\n{history}"},
                  {"role": "system", "content": f"Context:\n{context}"},
                  {"role": "system", "content": "Based on the above history, query, and retrieved context, generate a concise answer to the user's query. If the context does not contain relevant information, generate an answer based on the query and history alone. if you have correct answer or yu dont know what user is asking then say you dont know."}
                  ]
        )
    
    answer = reponse.choices[0].message.content
    
    
    return {"answer": answer, "retrieved_context": context, "tool_used": tool_selection}