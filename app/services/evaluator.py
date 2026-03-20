import json
from openai import AsyncOpenAI

async def evaluate_retriever(query: str, answer: str, context: str, client: AsyncOpenAI):
    
    eval_prompt = f"""
    You are an impartial AI judge evaluating a RAG system's response.
    
    User Query: {query}
    Retrieved Context: {context}
    System Answer: {answer}
    
    Evaluate the response on a scale of 1 to 5 for the following metrics:
    1. goal_completion: Did the answer fully satisfy the user's intent? (5 = Perfect)
    2. correctness_score: Is the answer factually accurate based on standard knowledge? (5 = Completely correct)
    3. faithfulness_score: Is the answer derived strictly from the retrieved context? (5 = Perfectly faithful, no hallucinations)
    4. struggle_metric: Did the system seem to struggle? (e.g., apologizing, failing to find context, giving partial answers). (1 = Confident/No struggle, 5 = Failed/Struggled heavily)
    5. context_relevance_score: How relevant and useful was the retrieved context for answering the user's query? (1 = Completely irrelevant/useless, 5 = Perfectly relevant and contained the exact information needed)
    
    Return ONLY a valid JSON object:
    {{"goal_completion": int, "correctness_score": int, "faithfulness_score": int, "struggle_metric": int}}
    """
    
    
    response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": eval_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1 
        )
    
    scores = json.loads(response.choices[0].message.content)
    
    return scores