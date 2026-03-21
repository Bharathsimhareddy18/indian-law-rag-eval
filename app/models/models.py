from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: List[Message] = []

class EvaluationResult(BaseModel):
    goal_completion: int = Field(ge=1, le=5)
    correctness_score: int = Field(ge=1, le=5)
    faithfulness_score: int = Field(ge=1, le=5)
    struggle_metric: int = Field(ge=1, le=5)
    context_relevance_score: int = Field(ge=1, le=5)
    
class MetricsQuery(BaseModel):
    user_prompt: str