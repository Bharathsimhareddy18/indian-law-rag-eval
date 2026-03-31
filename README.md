# Indian Law Assistant — Hybrid RAG + LLMOps

A production RAG system for querying Indian Penal Code (IPC) and 
Bharatiya Nyaya Sanhita (BNS) with full LLMOps observability.

Live (API): https://bharath-reddy-ai-indian-law-rag-eval.hf.space/docs  
UI: Under Development

---

## System Architecture

<img width="1982" height="2160" alt="LLM Retrieval Pipeline with-2026-03-23-173100 (2)" src="https://github.com/user-attachments/assets/8385cdf0-5cdd-495e-80d5-5903f87e9005" />


---

## What It Does

A legal query system that routes every question through the optimal 
retrieval strategy, evaluates every response automatically, and stores 
full telemetry for continuous improvement.

**Retrieval Router** — Uses HyDE (Hypothetical Document Embeddings) to 
rewrite the user query as a formal legal statute before retrieval. 
LLM selects between three retrieval strategies per query:

- BM25 — keyword search, best for exact section numbers and legal terms
- FAISS — semantic vector search, best for conceptual queries
- Hybrid — RRF scoring combining both, best for complex queries

**LLMOps Observability** — Every query triggers a background evaluator 
(fire and forget — zero latency impact) that scores the response on 
5 metrics via LLM-as-Judge and stores results to Supabase:

| Metric | Description |
|---|---|
| Goal Completion | Did the response answer the query |
| Correctness | Factual accuracy against retrieved context |
| Faithfulness | Response grounded in context, not hallucinated |
| Context Relevance | Quality of retrieved documents |
| Struggle Metric | Signs of model uncertainty |

**IntelliSQL** — Natural language queries against the observability 
database via /text_to_sql endpoint. Read-only SQL guardrails enforced 
at regex token level.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Streaming Responses |
| Retrieval | BM25, FAISS, Sentence Transformers |
| AI | OpenAI, LLM-as-Judge |
| Reliability | Tenacity retry logic |
| Database | Supabase (PostgreSQL) |
| Infrastructure | Hugging Face Spaces, Docker |

---

## API Endpoints

| Endpoint | Description |
|---|---|
| POST /chatbot | Legal query with streaming response |
| GET /observability | Full telemetry and evaluation logs |
| POST /text_to_sql | Natural language database queries |

---
## Screenshots

<img width="1918" height="1006" alt="image" src="https://github.com/user-attachments/assets/8931c5f1-8a5c-4e5d-a0d2-3b43b75420a4" />

## Dataset

IPC and BNS transformation dataset — nandhakumarg/IPC_and_BNS_transformation
