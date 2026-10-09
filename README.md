# AI Banking Support Chatbot

A demo RAG application built with FastAPI, ChromaDB, SentenceTransformers, and Groq. Upload PDF/TXT banking reference documents, ask questions grounded in those documents, and revisit persistent conversation history.

## Features
- PDF and TXT ingestion
- Chunking and semantic embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Persistent ChromaDB vector store
- Groq chat completion for grounded answers
- SQLite chat history and session list
- Responsive web interface
- Health endpoint at `/health`

## Run locally (Windows PowerShell)

1. Install Python 3.11 or 3.12.
2. Extract the ZIP and open a terminal in the project folder.
3. Create an environment and install dependencies:

   ```powershell
   py -3.12 -m venv .venv
   .venv\\Scripts\\Activate.ps1
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env`, add your Groq API key, then set the variables in PowerShell:

   ```powershell
   $env:GROQ_API_KEY="your_real_key"
   $env:DATA_DIR="./data"
   uvicorn main:app --reload
   ```

5. Open http://127.0.0.1:8000. API documentation is at http://127.0.0.1:8000/docs.

The app reads environment variables from the shell. The `.env.example` file is a reference template, not automatically loaded. You can test retrieval by uploading `sample_banking_faq.txt` through the interface; it contains fictional demonstration policies only.

## Deploy to Railway

1. Create a GitHub repository named `ai-banking-support-chatbot`.
2. Upload the contents of this folder to the repository root (not the containing folder).
3. In Railway, create a project and deploy from that GitHub repository.
4. Add the variable `GROQ_API_KEY` in Railway's service Variables tab. You can also set `GROQ_MODEL=llama-3.3-70b-versatile`.
5. Add a Railway Volume to the service with mount path `/data`; set `DATA_DIR=/data`.
6. The included `railway.toml` sets the start command. If needed, use `uvicorn main:app --host 0.0.0.0 --port $PORT`. A Dockerfile is also included for container-based deployment.
7. Generate a public domain under service Settings → Networking.
8. Visit `/health` and `/docs` to verify the deployment.

The first request that uses embeddings may take time because the SentenceTransformer model downloads on first use. Railway volume data persists across deployments when mounted at `/data`.

## Important limitations
- This is a portfolio/demo application, not a production banking system.
- Do not upload customer records, account numbers, card details, or confidential documents.
- Add authentication, upload authorization, rate limiting, monitoring, and stronger privacy controls before exposing it for real users.
- The reported ~35% improvement from the original resume description is not measured by this rebuilt code. To claim that metric, run and document a reproducible evaluation against a keyword-search baseline.
