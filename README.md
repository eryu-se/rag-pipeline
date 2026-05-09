# RAG Pipeline – Cloud Edition

4-agent RAG pipeline built with:
- **Groq** – ultra-fast LLM inference (free tier available)
- **Qdrant Cloud** – managed vector database (free 1GB cluster)
- **Streamlit Cloud** – free hosting

## Architecture

```
PDF/Text upload
     ↓
Sentence Transformers (embeddings, runs locally in Streamlit Cloud)
     ↓
Qdrant Cloud (vector storage)
     ↓
Query → 4-agent pipeline via Groq LLM:
  1. RETRIEVE  – rewrites query, semantic search
  2. ANALYZE   – extracts key info from chunks
  3. SYNTHESIZE – formulates structured answer
  4. CRITIC    – fact-checks and improves answer
```

## Local development

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add API keys to `.streamlit/secrets.toml` (see template).

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Under **Settings → Secrets**, add:

```toml
GROQ_API_KEY   = "gsk_..."
QDRANT_URL     = "https://your-cluster.qdrant.io:6333"
QDRANT_API_KEY = "your_key"
```

5. Click **Deploy** – done.

## Get free API keys

| Service | URL | Free tier |
|---|---|---|
| Groq | console.groq.com | 14,400 req/day |
| Qdrant Cloud | cloud.qdrant.io | 1GB free cluster |
| Streamlit Cloud | share.streamlit.io | Free public apps |
