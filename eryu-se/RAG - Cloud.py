"""
RAG Pipeline – Cloud Edition
Groq (LLM) + Qdrant Cloud (vector DB) + Streamlit Cloud

Requirements: see requirements.txt
Deploy: streamlit run app.py
"""

import os
import time
import hashlib
import streamlit as st
import fitz  # pymupdf
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)
from sentence_transformers import SentenceTransformer

# ── PAGE CONFIG ───────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root {
    --bg:#0a0a0f; --surface:#13131a; --border:#1e1e2e;
    --p:#7c6af7; --k:#6af7c8; --c:#f7826a; --y:#f7d76a;
    --text:#e8e8f0; --muted:#6b6b8a;
}
html,body,[class*="css"] { font-family:'Syne',sans-serif; background:var(--bg); color:var(--text); }
.stApp { background:var(--bg); }
section[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--border); }
.stButton>button {
    font-family:'JetBrains Mono',monospace; font-size:0.82rem; font-weight:700;
    border-radius:6px; border:1px solid var(--border); background:var(--surface);
    color:var(--text); padding:0.55rem 1rem; transition:all 0.2s; width:100%;
}
.stButton>button:hover { border-color:var(--p); color:var(--p); background:rgba(124,106,247,0.08); }
.stTextArea textarea {
    font-family:'JetBrains Mono',monospace; font-size:0.88rem;
    background:var(--surface); border:1px solid var(--border); border-radius:8px; color:var(--text);
}
.stTextArea textarea:focus { border-color:var(--p); box-shadow:0 0 0 2px rgba(124,106,247,0.15); }
.hero { padding:1.8rem 0 1.4rem; border-bottom:1px solid var(--border); margin-bottom:1.8rem; }
.hero h1 {
    font-family:'Syne',sans-serif; font-weight:800; font-size:2rem;
    letter-spacing:-0.02em; margin:0;
    background:linear-gradient(135deg,var(--p),var(--k));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p { color:var(--muted); font-size:0.82rem; margin-top:0.35rem; font-family:'JetBrains Mono',monospace; }
.result-box {
    background:var(--surface); border:1px solid var(--border);
    border-left:3px solid var(--p); border-radius:8px; padding:1.4rem;
    font-family:'JetBrains Mono',monospace; font-size:0.84rem; line-height:1.75;
    white-space:pre-wrap; word-wrap:break-word; margin-bottom:1rem;
}
.rb-retrieve   { border-left-color:var(--p); }
.rb-analyze    { border-left-color:var(--k); }
.rb-synthesize { border-left-color:var(--c); }
.rb-critic     { border-left-color:var(--y); }
.agent-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.7rem; font-weight:700;
    letter-spacing:0.1em; padding:3px 10px; border-radius:4px;
    display:inline-block; margin-bottom:0.6rem;
}
.ab-retrieve   { background:rgba(124,106,247,0.18); color:var(--p); }
.ab-analyze    { background:rgba(106,247,200,0.18); color:var(--k); }
.ab-synthesize { background:rgba(247,130,106,0.18); color:var(--c); }
.ab-critic     { background:rgba(247,215,106,0.18); color:var(--y); }
.chunk-card {
    background:var(--surface); border:1px solid var(--border); border-radius:8px;
    padding:1rem 1.2rem; margin-bottom:0.7rem;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; line-height:1.6;
}
.chunk-meta { color:var(--muted); font-size:0.7rem; margin-bottom:0.4rem; }
.doc-item {
    background:var(--surface); border:1px solid var(--border); border-radius:6px;
    padding:0.6rem 0.9rem; margin-bottom:0.4rem;
    font-family:'JetBrains Mono',monospace; font-size:0.75rem;
}
.status-ok  { color:var(--k); }
.status-err { color:var(--c); }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%;
       background:var(--k); margin-right:6px; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.25;} }
hr { border-color:var(--border); }
.stSpinner>div { border-top-color:var(--p)!important; }
.stTabs [data-baseweb="tab-list"] {
    background:var(--surface); border-radius:8px; padding:4px; gap:4px; border:1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; font-weight:700;
    letter-spacing:0.05em; color:var(--muted); border-radius:6px; padding:0.45rem 0.9rem;
}
.stTabs [aria-selected="true"] { background:rgba(124,106,247,0.15)!important; color:var(--p)!important; }
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────
COLLECTION_NAME = "rag_documents"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
EMBED_DIM       = 384
GROQ_MODELS     = {
    "Llama 3.3 70B  (best quality)":  "llama-3.3-70b-versatile",
    "Llama 3.1 8B   (fastest)":       "llama-3.1-8b-instant",
    "Mixtral 8x7B   (balanced)":      "mixtral-8x7b-32768",
    "Gemma2 9B      (efficient)":     "gemma2-9b-it",
}

# ── SESSION STATE ─────────────────────────────────────
for key, default in [
    ("history",       []),
    ("last_result",   {}),
    ("indexed_docs",  []),
    ("groq_model",    "llama-3.3-70b-versatile"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── SECRETS / API KEYS ────────────────────────────────
def get_secret(key: str, sidebar_val: str = "") -> str:
    """Read from Streamlit secrets first, fall back to sidebar input."""
    try:
        return st.secrets[key]
    except Exception:
        return sidebar_val


# ══════════════════════════════════════════════════════
# CLIENTS (cached)
# ══════════════════════════════════════════════════════
@st.cache_resource
def get_embed_model():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource
def get_qdrant(url: str, api_key: str):
    client = QdrantClient(url=url, api_key=api_key)
    # Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    return client


def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


# ══════════════════════════════════════════════════════
# DOCUMENT PROCESSING
# ══════════════════════════════════════════════════════
def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        data = uploaded_file.read()
        doc  = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    words  = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 40]


def index_document(uploaded_file, qdrant: QdrantClient, embedder) -> int:
    text   = extract_text(uploaded_file)
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    points = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        uid = int(hashlib.md5(
            f"{uploaded_file.name}_{i}_{chunk[:40]}".encode()
        ).hexdigest(), 16) % (2**63)
        points.append(PointStruct(
            id=uid,
            vector=emb,
            payload={
                "text":         chunk,
                "source":       uploaded_file.name,
                "chunk_index":  i,
                "total_chunks": len(chunks),
            }
        ))

    # Upsert in batches
    batch_size = 64
    for start in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + batch_size]
        )
    return len(chunks)


def retrieve_chunks(query: str, qdrant: QdrantClient, embedder,
                    n_results: int = 8) -> list[dict]:
    vec     = embedder.encode([query])[0].tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=n_results,
        with_payload=True,
    )
    return [
        {
            "text":        r.payload["text"],
            "source":      r.payload.get("source", "unknown"),
            "chunk_index": r.payload.get("chunk_index", 0),
            "relevance":   round(r.score, 3),
        }
        for r in results
    ]


# ══════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════
def run_llm(system: str, user: str, groq_client, max_tokens: int = 1500) -> str:
    resp = groq_client.chat.completions.create(
        model=st.session_state.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════
# AGENTS
# ══════════════════════════════════════════════════════
def agent_retrieve(question: str, qdrant, embedder, groq_client) -> tuple[list[dict], str]:
    rewrite_system = """You are a search query optimizer for a RAG system.
Rewrite the user question as 2-3 short, specific search queries for semantic search.
Return ONLY the queries separated by newlines. No numbering, no explanation."""

    rewritten = run_llm(rewrite_system, question, groq_client, max_tokens=120)
    queries   = [q.strip() for q in rewritten.strip().split("\n") if q.strip()][:3]

    seen, chunks = set(), []
    for q in queries:
        for c in retrieve_chunks(q, qdrant, embedder, n_results=4):
            if c["text"] not in seen:
                seen.add(c["text"])
                chunks.append(c)

    chunks.sort(key=lambda x: x["relevance"], reverse=True)
    return chunks[:8], "\n".join(queries)


def agent_analyze(question: str, chunks: list[dict], groq_client) -> str:
    system = """You are a precise document analyst.
Given retrieved document chunks and a question:
1. Identify which chunks are most relevant
2. Extract key facts, figures, and arguments
3. Note gaps, contradictions, or missing info
Do NOT answer the question yet – only analyze the evidence.
Be specific and reference the source chunks."""

    context = "\n\n".join(
        f"[Chunk {i+1} | {c['source']} | relevance {c['relevance']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return run_llm(system, f"Question: {question}\n\nChunks:\n{context}", groq_client)


def agent_synthesize(question: str, analysis: str, chunks: list[dict], groq_client) -> str:
    system = """You are a clear, authoritative writer.
Write a direct, well-structured answer grounded in the provided analysis and chunks.
Use specific details and numbers. Add headers for complex answers.
End with what the documents do NOT cover, if anything.
Respond in the same language as the question."""

    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks[:5]
    )
    prompt = f"Question: {question}\n\nAnalysis:\n{analysis}\n\nSource chunks:\n{context}"
    return run_llm(system, prompt, groq_client)


def agent_critic(question: str, answer: str, chunks: list[dict], groq_client) -> str:
    system = """You are a rigorous fact-checker.
Check every claim in the draft answer against the source chunks.
Flag anything NOT supported by the chunks.
Rate: ACCURATE / MOSTLY ACCURATE / NEEDS REVISION.
If revision needed, provide the corrected answer.
If accurate, confirm and add any missing nuance."""

    context = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks[:5]
    )
    prompt = f"Question: {question}\n\nDraft answer:\n{answer}\n\nSource chunks:\n{context}"
    return run_llm(system, prompt, groq_client)


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.4rem">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.25rem;
                    background:linear-gradient(135deg,#7c6af7,#6af7c8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            RAG PIPELINE<br>CLOUD
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                    color:#6b6b8a;margin-top:0.3rem;">
            <span class="dot"></span>Groq · Qdrant · Streamlit Cloud
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Keys ──
    st.markdown("**API KEYS**")
    st.caption("Set in Streamlit Secrets or enter below")

    groq_key   = st.text_input("Groq API key",   type="password",
                               value=get_secret("GROQ_API_KEY"),
                               placeholder="gsk_...")
    qdrant_url = st.text_input("Qdrant URL",
                               value=get_secret("QDRANT_URL"),
                               placeholder="https://xyz.qdrant.io:6333")
    qdrant_key = st.text_input("Qdrant API key", type="password",
                               value=get_secret("QDRANT_API_KEY"),
                               placeholder="your key")

    st.caption(
        "[Get Groq key](https://console.groq.com) · "
        "[Get Qdrant Cloud](https://cloud.qdrant.io)"
    )

    # Connection status
    groq_ok   = bool(groq_key)
    qdrant_ok = bool(qdrant_url and qdrant_key)

    for label, ok in [("Groq LLM", groq_ok), ("Qdrant DB", qdrant_ok)]:
        cls = "status-ok" if ok else "status-err"
        sym = "●" if ok else "○"
        st.markdown(
            f"<div style='font-family:JetBrains Mono;font-size:0.78rem' class='{cls}'>"
            f"{sym} {label}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Model selector ──
    st.markdown("**MODEL**")
    model_name = st.selectbox(
        "groq_model", list(GROQ_MODELS.keys()), label_visibility="collapsed"
    )
    st.session_state.groq_model = GROQ_MODELS[model_name]

    st.markdown("---")

    # ── Document upload ──
    st.markdown("**INDEX DOCUMENTS**")
    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and groq_ok and qdrant_ok:
        if st.button("📥  Index documents", type="primary"):
            try:
                embedder = get_embed_model()
                qdrant   = get_qdrant(qdrant_url, qdrant_key)
                progress = st.progress(0, text="Indexing...")
                total    = 0

                for i, f in enumerate(uploaded_files):
                    progress.progress(
                        (i + 1) / len(uploaded_files),
                        text=f"Indexing {f.name}..."
                    )
                    n = index_document(f, qdrant, embedder)
                    total += n
                    if f.name not in st.session_state.indexed_docs:
                        st.session_state.indexed_docs.append(f.name)

                progress.empty()
                st.success(f"✅ {len(uploaded_files)} file(s) · {total} chunks")
            except Exception as e:
                st.error(f"Index error: {e}")

    elif uploaded_files and not (groq_ok and qdrant_ok):
        st.warning("⚠️ Add API keys above first")

    st.markdown("---")

    # ── DB status ──
    st.markdown("**DATABASE**")
    if qdrant_ok:
        try:
            qdrant   = get_qdrant(qdrant_url, qdrant_key)
            info     = qdrant.get_collection(COLLECTION_NAME)
            count    = info.points_count or 0
            color    = "#6af7c8" if count > 0 else "#6b6b8a"
            st.markdown(
                f"<div style='font-family:JetBrains Mono;font-size:0.78rem;color:{color}'>"
                f"● {count} chunks indexed</div>",
                unsafe_allow_html=True
            )
            for doc in st.session_state.indexed_docs:
                st.markdown(f"<div class='doc-item'>📄 {doc}</div>", unsafe_allow_html=True)

            if count > 0 and st.button("🗑️  Clear all documents"):
                qdrant.delete_collection(COLLECTION_NAME)
                get_qdrant.clear()
                st.session_state.indexed_docs = []
                st.rerun()
        except Exception as e:
            st.caption(f"DB: {e}")
    else:
        st.markdown(
            "<div style='font-family:JetBrains Mono;font-size:0.73rem;color:#6b6b8a'>"
            "Add Qdrant credentials above.</div>", unsafe_allow_html=True
        )

    st.markdown("---")

    # ── History ──
    st.markdown("**HISTORY**")
    if not st.session_state.history:
        st.markdown(
            "<div style='font-family:JetBrains Mono;font-size:0.73rem;color:#6b6b8a'>"
            "No questions yet.</div>", unsafe_allow_html=True
        )
    else:
        for item in reversed(st.session_state.history[-5:]):
            short = item[:40] + "..." if len(item) > 40 else item
            st.markdown(
                f"<div style='font-family:JetBrains Mono;font-size:0.72rem;"
                f"background:#13131a;border:1px solid #1e1e2e;border-radius:5px;"
                f"padding:6px 10px;margin-bottom:4px'>{short}</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════
chunk_count = 0
if qdrant_ok:
    try:
        qdrant      = get_qdrant(qdrant_url, qdrant_key)
        chunk_count = qdrant.get_collection(COLLECTION_NAME).points_count or 0
    except Exception:
        pass

st.markdown(f"""
<div class="hero">
    <h1>📚 RAG Pipeline</h1>
    <p>// Groq · {st.session_state.groq_model} · Qdrant Cloud · {chunk_count} chunks · 4-agent pipeline</p>
</div>""", unsafe_allow_html=True)

# Pipeline flow
st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#6b6b8a;
            margin-bottom:1.5rem;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <span style="background:rgba(124,106,247,0.15);color:#7c6af7;padding:3px 10px;
                 border-radius:4px;font-weight:700">1 · RETRIEVE</span>
    <span>→</span>
    <span style="background:rgba(106,247,200,0.15);color:#6af7c8;padding:3px 10px;
                 border-radius:4px;font-weight:700">2 · ANALYZE</span>
    <span>→</span>
    <span style="background:rgba(247,130,106,0.15);color:#f7826a;padding:3px 10px;
                 border-radius:4px;font-weight:700">3 · SYNTHESIZE</span>
    <span>→</span>
    <span style="background:rgba(247,215,106,0.15);color:#f7d76a;padding:3px 10px;
                 border-radius:4px;font-weight:700">4 · CRITIC</span>
</div>
""", unsafe_allow_html=True)

# Guards
if not groq_ok or not qdrant_ok:
    st.warning("⚠️  Add your Groq and Qdrant API keys in the sidebar to get started.")
elif chunk_count == 0:
    st.info("📂  Upload and index documents in the sidebar to get started.")

# Input
col1, col2 = st.columns([4, 1])
with col1:
    question = st.text_area(
        "Question", height=110,
        placeholder="Ask a question about your indexed documents...",
        label_visibility="collapsed",
        key="main_input"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button(
        "▶  ASK", type="primary",
        use_container_width=True,
        disabled=(not groq_ok or not qdrant_ok or chunk_count == 0)
    )
    if st.session_state.last_result.get("critic"):
        final = st.session_state.last_result["critic"].get("text", "")
        st.download_button(
            "💾  SAVE", data=f"Q: {question}\n\nANSWER:\n{final}",
            file_name="rag_answer.txt", mime="text/plain",
            use_container_width=True
        )

# ── Run pipeline ──────────────────────────────────────
if run_btn and question.strip():
    try:
        embedder     = get_embed_model()
        qdrant       = get_qdrant(qdrant_url, qdrant_key)
        groq_client  = get_groq_client(groq_key)

        tab_final, tab_steps, tab_chunks = st.tabs([
            "✅  FINAL ANSWER", "🔬  AGENT STEPS", "📄  RETRIEVED CHUNKS"
        ])

        steps_cfg = [
            ("retrieve",   "1 · RETRIEVE",   "ab-retrieve",   "rb-retrieve"),
            ("analyze",    "2 · ANALYZE",    "ab-analyze",    "rb-analyze"),
            ("synthesize", "3 · SYNTHESIZE", "ab-synthesize", "rb-synthesize"),
            ("critic",     "4 · CRITIC",     "ab-critic",     "rb-critic"),
        ]

        with tab_steps:
            slots = {k: st.empty() for k, *_ in steps_cfg}
            for key, label, badge, _ in steps_cfg:
                slots[key].markdown(
                    f'<div class="agent-badge {badge}">{label} · waiting...</div>',
                    unsafe_allow_html=True
                )

        progress = st.progress(0, text="Starting pipeline...")
        results  = {}
        t_total  = time.time()

        # ── 1. Retrieve ──
        progress.progress(0.1, text="1/4 · Retrieving chunks...")
        t0 = time.time()
        chunks, queries = agent_retrieve(question, qdrant, embedder, groq_client)
        results["retrieve"] = {"chunks": chunks, "queries": queries, "time": round(time.time()-t0,1)}

        with tab_steps:
            slots["retrieve"].markdown(
                f'<div class="agent-badge ab-retrieve">1 · RETRIEVE · ✓ {results["retrieve"]["time"]}s</div>'
                f'<div class="result-box rb-retrieve">'
                f'Search queries:\n{queries}\n\n'
                f'Retrieved {len(chunks)} chunks from '
                f'{len(set(c["source"] for c in chunks))} source(s).</div>',
                unsafe_allow_html=True
            )

        if not chunks:
            st.error("No relevant chunks found. Try rephrasing or index more documents.")
            st.stop()

        # ── 2. Analyze ──
        progress.progress(0.35, text="2/4 · Analyzing content...")
        t0       = time.time()
        analysis = agent_analyze(question, chunks, groq_client)
        results["analyze"] = {"text": analysis, "time": round(time.time()-t0,1)}

        with tab_steps:
            slots["analyze"].markdown(
                f'<div class="agent-badge ab-analyze">2 · ANALYZE · ✓ {results["analyze"]["time"]}s</div>'
                f'<div class="result-box rb-analyze">{analysis}</div>',
                unsafe_allow_html=True
            )

        # ── 3. Synthesize ──
        progress.progress(0.6, text="3/4 · Synthesizing answer...")
        t0     = time.time()
        answer = agent_synthesize(question, analysis, chunks, groq_client)
        results["synthesize"] = {"text": answer, "time": round(time.time()-t0,1)}

        with tab_steps:
            slots["synthesize"].markdown(
                f'<div class="agent-badge ab-synthesize">3 · SYNTHESIZE · ✓ {results["synthesize"]["time"]}s</div>'
                f'<div class="result-box rb-synthesize">{answer}</div>',
                unsafe_allow_html=True
            )

        # ── 4. Critic ──
        progress.progress(0.85, text="4/4 · Reviewing answer...")
        t0       = time.time()
        critique = agent_critic(question, answer, chunks, groq_client)
        results["critic"] = {"text": critique, "time": round(time.time()-t0,1)}

        with tab_steps:
            slots["critic"].markdown(
                f'<div class="agent-badge ab-critic">4 · CRITIC · ✓ {results["critic"]["time"]}s</div>'
                f'<div class="result-box rb-critic">{critique}</div>',
                unsafe_allow_html=True
            )

        total = round(time.time() - t_total, 1)
        progress.progress(1.0, text=f"✅ Done in {total}s")

        # Final answer tab
        with tab_final:
            times = " · ".join(
                f"{k} {results[k]['time']}s"
                for k, *_ in steps_cfg if k in results and "time" in results[k]
            )
            st.markdown(
                f'<div style="font-family:JetBrains Mono;font-size:0.7rem;color:#6b6b8a;'
                f'margin-bottom:0.8rem">{times} · total {total}s</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="result-box rb-critic">{critique}</div>',
                unsafe_allow_html=True
            )

        # Chunks tab
        with tab_chunks:
            st.markdown(
                f"<div style='font-family:JetBrains Mono;font-size:0.72rem;color:#6b6b8a;"
                f"margin-bottom:0.8rem'>{len(chunks)} chunks retrieved</div>",
                unsafe_allow_html=True
            )
            for i, c in enumerate(chunks):
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<div class="chunk-meta">#{i+1} · {c["source"]} · '
                    f'chunk {c["chunk_index"]} · relevance {c["relevance"]}</div>'
                    f'{c["text"]}</div>',
                    unsafe_allow_html=True
                )

        st.session_state.last_result = results
        st.session_state.history.append(question)

    except Exception as e:
        st.error(f"❌ Error: {e}")

elif st.session_state.last_result.get("critic") and not run_btn:
    st.markdown(
        "<div style='font-family:JetBrains Mono;font-size:0.72rem;"
        "color:#6b6b8a;margin-bottom:0.5rem'>// last answer</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="result-box rb-critic">'
        f'{st.session_state.last_result["critic"]["text"]}</div>',
        unsafe_allow_html=True
    )