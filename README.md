# DocuMind 📄
### Multi-Document RAG Agent — Python + LangChain + ChromaDB + Streamlit

A production-quality Gen AI application that lets you upload multiple PDFs and
ask an AI agent questions across all of them. The agent autonomously decides
which tool to call, retrieves grounded answers, and cites every source.

---

## Architecture

```
PDF files
   │
   ▼
[PyMuPDF]  ──── extract text per page
   │
   ▼
[RecursiveCharacterTextSplitter]  ──── chunk (800 tokens, 100 overlap)
   │
   ▼
[OpenAI text-embedding-3-small]  ──── embed each chunk
   │
   ▼
[ChromaDB]  ──── persist vectors locally
   │
   ▼
[LangChain Tool-Calling Agent]  ──── GPT-4o decides which tool to call
   │
   ├── search_docs(query)       → semantic search + confidence scores
   ├── summarize_doc(filename)  → per-doc structured summary
   └── compare_docs(topic)      → cross-document analysis
   │
   ▼
[Streamlit]  ──── streaming chat UI with citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-small |
| Vector DB | ChromaDB (local) |
| RAG + Agent | LangChain |
| PDF parsing | PyMuPDF (fitz) |
| Environment | python-dotenv |

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/documind.git
cd documind

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add your OpenAI API key

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## How to Use

1. **Upload PDFs** in the sidebar (annual reports, research papers, contracts — anything)
2. Click **⚡ Ingest Documents** — this runs the full RAG pipeline
3. **Ask questions** in the chat, for example:
   - _"What are the main findings?"_
   - _"Summarize the document named 'report.pdf'"_
   - _"Compare how both documents discuss revenue"_
   - _"What methodology was used in the research?"_

---

## Project Structure

```
documind/
├── app.py               # Streamlit UI + chat interface
├── src/
│   ├── ingestion.py     # PDF → chunks → embeddings → ChromaDB
│   ├── tools.py         # Agent tools: search, summarize, compare
│   └── agent.py         # LangChain agent + prompt + history
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Concepts Demonstrated

- **RAG pipeline**: chunking strategy, overlap, embedding model selection
- **Semantic search**: vector similarity with relevance scores
- **Tool-calling agent**: LLM autonomously decides which tool to invoke
- **Hallucination control**: citations + confidence scores + "I don't know" fallback
- **Multi-turn memory**: chat history passed to every agent call
- **Cross-document reasoning**: compare_docs tool synthesises across sources

---

## Resume Talking Points

> *Built a multi-document RAG agent in Python using LangChain, GPT-4o, and ChromaDB.
> Implemented a full ingestion pipeline (PDF parsing → chunking → embeddings → vector store)
> and a tool-calling agent with three tools: semantic search, document summarization,
> and cross-document comparison. Includes citation grounding and confidence scoring
> to minimise hallucination.*

---

## Possible Extensions (Day 4+)

- [ ] Add support for `.txt`, `.docx`, `.csv` files
- [ ] Switch to **Pinecone** for cloud-hosted vectors (production-ready)
- [ ] Add **LangGraph** for more complex multi-step agent workflows
- [ ] Stream the agent's response token-by-token with `st.write_stream`
- [ ] Add a **re-ranking step** (Cohere Rerank) before passing results to the LLM
- [ ] Deploy to **Streamlit Community Cloud** (free, one-click)
