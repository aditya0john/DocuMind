"""
tools.py
--------
Three LangChain tools that give the agent its capabilities:

  search_docs   -> semantic search with confidence scores + citations
  summarize_doc -> summarize one document by filename
  compare_docs  -> cross-document comparison on a topic

Uses ChatMistralAI as the reasoning LLM and MistralAIEmbeddings for retrieval.
"""

from langchain.tools import tool
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "mistral-large-latest"


def _get_vectorstore() -> Chroma:
    """Return the ChromaDB vector store using Mistral embeddings."""
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


def make_tools() -> list:
    """
    Build and return the three agent tools.
    Tools share a single vectorstore + llm instance for efficiency.
    """
    vectorstore = _get_vectorstore()
    llm = ChatMistralAI(model=LLM_MODEL, temperature=0)

    # ── Tool 1: Semantic search across all docs ────────────────────────
    @tool
    def search_docs(query: str) -> str:
        """
        Search across ALL uploaded documents for content relevant to a query.
        Returns the top matching chunks with source filename, page number,
        and a relevance score. Always use this tool first before answering.
        """
        results = vectorstore.similarity_search_with_relevance_scores(query, k=5)

        if not results:
            return "No relevant content found in the uploaded documents."

        output_parts = []
        for doc, score in results:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            confidence = round(score * 100, 1)

            output_parts.append(
                f"[Source: {source} | Page {page} | Relevance: {confidence}%]\n"
                f"{doc.page_content}"
            )

        return "\n\n---\n\n".join(output_parts)

    # ── Tool 2: Summarize a single document ────────────────────────────
    @tool
    def summarize_doc(filename: str) -> str:
        """
        Generate a structured summary of a specific document.
        Input must be the exact filename (e.g. 'annual_report.pdf').
        Covers key topics, main arguments, and important figures.
        """
        results = vectorstore.similarity_search(
            "key topics overview introduction summary conclusion",
            k=10,
            filter={"source": filename},
        )

        if not results:
            return (
                f"No document named '{filename}' found. "
                "Check the sidebar for the exact filenames."
            )

        combined = "\n\n".join(r.page_content for r in results)

        response = llm.invoke(
            f"You are a document analyst. Provide a clear, structured summary of "
            f"the following content from '{filename}'.\n\n"
            f"Include:\n"
            f"- Main topic / purpose\n"
            f"- Key points (3-5 bullets)\n"
            f"- Important figures or dates (if any)\n\n"
            f"Content:\n{combined}"
        )

        return f"**Summary of '{filename}':**\n\n{response.content}"

    # ── Tool 3: Cross-document comparison ─────────────────────────────
    @tool
    def compare_docs(topic: str) -> str:
        """
        Compare how different uploaded documents address the same topic.
        Input is a topic or question (e.g. 'revenue growth', 'methodology').
        Highlights agreements, contradictions, and unique insights per document.
        Requires at least 2 documents to be uploaded.
        """
        results = vectorstore.similarity_search(topic, k=12)

        if not results:
            return "No relevant content found to compare."

        # Group chunks by source document
        by_source: dict[str, list[str]] = {}
        for doc in results:
            src = doc.metadata.get("source", "Unknown")
            by_source.setdefault(src, []).append(doc.page_content)

        if len(by_source) < 2:
            found = list(by_source.keys())
            return (
                f"Only found relevant content in one document ({found[0]}). "
                "Upload more documents to enable comparison."
            )

        comparison_sections = []
        for src, chunks in by_source.items():
            excerpt = "\n".join(chunks[:3])  # top 3 chunks per source
            comparison_sections.append(f"=== {src} ===\n{excerpt}")

        combined = "\n\n".join(comparison_sections)

        response = llm.invoke(
            f"You are a research analyst. Compare how these documents address: '{topic}'\n\n"
            f"{combined}\n\n"
            f"Structure your answer as:\n"
            f"1. Where the documents AGREE\n"
            f"2. Where they DIFFER\n"
            f"3. Unique insights from each document\n"
            f"Always mention the document name when referencing content."
        )

        return response.content

    return [search_docs, summarize_doc, compare_docs]