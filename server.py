import requests
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from fastmcp import FastMCP
from mcp.types import Resource


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
PAGES = {
    # -------- LangGraph Docs --------
    "langgraph-overview": "https://docs.langchain.com/oss/python/langgraph/overview",
    "langgraph-building-graphs": "https://docs.langchain.com/oss/python/langgraph/graph-api",
    "langgraph-agents": "https://docs.langchain.com/oss/python/langgraph/workflows-agents",
    "langgraph-state": "https://docs.langchain.com/oss/python/learn",

    # -------- LangChain Docs --------
    "langchain-overview": "https://docs.langchain.com/oss/python/langchain/overview",
    "langchain-agents": "https://docs.langchain.com/oss/python/langchain/agents",
    "langchain-models": "https://docs.langchain.com/oss/python/langchain/models",
    "langchain-tools": "https://docs.langchain.com/oss/python/langchain/tools",
}

EMBED_MODEL = "all-MiniLM-L6-v2"


# ----------------------------------------------------
# GLOBALS
# ----------------------------------------------------
index = None
model = SentenceTransformer(EMBED_MODEL)
doc_chunks = []


# ----------------------------------------------------
# FETCH & CLEAN
# ----------------------------------------------------
def fetch_page_text(url):
    """Fetch a page HTML and convert to clean text."""
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # remove noise
        for tag in soup(["nav", "footer", "script", "style", "head"]):
            tag.decompose()

        # cleaned text
        text = soup.get_text(separator="\n")
        text = "\n".join(
            line.strip()
            for line in text.splitlines()
            if line.strip()
        )
        return text

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def load_docs():
    """Fetch LangChain + LangGraph docs."""
    docs = []
    for name, url in PAGES.items():
        print(f"Fetching {name} ...")
        text = fetch_page_text(url)
        if text:
            prefix = "langchain" if name.startswith("langchain") else "langgraph"
            uri = f"{prefix}://docs/{name.replace(prefix + '-', '')}"

            docs.append({
                "uri": uri,
                "title": name,
                "text": text
            })
    return docs


# ----------------------------------------------------
# CHUNK + EMBED + INDEX
# ----------------------------------------------------
def chunk_text(text, chunk_size=400):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def build_vector_index():
    global index, doc_chunks

    docs = load_docs()
    embeddings = []

    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            doc_chunks.append({
                "uri": doc["uri"],
                "title": doc["title"],
                "text": chunk
            })
            embeddings.append(model.encode(chunk))

    embeddings = np.array(embeddings)
    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(embeddings)

    print(f"Indexed {len(doc_chunks)} chunks from {len(docs)} documents.")


# ----------------------------------------------------
# SEARCH
# ----------------------------------------------------
def search_docs(query, top_k=3):
    if index is None:
        raise RuntimeError("Index not built yet.")

    q_vec = model.encode([query])
    distances, indices = index.kneighbors(q_vec, n_neighbors=top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        chunk = doc_chunks[idx]
        results.append({
            "uri": chunk["uri"],
            "title": chunk["title"],
            "score": float(dist),
            "preview": chunk["text"][:500],
        })

    return results


# ----------------------------------------------------
# MCP SERVER
# ----------------------------------------------------
mcp = FastMCP("langchain-langgraph-docs")


# ---- Search Tool ----
@mcp.tool
async def search_tool(query: str, top_k: int = 3):
    """Semantic search across LangChain + LangGraph docs."""
    return search_docs(query, top_k)


# ----------------------------------------------------
# DYNAMIC MCP RESOURCES (required by FastMCP)
# ----------------------------------------------------
@mcp.resource("langgraph://docs/{name}")
async def langgraph_docs(name: str):
    key = f"langgraph-{name}"
    if key in PAGES:
        return fetch_page_text(PAGES[key])
    return f"Document not found: {name}"


@mcp.resource("langchain://docs/{name}")
async def langchain_docs(name: str):
    key = f"langchain-{name}"
    if key in PAGES:
        return fetch_page_text(PAGES[key])
    return f"Document not found: {name}"


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    build_vector_index()
    mcp.run()
