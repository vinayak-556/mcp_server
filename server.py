# server.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# -------------------------------------
# CONFIG
# -------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

LANG_INDEX_NAME = os.getenv("LANG_INDEX_NAME", "lang-docs")
RETELL_INDEX_NAME = os.getenv("RETELL_INDEX_NAME", "retell-docs")

EMBED_MODEL = "all-MiniLM-L6-v2"

# Load embedding model once
model = SentenceTransformer(EMBED_MODEL)

# Connect Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

lang_index = pc.Index(LANG_INDEX_NAME)
retell_index = pc.Index(RETELL_INDEX_NAME)

# MCP server
mcp = FastMCP("pinecone-multi-search",protocol_version="2025-03-26")


# -------------------------------------
# HELPERS
# -------------------------------------
def pinecone_query(index, query, top_k):
    """Run semantic search on ANY Pinecone index."""
    q_vec = model.encode(query).tolist()

    results = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True
    )

    output = []
    for match in results["matches"]:
        md = match["metadata"]

        output.append({
            "score": match["score"],
            "uri": md.get("uri"),
            "title": md.get("title"),
            "source_url": md.get("source_url"),
            "preview": md.get("text", "")[:350]
        })

    return output


# -------------------------------------
# TOOLS
# -------------------------------------

@mcp.tool
async def lang_search(query: str, top_k: int = 5):
    """Search the LangChain/LangGraph documentation."""
    return pinecone_query(lang_index, query, top_k)


@mcp.tool
async def retell_search(query: str, top_k: int = 5):
    """Search the RetellAI documentation stored in Pinecone."""
    return pinecone_query(retell_index, query, top_k)


# -------------------------------------
# RUN SERVER
# -------------------------------------
if __name__ == "__main__":
    mcp.run()

