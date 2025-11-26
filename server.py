# server.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from fastmcp import FastMCP

# Load .env
load_dotenv()

# Config
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "lang-docs")

model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

mcp = FastMCP("pinecone-doc-search")


def pinecone_search(query, top_k=3):
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
            "uri": md["uri"],
            "title": md["title"],
            "preview": md["text"][:350],
        })

    return output


@mcp.tool
async def search_tool(query: str, top_k: int = 3):
    """Semantic search using Pinecone vector DB."""
    return pinecone_search(query, top_k)


if __name__ == "__main__":
    mcp.run()
