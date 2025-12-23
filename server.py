import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from fastmcp import FastMCP
 
# Load environment variables
load_dotenv()
 
# -------------------------------------
# CONFIG
# -------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 
LANG_INDEX_NAME = os.getenv("LANG_INDEX_NAME", "lang-docs")
RETELL_INDEX_NAME = os.getenv("RETELL_INDEX_NAME", "retell-docs")
MAKE_INDEX_NAME = os.getenv("MAKE_INDEX_NAME", "make-docs")
 
EMBED_MODEL = "text-embedding-3-small"
 
# Load OpenAI client for embeddings
client = OpenAI(api_key=OPENAI_API_KEY)
 
# Connect Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
 
lang_index = pc.Index(LANG_INDEX_NAME)
retell_index = pc.Index(RETELL_INDEX_NAME)
make_index = pc.Index(MAKE_INDEX_NAME)
 
# MCP server
mcp = FastMCP("pinecone-multi-search")
 
 
# -------------------------------------
# HELPERS
# -------------------------------------
def get_embedding(text):
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return response.data[0].embedding
 
 
def pinecone_query(index, query, top_k):
    """Run semantic search on ANY Pinecone index."""
    q_vec = get_embedding(query)
 
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
            "filename": md.get("filename"),
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
 
 
@mcp.tool
async def make_search(query: str, top_k: int = 5):
    """Search the Make.com documentation and guides stored in Pinecone."""
    return pinecone_query(make_index, query, top_k)
 
 
# -------------------------------------
# RUN SERVER
# -------------------------------------
if __name__ == "__main__":
    mcp.run()
