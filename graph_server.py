

import os

from dotenv import load_dotenv
load_dotenv()

from src.vector.vector_store import VectorStore

vector_store = VectorStore()

AGENT_VERSION = os.getenv("AGENT_VERSION", "v1")
if AGENT_VERSION == "v2":
    from src.agent_v2.graph import build_rag_graph
else:
    from src.agent.graph import build_rag_graph

graph = build_rag_graph(vector_store)
