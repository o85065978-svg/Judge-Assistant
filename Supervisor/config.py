"""
config.py

Central configuration for the Supervisor agent.

All tuneable parameters live here so that nothing is hardcoded in the
node or adapter modules.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("SUPERVISOR_LLM_MODEL", "gemini-1.5-flash")
LLM_TEMPERATURE: float = float(os.getenv("SUPERVISOR_LLM_TEMPERATURE", "0"))

# ---------------------------------------------------------------------------
# Retry / validation
# ---------------------------------------------------------------------------
MAX_RETRIES: int = int(os.getenv("SUPERVISOR_MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
MAX_CONVERSATION_TURNS: int = int(
    os.getenv("SUPERVISOR_MAX_CONVERSATION_TURNS", "20")
)

# ---------------------------------------------------------------------------
# Agent registry -- canonical names used in target_agents lists
# ---------------------------------------------------------------------------
AGENT_NAMES = [
    "ocr",
    "summarize",
    "civil_law_rag",
    "case_doc_rag",
    "reason",
]

# ---------------------------------------------------------------------------
# Valid intents the classifier may return
# ---------------------------------------------------------------------------
VALID_INTENTS = AGENT_NAMES + ["multi", "off_topic"]

# ---------------------------------------------------------------------------
# MongoDB configuration
# ---------------------------------------------------------------------------
MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB: str = os.getenv("MONGO_DB", "Rag")
MONGO_COLLECTION: str = os.getenv("MONGO_COLLECTION", "Document Storage")

# ---------------------------------------------------------------------------
# Vector store (Chroma) configuration -- shared with Case Doc RAG
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"
)
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "judicial_docs")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "")
