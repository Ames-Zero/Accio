"""
Backend Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model names
GEMINI_PRO_MODEL = "gemini-2.5-flash"  # Fast, available model
# Note: Embeddings now use local sentence-transformers (no API calls)

# Retrieval parameters
TOP_K_CHUNKS = 5  # Limit to top 5 most relevant sources for cleaner UI
VECTOR_SIMILARITY_THRESHOLD = 0.5

# API Configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
