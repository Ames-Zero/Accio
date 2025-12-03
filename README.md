# PDF Knowledge Extraction & Q&A System

A simplified PDF knowledge extraction and Q&A chatbot system for college demo. Extracts knowledge from pre-ingested company PDFs, builds a Neo4j knowledge graph with vector embeddings, and provides a Streamlit chat interface with grounded citations.

## Key Features

- **Offline PDF Processing**: Pre-ingest PDFs (24 US federal documents included)
- **Layout-Aware Extraction**: Uses PyMuPDF for text extraction with bounding boxes
- **Knowledge Graph**: Neo4j with vector embeddings (local sentence-transformers)
- **Hybrid Retrieval**: Vector similarity + graph traversal + full-text search
- **Local Embeddings**: No API calls for embeddings - instant, free, and private
- **Grounded Answers**: Top 5 citations with PDF screenshots and bbox highlights
- **Simple UI**: Streamlit chat interface with example questions

## Project Structure

```
pdf-knowledge-system/
├── data/
│   ├── company_1/
│   │   ├── pdfs/                # 24 sample PDFs (US federal documents)
│   │   └── images/              # 61 page screenshots (300 DPI)
│   ├── processed/company_1/     # 24 extracted JSON files
│   └── companies.yaml           # Company metadata
├── scripts/
│   ├── extract_simple.py        # PDF extraction (PyMuPDF)
│   ├── build_knowledge_graph.py # Neo4j population
│   └── test_query.py           # Test RAG system
├── backend/                     # FastAPI REST API
├── frontend/                    # Streamlit chat UI
├── neo4j_data/                  # Neo4j database (529MB, persisted)
└── docker-compose.yml           # Neo4j container setup
```

## Quick Start

### 1. Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Google Gemini API key ([get one here](https://ai.google.dev/))
- CUDA GPU (optional, for faster local embeddings)

### 2. Installation

```bash
# Clone repository
cd pdf-knowledge-system

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install streamlit requests pillow

# Setup environment
cp .env.example .env
nano .env  # Add your GEMINI_API_KEY

# Start Neo4j
docker compose up -d

# Wait 15 seconds for Neo4j to initialize
sleep 15
```

### 3. Download Pre-built Database (Recommended)

**Option A: Use Pre-populated Database**

1. Download `neo4j_database_backup.tar.gz` (20MB) from shared location
2. Extract: `tar -xzf neo4j_database_backup.tar.gz`
3. Restart Neo4j: `docker compose restart neo4j`
4. Skip to step 5 (testing)

**Option B: Build From Scratch (30-40 minutes)**

```bash
# PDFs already included in data/company_1/pdfs/ (24 files)

# Step 1: Extract PDFs (2-3 minutes)
python3 scripts/extract_simple.py

# Step 2: Build knowledge graph (30-35 minutes due to Gemini API limits)
python3 scripts/build_knowledge_graph.py
```

**Database stats:**
- 3,239 nodes (2,095 Entities, 1,061 Chunks, 58 Pages, 24 Documents)
- 4,941 relationships
- 1,061 chunks with 768-dim embeddings (local, no API)

### 4. Test the System

```bash
# Quick test without starting servers
python3 scripts/test_query.py "What happened to NSF's Total Liabilities in FY 2011?"
```

### 5. Start the System

```bash
# Terminal 1: Start FastAPI backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &

# Terminal 2: Start Streamlit frontend  
cd frontend
streamlit run streamlit_app.py --server.port 8501 > ../frontend.log 2>&1 &
```

**Access:**
- **Streamlit UI**: http://localhost:8501 ← Main interface
- **FastAPI docs**: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474

**Stop all services:**
```bash
pkill -f uvicorn && pkill -f streamlit && docker compose down
```

**Restart:**
```bash
docker compose up -d && sleep 15
# Then start backend and frontend as above
```

## Testing the System

### Example Questions

Try these in the Streamlit UI (based on actual documents):

**NSF Financial Information:**
- "What happened to NSF's Total Liabilities in FY 2011?"
- "Summarize NSF's financial position"

**Nanotechnology Research:**
- "What is the NNI's approach to addressing ethical questions?"
- "What are nanotechnology's potential environmental impacts?"

**Accessibility Regulations:**
- "Where must accessible parking spaces be located?"
- "What are the accessibility requirements for entrances?"

### Verify System Health

```bash
# Test RAG system directly
python3 scripts/test_query.py "What happened to NSF's Total Liabilities in FY 2011?"

# Check Neo4j data (in browser at http://localhost:7474)
MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC;

# Verify embeddings
MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c);

# Check processed files
ls -lh data/processed/company_1/  # Should show 24 JSON files
ls -lh data/company_1/images/     # Should show 61 PNG files
```

## Architecture

### Tech Stack

**Extraction:**
- Docling - IBM's AI PDF text and layout extraction
- PyMuPDF (fitz) - PDF text extraction with bboxes
- pdf2image + poppler - 300 DPI screenshots

**Knowledge Graph:**
- Neo4j 5.15 - Graph database with vector index
- sentence-transformers (all-mpnet-base-v2) - Local 768-dim embeddings
- Google Gemini (gemini-2.5-flash) - Triplet extraction only

**Backend:**
- FastAPI - REST API
- Neo4j Python driver - Database queries
- Local embeddings - Zero API calls for retrieval

**Frontend:**
- Streamlit - Chat UI with citations
- PIL - Image display with bbox overlays

### API Endpoints

- `POST /query` - Ask questions, get grounded answers with citations
- `GET /companies` - List available companies

## Component Documentation

### Scripts

- `scripts/extract_simple.py` - PyMuPDF extraction with bboxes
- `scripts/extract_docling.py` - extraction using Docling
- `scripts/build_knowledge_graph.py` - Neo4j population (local embeddings + Gemini triplets)
- `scripts/test_query.py` - Test RAG system without API
- See `scripts/README.md` for detailed usage

### Backend

- `backend/app/main.py` - FastAPI application with CORS
- `backend/app/rag_engine.py` - Hybrid retrieval (vector + text + graph) + answer generation
- `backend/app/config.py` - Configuration (TOP_K_CHUNKS = 5)
- `backend/app/models.py` - Pydantic models with citations

### Frontend

- `frontend/streamlit_app.py` - Chat UI with:
  - Company selector
  - Example questions sidebar
  - Citation display with screenshots
  - Bounding box overlays (scaled for 300 DPI)

## Configuration

### Environment Variables

```bash
GEMINI_API_KEY=your-api-key        # Required: Gemini API key
NEO4J_URI=bolt://localhost:7687    # Neo4j connection
NEO4J_USER=<DB_NAME>                   # Neo4j username
NEO4J_PASSWORD=<DB_PASSWORD>         # Neo4j password
```

### Company Metadata

Edit `data/companies.yaml` to customize company names and descriptions.


## System Stats

**Current Dataset:**
- **PDFs:** 24 US federal documents (company_1)
- **Pages:** 58 total pages
- **Screenshots:** 61 PNG files at 300 DPI
- **Neo4j Nodes:** 3,239 (2,095 Entities, 1,061 Chunks, 58 Pages, 24 Documents, 1 Company)
- **Relationships:** 4,941
- **Database Size:** 529MB (persisted in neo4j_data/)

**Configuration:**
- **Chunk size:** 512 tokens with 50-token overlap
- **Embeddings:** 768-dim (sentence-transformers all-mpnet-base-v2, local)
- **Retrieval:** Top 5 most relevant chunks per query
- **Gemini usage:** Only for triplet extraction (graph building) and answer generation
- **API calls per query:** 1 (answer generation only, embeddings are local)


## Additional Documentation

- **[scripts/README.md](scripts/README.md)** - Extraction and graph building details
- **[backend/README.md](backend/README.md)** - Backend API documentation
- **[frontend/README.md](frontend/README.md)** - Frontend UI details

