# FastAPI Backend

REST API for PDF knowledge extraction and Q&A system with hybrid retrieval and local embeddings.

## Features

- **POST /query** - Answer questions with top 5 grounded citations
- **GET /companies** - List available companies
- **Hybrid retrieval** - Vector search + full-text + graph traversal
- **Local embeddings** - sentence-transformers (zero API calls for retrieval)
- **Grounded answers** - Gemini 2.5 Flash with strict citation requirements

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Configuration

Create `.env` file in project root:

```bash
GEMINI_API_KEY=your-api-key-here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

## Running the Server

### Development Mode

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, visit:
- **Interactive docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## API Endpoints

### GET /
Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "message": "PDF Knowledge Q&A API is running"
}
```

### GET /companies
Get list of available companies

**Response:**
```json
{
  "companies": [
    {
      "id": "company_1",
      "name": "Acme Corp",
      "description": "Manufacturing contracts and invoices"
    }
  ]
}
```

### POST /query
Ask a question and get grounded answer with top 5 citations

**Request:**
```json
{
  "question": "What happened to NSF's Total Liabilities in FY 2011?",
  "company_id": "company_1"
}
```

**Response:**
```json
{
  "answer": "NSF's Total Liabilities decreased by 2.5 percent in FY 2011. This change primarily relates to NSF encouraging its partnering agencies to work on a reimbursable basis, reducing the related Advances from Others liability [1].",
  "citations": [
    {
      "chunk_id": "us-018_p2_b4_c0",
      "text": "NSF's Total Liabilities (Figure 11) decreased by 2.5 percent in FY 2011. The majority of this change is related to NSF's strides to encourage its partnering agencies to work on a reimbursable basis, reducing the related Advances from Others liability.",
      "page_num": 2,
      "doc_id": "us-018",
      "doc_name": "us-018.pdf",
      "image_path": "data/company_1/images/us-018_page_2.png",
      "bbox": [72.01, 331.03, 290.05, 406.48],
      "score": 0.887
    }
  ]
}
```

**Note:** Maximum 5 citations returned (configured in `config.py` as `TOP_K_CHUNKS = 5`)

## Architecture

### Components

- **main.py** - FastAPI application and endpoints
- **rag_engine.py** - RAG logic (retrieval + generation)
- **config.py** - Configuration management
- **models.py** - Pydantic data models

### RAG Pipeline

1. **Query Embedding** - Generate 768-dim vector with **local sentence-transformers** (instant, no API)
2. **Hybrid Retrieval:**
   - Vector similarity search in Neo4j (cosine similarity)
   - Full-text search (keyword matching)
   - Graph traversal via entity relations
   - Merge and rank by combined score
   - Return top 5 chunks
3. **Answer Generation** - Gemini 2.5 Flash with grounding prompt (only 1 API call)
4. **Citation Formatting** - Return sources with metadata (screenshots, bboxes, scores)

### Retrieval Strategy

**Vector Search (Weight: 1.0)**
- Uses Neo4j vector index on pre-computed embeddings
- Cosine similarity on 768-dim vectors
- Local embedding generation (no API call)
- Filter by company_id
- Threshold: 0.5

**Full-Text Search (Weight: 0.5)**
- Neo4j full-text index on chunk text
- Keyword matching
- Known issue: Session.run() parameter conflict (non-blocking)

**Graph Traversal (Weight: 0.3)**
- Extract entities from query text
- Follow RELATES_TO relationships
- Find chunks connected to relevant entities

**Result Merging:**
- Deduplicate by chunk_id
- Combined score = vector_score × 1.0 + fulltext_score × 0.5 + graph_score × 0.3
- Return top 5 chunks

## Configuration Options

Edit `app/config.py`:

```python
# Retrieval parameters
TOP_K_CHUNKS = 5                     # Return top 5 most relevant (optimized for UI)
VECTOR_SIMILARITY_THRESHOLD = 0.5    # Minimum similarity score

# Model names  
GEMINI_PRO_MODEL = "gemini-2.5-flash"  # Fast, available model
# Note: Embeddings use local sentence-transformers (no API calls)
```

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8000/

# Get companies
curl http://localhost:8000/companies

# Query (returns top 5 citations)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the payment terms?",
    "company_id": "company_1"
  }'
```

### Using Python

```python
import requests

# Query endpoint
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What are the deliverables?",
        "company_id": "company_1"
    }
)

result = response.json()
print(result['answer'])
for citation in result['citations']:
    print(f"- {citation['doc_name']}, page {citation['page_num']}")
```

## Troubleshooting

### Error: "Neo4j connection refused"

**Solution:**
```bash
# Check Neo4j is running
docker ps

# Restart Neo4j
docker-compose restart neo4j
```

### Error: "GEMINI_API_KEY not found"

**Solution:** Add key to `.env` file in project root

### Error: "Vector index not found"

**Solution:** Run knowledge graph builder:
```bash
python scripts/build_knowledge_graph.py
```

### Error: No results returned

**Possible causes:**
- Knowledge graph is empty
- Company ID is wrong
- Vector index not built yet

**Check Neo4j:**
```cypher
MATCH (c:Chunk) WHERE c.company_id = 'company_1' RETURN count(c);
```

## Performance Optimization

### Faster Retrieval

1. **Adjust TOP_K_CHUNKS** - Lower value = faster
2. **Increase similarity threshold** - Fewer results to process
3. **Disable graph traversal** - Comment out in `rag_engine.py`

### Better Answers

1. **Increase TOP_K_CHUNKS** - More context for LLM
2. **Improve prompts** - Edit grounding prompt in `rag_engine.py`
3. **Add re-ranking** - Implement post-retrieval scoring

## Logging

Logs are written to console (stdout). View with:

```bash
# Follow logs in real-time
uvicorn app.main:app --log-level info
```

## Next Steps

- Start Streamlit frontend: `cd frontend && streamlit run streamlit_app.py`
- View API docs: http://localhost:8000/docs
- Test queries through UI: http://localhost:8501
