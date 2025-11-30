# Processing & Testing Scripts

This directory contains scripts for offline PDF extraction, knowledge graph building, and testing.

## Scripts Overview

### 1. `extract_simple.py`
Extracts text from PDFs using PyMuPDF and generates page screenshots at 300 DPI.

### 2. `build_knowledge_graph.py`
Populates Neo4j with extracted content, generates embeddings (local), and extracts knowledge triplets using Gemini.

### 3. `test_query.py`
Tests the RAG system directly without running the API. Useful for quick validation.

## Prerequisites

Before running these scripts:

1. **Add PDFs to data directories:**
   ```
   data/company_1/pdfs/    ← 15-20 PDF files
   data/company_2/pdfs/    ← 15-20 PDF files
   data/company_3/pdfs/    ← 15-20 PDF files
   ```

2. **Start Neo4j:**
   ```bash
   cd ..
   docker-compose up -d
   # Wait 30 seconds for initialization
   ```

3. **Configure environment:**
   ```bash
   cp ../.env.example ../.env
   # Edit .env and add GEMINI_API_KEY
   ```

4. **Install dependencies:**
   ```bash
   pip install -r ../backend/requirements.txt
   ```

## Usage

### Step 1: Extract PDFs

```bash
python extract_with_docling.py
```

**What it does:**
- Processes all PDFs in `data/{company_id}/pdfs/`
- Extracts text blocks with layout types (title, paragraph, list, etc.)
- Extracts tables with structure (headers, rows, cells)
- Extracts figures with captions
- Generates 300 DPI page screenshots
- Saves JSON outputs to `data/processed/{company_id}/`
- Saves screenshots to `data/{company_id}/images/`

**Output files:**
```
data/processed/company_1/contract_001.json
data/processed/company_1/invoice_002.json
...
data/company_1/images/contract_001_page_1.png
data/company_1/images/contract_001_page_2.png
...
```

**Time estimate:** ~30-60 seconds per PDF

**Troubleshooting:**
- If extraction fails, check PDF is not password-protected
- For scanned PDFs, Docling will use OCR (slower)
- Check `extraction.log` for detailed error messages

### Step 2: Build Knowledge Graph

```bash
python build_knowledge_graph.py
```

**What it does:**
- Connects to Neo4j database
- Creates schema (nodes, relationships, indexes)
- Loads extracted JSON files
- Chunks text into 512-token segments with 50-token overlap
- Generates embeddings using Gemini text-embedding-004 (768-dim)
- Extracts knowledge triplets using Gemini 1.5 Pro
- Populates Neo4j with chunks, entities, relations
- Creates vector index for similarity search

**Time estimate:** ~2-3 minutes per PDF (Gemini API has 4-second delays for rate limiting)

**Rate limits:**
- Gemini free tier: 15 RPM (requests per minute)
- Script includes 4-second delays between API calls
- For 50 PDFs with ~10 pages each: ~2-3 hours total

**Troubleshooting:**
- If rate limit errors occur, script will continue with next item
- Check `graph_builder.log` for detailed progress
- Verify Neo4j is running: `docker ps`
- Test Neo4j connection: http://localhost:7474

## Verification

### Check Extracted Files

```bash
# Count JSON files
ls -1 data/processed/company_1/*.json | wc -l

# Count screenshots
ls -1 data/company_1/images/*.png | wc -l

# View sample JSON
cat data/processed/company_1/sample.json | jq '.pages[0]'
```

### Check Neo4j Data

Open Neo4j Browser at http://localhost:7474 (neo4j/password123)

```cypher
// Count chunks
MATCH (c:Chunk) RETURN count(c) as total_chunks;

// Count entities
MATCH (e:Entity) RETURN count(e) as total_entities;

// Count relations
MATCH ()-[r:RELATES_TO]->() RETURN count(r) as total_relations;

// View sample chunk with embedding
MATCH (c:Chunk) 
WHERE c.company_id = 'company_1'
RETURN c.chunk_id, c.text, size(c.embedding) as embedding_dim
LIMIT 5;

// View sample knowledge triplet
MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
WHERE r.company_id = 'company_1'
RETURN s.name, r.relation, o.name, r.evidence
LIMIT 10;

// Check vector index status
SHOW INDEXES;
```

## Output Format

### Extraction JSON Schema

```json
{
  "doc_id": "contract_001",
  "company_id": "company_1",
  "filename": "contract_001.pdf",
  "total_pages": 5,
  "pages": [
    {
      "page_num": 1,
      "image_path": "data/company_1/images/contract_001_page_1.png",
      "layout_blocks": [
        {
          "type": "title",
          "text": "Manufacturing Agreement",
          "bbox": [50, 50, 550, 100],
          "confidence": 0.95
        }
      ],
      "tables": [
        {
          "table_id": "table_1_1",
          "headers": ["Item", "Quantity", "Price"],
          "rows": [
            ["Widget A", "100", "$10"],
            ["Widget B", "200", "$20"]
          ],
          "bbox": [50, 200, 550, 400]
        }
      ],
      "figures": [
        {
          "figure_id": "figure_1_1",
          "caption": "Product Diagram",
          "bbox": [50, 450, 550, 650]
        }
      ]
    }
  ]
}
```

### Neo4j Graph Schema

**Nodes:**
- `Company` - Company metadata
- `Document` - PDF document
- `Page` - Individual page with screenshot path
- `Chunk` - Text chunk with embedding (768-dim)
- `Entity` - Extracted entity (person, organization, concept)
- `Table` - Table structure
- `Figure` - Figure/image

**Relationships:**
- `(Document)-[:BELONGS_TO]->(Company)`
- `(Page)-[:IN_DOCUMENT]->(Document)`
- `(Chunk)-[:FROM_PAGE]->(Page)`
- `(Entity)-[:MENTIONED_IN]->(Chunk)`
- `(Entity)-[:RELATES_TO {relation, evidence}]->(Entity)`

## Performance Tips

### For Faster Processing

1. **Use paid Gemini tier** - Higher rate limits (60 RPM vs 15 RPM)
2. **Process companies in parallel** - Run separate scripts for each company
3. **Cache API responses** - Save Gemini outputs to avoid re-processing
4. **Reduce chunk size** - Fewer chunks = fewer API calls

### For Better Quality

1. **Use higher DPI screenshots** - Edit `SCREENSHOT_DPI = 300` to 600
2. **Adjust chunking** - Edit `CHUNK_SIZE` and `CHUNK_OVERLAP` in build script
3. **Improve triplet extraction** - Edit Gemini prompt in `extract_triplets()`
4. **Add table parsing** - Enhance `_extract_tables()` for complex formats

## Common Issues

### Issue: "No PDF files found"
**Solution:** Verify PDFs are in `data/{company_id}/pdfs/` directory

### Issue: "Gemini API key not found"
**Solution:** Set `GEMINI_API_KEY` in `.env` file

### Issue: "Neo4j connection refused"
**Solution:** 
```bash
docker-compose down
docker-compose up -d
# Wait 30 seconds
```

### Issue: "Rate limit exceeded"
**Solution:** Script has built-in delays. For large batches, consider paid tier.

### Issue: "Vector index not found"
**Solution:** Index takes time to build. Wait a few minutes or re-run schema setup:
```python
builder = KnowledgeGraphBuilder()
builder.setup_schema()
```

## Testing

### Phase 3: Test RAG System

Use `test_query.py` to test the system directly without starting the API:

```bash
python3 scripts/test_query.py "Your question here"
```

**Examples:**
```bash
# Test with specific question
python3 scripts/test_query.py "What happened to NSF's Total Liabilities in FY 2011?"

# Run with default question
python3 scripts/test_query.py
```

**What it does:**
- Tests hybrid retrieval (vector + text + graph search)
- Shows retrieved chunks with relevance scores
- Generates answer using Gemini
- Displays sources and citations
- Verifies end-to-end pipeline works

## Next Steps

After successful extraction and graph building:

1. Verify data in Neo4j Browser
2. Test with `test_query.py`
3. Start the FastAPI backend (see `backend/README.md`)
4. Start the Streamlit frontend (see `frontend/README.md`)
5. Test queries through the UI

## Logs

Scripts generate detailed logs:
- `extraction.log` - PDF extraction progress and errors
- `graph_builder.log` - Graph building progress and API calls

Check these files for troubleshooting.
