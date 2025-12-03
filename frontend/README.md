# Streamlit Chat Frontend

Interactive chat interface for PDF knowledge Q&A system with visual grounding.

## Features

- **Company selector** - Choose which company's documents to query (currently: company_1)
- **Chat interface** - Natural conversation with message history
- **Citation display** - Top 5 sources with PDF screenshots and bbox overlays
- **Example questions** - Pre-written queries based on actual documents
- **Screenshot highlighting** - Red boxes show exact text locations (scaled for 300 DPI)
- **Relevance scores** - See how relevant each source is (0-1 scale)

## Installation

```bash
cd frontend
pip install -r requirements.txt
```

## Running the Application

### Prerequisites

1. **Backend must be running:**
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Neo4j must be running:**
   ```bash
   docker-compose up -d
   ```

### Start Streamlit

```bash
cd frontend
streamlit run streamlit_app.py
```

Application opens at: http://localhost:8501
