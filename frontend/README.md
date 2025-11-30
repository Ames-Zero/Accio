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

## Usage

### 1. Select Company

Use the sidebar dropdown (currently only company_1: Acme Corp with 24 US federal documents)

### 2. Ask Questions

Try the example questions in the sidebar:
- "What happened to NSF's Total Liabilities in FY 2011?"
- "What is the NNI's approach to addressing ethical questions?"
- "Where must accessible parking spaces be located?"
- "What are nanotechnology's potential environmental impacts?"
- "Summarize NSF's financial position"

Or type your own questions about:
- NSF financial statements
- Nanotechnology research and ethics (NNI)
- Accessibility regulations
- Federal budget allocations

### 3. View Citations

Each answer includes up to 5 expandable citations showing:
- **Screenshot** - Document page with highlighted bbox (red rectangle at 300 DPI scale)
- **Document name** - Source PDF filename
- **Page number** - Specific page location
- **Text excerpt** - First 500 chars of relevant text
- **Relevance score** - Retrieval confidence (0.500-1.000)

### 4. Clear Chat

Click "Clear Chat History" in sidebar to reset conversation.

## Interface Layout

```
┌─────────────────────────────────────────────┐
│  Sidebar                │  Main Chat Area   │
│  ─────────              │  ─────────────    │
│  🏢 Company Selector    │  💬 Chat Messages │
│                         │                   │
│  Company Description    │  User Question    │
│                         │  ↓                │
│  💡 Example Questions   │  Bot Answer       │
│  - Question 1           │  with [1][2]      │
│  - Question 2           │                   │
│                         │  📎 Citations     │
│  🗑️ Clear Chat         │  [Screenshot][Text]│
└─────────────────────────────────────────────┘
```

## Configuration

### API URL

Edit `streamlit_app.py`:

```python
API_BASE_URL = "http://localhost:8000"  # Change if backend runs elsewhere
```

### Page Configuration

```python
st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

## Features Explained

### Company Selector

Dynamically loads companies from backend `/companies` endpoint.

**Data source:** `data/companies.yaml`

### Chat History

Stored in `st.session_state.messages` (in-memory, resets on page refresh).

**Format:**
```python
{
    "role": "user" | "assistant",
    "content": "message text",
    "citations": [...]  # Only for assistant
}
```

### Citation Display

Uses `PIL.ImageDraw` to highlight bounding boxes:

```python
def highlight_bbox(image_path, bbox):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline="red", width=5)
    return img
```

### Example Questions

Hardcoded suggestions. Customize in `streamlit_app.py`:

```python
example_questions = [
    "What are the payment terms?",
    "What is the warranty period?",
    # Add more...
]
```

## Troubleshooting

### Error: "Failed to load companies"

**Cause:** Backend not running or wrong URL

**Solution:**
```bash
# Check backend is running
curl http://localhost:8000/companies

# Start backend
cd backend && uvicorn app.main:app --reload
```

### Error: "Could not load image"

**Cause:** Image path incorrect or screenshots not generated

**Solution:**
```bash
# Verify screenshots exist
ls data/company_1/images/

# Re-run extraction if needed
python scripts/extract_with_docling.py
```

### Citations not showing

**Cause:** No matching documents in knowledge graph

**Solution:**
```bash
# Check Neo4j has data
# Open http://localhost:7474
MATCH (c:Chunk) RETURN count(c);

# If empty, run graph builder
python scripts/build_knowledge_graph.py
```

### Bounding boxes not visible

**Cause:** Coordinates may be outside image bounds

**Solution:** Check bbox values in Neo4j. Docling sometimes returns normalized coordinates (0-1 range).

## Customization

### Change Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Add Features

**Multi-turn conversation:**
```python
# Add context from previous messages
context = "\n".join([m["content"] for m in st.session_state.messages[-3:]])
question_with_context = f"{context}\n\n{user_question}"
```

**Export chat:**
```python
if st.sidebar.button("Export Chat"):
    chat_text = "\n\n".join([
        f"{m['role']}: {m['content']}"
        for m in st.session_state.messages
    ])
    st.download_button("Download", chat_text, "chat.txt")
```

### Improve UX

**Loading indicators:**
```python
with st.spinner("Analyzing documents..."):
    result = query_api(question, company_id)
```

**Error handling:**
```python
try:
    result = query_api(question, company_id)
except Exception as e:
    st.error(f"Query failed: {e}")
    st.stop()
```

## Performance Tips

### Faster Image Loading

Cache images in session state:

```python
if 'image_cache' not in st.session_state:
    st.session_state.image_cache = {}

if image_path not in st.session_state.image_cache:
    img = highlight_bbox(image_path, bbox)
    st.session_state.image_cache[image_path] = img
```

### Limit Chat History

```python
# Keep only last 10 messages
if len(st.session_state.messages) > 10:
    st.session_state.messages = st.session_state.messages[-10:]
```

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect repository
4. Set environment variables (API_BASE_URL)
5. Deploy

**Note:** Backend must be publicly accessible

### Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit_app.py .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Run:
```bash
docker build -t pdf-frontend .
docker run -p 8501:8501 -e API_BASE_URL=http://backend:8000 pdf-frontend
```

## Next Steps

- Customize example questions for your domain
- Adjust citation display layout
- Add multi-language support
- Implement conversation memory
- Add export functionality

## Support

For issues:
1. Check backend is running: http://localhost:8000/docs
2. Verify data in Neo4j: http://localhost:7474
3. Check browser console for errors (F12)
4. Review Streamlit logs in terminal
