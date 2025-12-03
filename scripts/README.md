# Processing & Testing Scripts

This directory contains scripts for offline PDF extraction, knowledge graph building, and testing.

## Scripts Overview

### 1. `extract_simple.py`
Extracts text from PDFs using PyMuPDF and generates page screenshots at 300 DPI.

### 2. `build_knowledge_graph.py`
Populates Neo4j with extracted content, generates embeddings (local), and extracts knowledge triplets using Gemini.

### 3. `test_query.py`
Tests the RAG system directly without running the API. Useful for quick validation.

### 4. `extract_docling.py`
Extracts text from PDFs using IBM's Docling.

