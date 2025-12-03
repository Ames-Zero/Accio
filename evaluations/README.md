# Evaluation Framework

This directory contains a comprehensive evaluation framework for the RAG system, covering all stages of the pipeline from OCR to retrieval.

## Structure

- `ocr_evaluation.py` - OCR and text extraction evaluation (WER, CER)
- `entity_evaluation.py` - Entity and relation extraction evaluation (Precision, Recall, F1)
- `kg_evaluation.py` - Knowledge graph construction evaluation (Completeness, Correctness, Provenance)
- `retrieval_evaluation.py` - Retrieval and query relevance evaluation (MRR, NDCG, Response time)
- `evaluation_runner.py` - Main orchestrator for running evaluations
- `ground_truth/` - Directory for ground truth data
- `results/` - Directory for evaluation results

## Ground Truth Data Structure

### OCR Ground Truth
Place manually transcribed text files in `ground_truth/ocr/`:
- `{doc_id}.txt` - Full document transcription
- `{doc_id}_page_{page_num}.txt` - Page-level transcription

### Entity/Relation Ground Truth
Place JSON files in `ground_truth/entities/` with format:
```json
{
  "entities": [
    {"name": "Entity Name", "type": "PERSON"}
  ],
  "relations": [
    {"subject": "Entity 1", "predicate": "works_for", "object": "Entity 2"}
  ]
}
```

### Test Queries
Create JSON files with test queries:
```json
{
  "queries": [
    {
      "query": "What is the payment terms?",
      "ground_truth": {
        "relevant_chunk_ids": ["chunk_1", "chunk_2"],
        "relevance_scores": {"chunk_1": 2, "chunk_2": 1},
        "expected_answer": "Net 30 days"
      }
    }
  ]
}
```

## Usage

### Command Line

```bash
# Run full evaluation pipeline
python -m evaluations.evaluation_runner --company-id company_1 --evaluation-type full \
  --extracted-files data/processed/company_1/*.json \
  --doc-ids doc1 doc2 doc3 \
  --test-queries evaluations/ground_truth/test_queries.json

# Run individual evaluations
python -m evaluations.evaluation_runner --company-id company_1 --evaluation-type ocr \
  --extracted-files data/processed/company_1/*.json

python -m evaluations.evaluation_runner --company-id company_1 --evaluation-type entity \
  --doc-ids doc1 doc2 --cross-validate

python -m evaluations.evaluation_runner --company-id company_1 --evaluation-type kg

python -m evaluations.evaluation_runner --company-id company_1 --evaluation-type retrieval \
  --test-queries evaluations/ground_truth/test_queries.json
```

### Python API

```python
from evaluations import EvaluationRunner
from pathlib import Path
from neo4j import GraphDatabase
from app.rag_engine import RAGEngine

# Initialize runner
runner = EvaluationRunner()

# Run OCR evaluation
ocr_results = runner.run_ocr_evaluation(
    extracted_files=[Path("data/processed/company_1/doc1.json")]
)

# Run entity/relation evaluation
neo4j_driver = GraphDatabase.driver(...)
entity_results = runner.run_entity_evaluation(
    doc_ids=["doc1", "doc2"],
    neo4j_driver=neo4j_driver,
    cross_validate=True
)

# Run KG evaluation
kg_results = runner.run_kg_evaluation(company_id="company_1")

# Run retrieval evaluation
rag_engine = RAGEngine()
retrieval_results = runner.run_retrieval_evaluation(
    test_queries_file=Path("evaluations/ground_truth/test_queries.json"),
    company_id="company_1",
    rag_engine=rag_engine
)

# Run full evaluation
full_results = runner.run_full_evaluation(
    company_id="company_1",
    extracted_files=[...],
    doc_ids=[...],
    test_queries_file=Path("..."),
    neo4j_driver=neo4j_driver,
    rag_engine=rag_engine
)

# Generate report
report = runner.generate_evaluation_report(
    results_file=Path("evaluations/results/full_evaluation_...json"),
    output_file=Path("evaluations/results/report.txt")
)
```

## CI/CD Integration

Create a CI/CD script (e.g., `.github/workflows/evaluate.yml`):

```yaml
name: Evaluation Pipeline

on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  workflow_dispatch:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r backend/requirements.txt
      - name: Run evaluations
        run: |
          python -m evaluations.evaluation_runner \
            --company-id company_1 \
            --evaluation-type full \
            --extracted-files data/processed/company_1/*.json \
            --doc-ids doc1 doc2 \
            --test-queries evaluations/ground_truth/test_queries.json
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: evaluations/results/
```

## Metrics Explained

### OCR Metrics
- **WER (Word Error Rate)**: Percentage of words incorrectly extracted
- **CER (Character Error Rate)**: Percentage of characters incorrectly extracted

### Entity/Relation Metrics
- **Precision**: Percentage of extracted entities/relations that are correct
- **Recall**: Percentage of ground truth entities/relations that were extracted
- **F1-Score**: Harmonic mean of precision and recall

### Knowledge Graph Metrics
- **Completeness**: Percentage of expected entities/relations present in graph
- **Correctness**: Percentage of entities/relations verified as correct (manual review)
- **Provenance Coverage**: Percentage of entities/relations with source tracking
- **Graph Quality**: Connectivity, density, component analysis

### Retrieval Metrics
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first relevant result
- **NDCG@k**: Normalized Discounted Cumulative Gain at rank k
- **Precision@k**: Percentage of top-k results that are relevant
- **Recall@k**: Percentage of relevant results found in top-k
- **Response Time**: Average query processing time

## Continuous Improvement

1. **Error Logging**: All evaluation modules log errors and warnings
2. **Metric Tracking**: Results are saved in JSON format for trend analysis
3. **Iterative Refinement**: Use evaluation results to:
   - Retrain extraction models
   - Adjust chunking parameters
   - Optimize retrieval weights
   - Refine knowledge graph schema

## Requirements

Add to `requirements.txt`:
```
numpy>=1.21.0
neo4j>=5.0.0
```

