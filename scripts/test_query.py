#!/usr/bin/env python3
"""
Test script to query the RAG system directly without API
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.rag_engine import RAGEngine
from app.models import QueryRequest, QueryResponse
import json

def test_query(question: str, company_id: str = "company_1"):
    """
    Test a query against the RAG engine
    
    Args:
        question: User question
        company_id: Company to query
    """
    print("=" * 70)
    print("RAG SYSTEM TEST")
    print("=" * 70)
    print(f"\nQuestion: {question}")
    print(f"Company: {company_id}")
    print("\n" + "-" * 70)
    
    # Initialize RAG engine
    print("Initializing RAG engine...")
    engine = RAGEngine()
    
    try:
        # Perform hybrid retrieval (embedding generation happens inside)
        print("Performing hybrid retrieval (vector + text + graph)...")
        chunks = engine.hybrid_retrieval(question, company_id)
        print(f"Retrieved {len(chunks)} relevant chunks")
        
        # Show top chunks
        print("\nTop 3 relevant chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n  [{i}] Score: {chunk['score']:.3f} | Doc: {chunk['doc_id']} | Page: {chunk['page_num']}")
            print(f"      {chunk['text'][:120]}...")
        
        # Generate answer
        print("\n" + "-" * 70)
        print("Generating answer with Gemini...")
        answer = engine.generate_answer(question, chunks)
        
        # Display results
        print("\n" + "=" * 70)
        print("ANSWER:")
        print("=" * 70)
        print(answer)
        
        print("\n" + "=" * 70)
        print(f"SOURCE CHUNKS ({len(chunks)}):")
        print("=" * 70)
        for i, chunk in enumerate(chunks, 1):
            print(f"\n[{i}] Document: {chunk['doc_id']} | Page: {chunk['page_num']} | Score: {chunk['score']:.3f}")
            print(f"    {chunk['text'][:150]}...")
        
        print("\n" + "=" * 70)
        print("✓ TEST COMPLETE")
        print("=" * 70)
        
    finally:
        engine.close()


def main():
    """Main function"""
    # Default test questions
    test_questions = [
        "What is nanotechnology?",
        "What are the main research areas?",
        "What is the budget allocation?",
    ]
    
    # Use command line argument or default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        print("Available test questions:")
        for i, q in enumerate(test_questions, 1):
            print(f"  {i}. {q}")
        print("\nUsing default question...")
        question = test_questions[0]
    
    test_query(question)


if __name__ == "__main__":
    main()
