"""
Retrieval and Query Relevance Evaluation
Metrics: Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG), Query response time
"""

import logging
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics"""
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_k: Dict[int, float]  # NDCG at different k values
    avg_response_time: float  # Average query response time in seconds
    precision_at_k: Dict[int, float]  # Precision at k
    recall_at_k: Dict[int, float]  # Recall at k


class RetrievalEvaluator:
    """
    Evaluates retrieval and query relevance.
    Measures MRR, NDCG, response time, and factuality/hallucination rates.
    """
    
    def __init__(self, rag_engine=None):
        """
        Initialize retrieval evaluator
        
        Args:
            rag_engine: RAGEngine instance for querying
        """
        self.rag_engine = rag_engine
        logger.info("Retrieval Evaluator initialized")
    
    def calculate_mrr(self, query_results: List[Dict[str, Any]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        
        Args:
            query_results: List of query evaluation results, each with 'ranks' list
                          indicating positions of relevant documents (1-indexed)
            
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for result in query_results:
            ranks = result.get("ranks", [])
            if ranks:
                # MRR uses the first relevant document's rank
                first_relevant_rank = min(ranks)
                reciprocal_ranks.append(1.0 / first_relevant_rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_ndcg_at_k(
        self,
        relevance_scores: List[int],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k
        
        Args:
            relevance_scores: List of relevance scores for retrieved documents (in order)
            k: Cutoff rank
            
        Returns:
            NDCG@k score
        """
        # Truncate to k
        scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = sum(score / np.log2(i + 2) for i, score in enumerate(scores))
        
        # Calculate IDCG (ideal DCG - sorted in descending order)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        # NDCG = DCG / IDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_precision_at_k(
        self,
        relevant_doc_ids: Set[str],
        retrieved_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision at k
        
        Args:
            relevant_doc_ids: Set of relevant document/chunk IDs
            retrieved_doc_ids: List of retrieved document/chunk IDs (in order)
            k: Cutoff rank
            
        Returns:
            Precision@k score
        """
        retrieved_at_k = retrieved_doc_ids[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_doc_ids)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def calculate_recall_at_k(
        self,
        relevant_doc_ids: Set[str],
        retrieved_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall at k
        
        Args:
            relevant_doc_ids: Set of relevant document/chunk IDs
            retrieved_doc_ids: List of retrieved document/chunk IDs (in order)
            k: Cutoff rank
            
        Returns:
            Recall@k score
        """
        retrieved_at_k = retrieved_doc_ids[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_doc_ids)
        return relevant_retrieved / len(relevant_doc_ids) if len(relevant_doc_ids) > 0 else 0.0
    
    def evaluate_query(
        self,
        query: str,
        company_id: str,
        ground_truth: Dict[str, Any],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Evaluate a single query
        
        Args:
            query: Query text
            company_id: Company identifier
            ground_truth: Dictionary with 'relevant_chunk_ids' and optional 'relevance_scores'
            k_values: List of k values for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        if not self.rag_engine:
            raise ValueError("RAGEngine not initialized")
        
        # Measure response time
        start_time = time.time()
        
        # Retrieve chunks
        chunks = self.rag_engine.hybrid_retrieval(query, company_id)
        
        response_time = time.time() - start_time
        
        # Get retrieved chunk IDs
        retrieved_chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        
        # Get ground truth
        relevant_chunk_ids = set(ground_truth.get("relevant_chunk_ids", []))
        relevance_scores = ground_truth.get("relevance_scores", {})
        
        # Calculate metrics at different k values
        results = {
            "query": query,
            "response_time": response_time,
            "retrieved_count": len(retrieved_chunk_ids),
            "relevant_count": len(relevant_chunk_ids),
            "metrics_at_k": {}
        }
        
        for k in k_values:
            # Precision and Recall
            precision = self.calculate_precision_at_k(relevant_chunk_ids, retrieved_chunk_ids, k)
            recall = self.calculate_recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, k)
            
            # NDCG (if relevance scores provided)
            if relevance_scores:
                scores_list = [relevance_scores.get(chunk_id, 0) for chunk_id in retrieved_chunk_ids]
                ndcg = self.calculate_ndcg_at_k(scores_list, k)
            else:
                # Binary relevance (0 or 1)
                scores_list = [1 if chunk_id in relevant_chunk_ids else 0 for chunk_id in retrieved_chunk_ids]
                ndcg = self.calculate_ndcg_at_k(scores_list, k)
            
            results["metrics_at_k"][k] = {
                "precision": precision,
                "recall": recall,
                "ndcg": ndcg
            }
        
        # Calculate MRR
        ranks = [i + 1 for i, chunk_id in enumerate(retrieved_chunk_ids) if chunk_id in relevant_chunk_ids]
        results["mrr"] = 1.0 / min(ranks) if ranks else 0.0
        results["ranks"] = ranks
        
        return results
    
    def evaluate_question_answering(
        self,
        query: str,
        company_id: str,
        ground_truth: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end question answering
        
        Args:
            query: Query text
            company_id: Company identifier
            ground_truth: Dictionary with 'expected_answer' and optional 'factual_claims'
            
        Returns:
            Dictionary with QA evaluation results
        """
        if not self.rag_engine:
            raise ValueError("RAGEngine not initialized")
        
        # Measure response time
        start_time = time.time()
        
        # Generate answer
        answer, citations = self.rag_engine.query(query, company_id)
        
        response_time = time.time() - start_time
        
        # Check factuality (simple keyword matching - can be enhanced with LLM)
        expected_answer = ground_truth.get("expected_answer", "")
        factual_claims = ground_truth.get("factual_claims", [])
        
        # Simple factuality check (can be enhanced)
        factuality_score = 0.0
        if expected_answer:
            # Simple overlap check
            answer_words = set(answer.lower().split())
            expected_words = set(expected_answer.lower().split())
            if expected_words:
                factuality_score = len(answer_words & expected_words) / len(expected_words)
        
        # Check for hallucinations (citations without relevant chunks)
        relevant_chunk_ids = set(ground_truth.get("relevant_chunk_ids", []))
        cited_chunk_ids = {citation.chunk_id for citation in citations}
        
        hallucinated_citations = cited_chunk_ids - relevant_chunk_ids
        hallucination_rate = len(hallucinated_citations) / len(cited_chunk_ids) if cited_chunk_ids else 0.0
        
        return {
            "query": query,
            "answer": answer,
            "response_time": response_time,
            "factuality_score": factuality_score,
            "hallucination_rate": hallucination_rate,
            "hallucinated_citations": list(hallucinated_citations),
            "citation_count": len(citations)
        }
    
    def compare_with_baseline(
        self,
        queries: List[Dict[str, Any]],
        company_id: str
    ) -> Dict[str, Any]:
        """
        Compare hybrid retrieval with baseline keyword-based retrieval
        
        Args:
            queries: List of query dictionaries with 'query' and 'ground_truth'
            company_id: Company identifier
            
        Returns:
            Dictionary comparing hybrid vs baseline
        """
        if not self.rag_engine:
            raise ValueError("RAGEngine not initialized")
        
        hybrid_results = []
        baseline_results = []
        
        for query_data in queries:
            query = query_data["query"]
            ground_truth = query_data["ground_truth"]
            relevant_chunk_ids = set(ground_truth.get("relevant_chunk_ids", []))
            
            # Hybrid retrieval
            hybrid_chunks = self.rag_engine.hybrid_retrieval(query, company_id)
            hybrid_retrieved = [chunk["chunk_id"] for chunk in hybrid_chunks]
            hybrid_precision_5 = self.calculate_precision_at_k(relevant_chunk_ids, hybrid_retrieved, 5)
            hybrid_recall_5 = self.calculate_recall_at_k(relevant_chunk_ids, hybrid_retrieved, 5)
            
            # Baseline keyword-based (full-text search only)
            baseline_chunks = self.rag_engine.fulltext_search(query, company_id, top_k=5)
            baseline_retrieved = [chunk["chunk_id"] for chunk in baseline_chunks]
            baseline_precision_5 = self.calculate_precision_at_k(relevant_chunk_ids, baseline_retrieved, 5)
            baseline_recall_5 = self.calculate_recall_at_k(relevant_chunk_ids, baseline_retrieved, 5)
            
            hybrid_results.append({
                "precision_5": hybrid_precision_5,
                "recall_5": hybrid_recall_5
            })
            
            baseline_results.append({
                "precision_5": baseline_precision_5,
                "recall_5": baseline_recall_5
            })
        
        return {
            "hybrid": {
                "avg_precision_5": np.mean([r["precision_5"] for r in hybrid_results]),
                "avg_recall_5": np.mean([r["recall_5"] for r in hybrid_results])
            },
            "baseline": {
                "avg_precision_5": np.mean([r["precision_5"] for r in baseline_results]),
                "avg_recall_5": np.mean([r["recall_5"] for r in baseline_results])
            },
            "improvement": {
                "precision_5": np.mean([r["precision_5"] for r in hybrid_results]) - np.mean([r["precision_5"] for r in baseline_results]),
                "recall_5": np.mean([r["recall_5"] for r in hybrid_results]) - np.mean([r["recall_5"] for r in baseline_results])
            }
        }
    
    def evaluate_test_set(
        self,
        test_queries_file: Path,
        company_id: str,
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Evaluate on a test set of queries
        
        Args:
            test_queries_file: Path to JSON file with test queries and ground truth
            company_id: Company identifier
            output_file: Optional path to save results
            
        Returns:
            Dictionary with evaluation results
        """
        # Load test queries
        with open(test_queries_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        queries = test_data.get("queries", [])
        
        all_results = []
        response_times = []
        
        for query_data in queries:
            query = query_data["query"]
            ground_truth = query_data["ground_truth"]
            
            logger.info(f"Evaluating query: {query[:50]}...")
            
            result = self.evaluate_query(query, company_id, ground_truth)
            all_results.append(result)
            response_times.append(result["response_time"])
        
        # Calculate aggregate metrics
        mrr = self.calculate_mrr(all_results)
        
        # Average metrics at different k values
        k_values = [1, 3, 5, 10]
        avg_metrics = {}
        for k in k_values:
            avg_metrics[k] = {
                "precision": np.mean([r["metrics_at_k"][k]["precision"] for r in all_results]),
                "recall": np.mean([r["metrics_at_k"][k]["recall"] for r in all_results]),
                "ndcg": np.mean([r["metrics_at_k"][k]["ndcg"] for r in all_results])
            }
        
        summary = {
            "total_queries": len(queries),
            "mrr": mrr,
            "avg_response_time": np.mean(response_times),
            "avg_metrics_at_k": avg_metrics,
            "results": all_results
        }
        
        # Save results if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation results to {output_file}")
        
        return summary

