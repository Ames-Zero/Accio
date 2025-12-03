"""
CI/CD Integration for Continuous Evaluation
Automates evaluation pipeline and metric tracking
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import os

try:
    from .evaluation_runner import EvaluationRunner
except ImportError:
    from evaluation_runner import EvaluationRunner

logger = logging.getLogger(__name__)


def load_previous_metrics(metrics_file: Path) -> Dict[str, Any]:
    """Load previous evaluation metrics for comparison"""
    if not metrics_file.exists():
        return {}
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metrics(metrics: Dict[str, Any], metrics_file: Path):
    """Save evaluation metrics"""
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing metrics
    all_metrics = load_previous_metrics(metrics_file)
    
    # Add new metrics with timestamp
    timestamp = datetime.now().isoformat()
    all_metrics[timestamp] = metrics
    
    # Keep only last 100 evaluations
    if len(all_metrics) > 100:
        sorted_keys = sorted(all_metrics.keys())
        for key in sorted_keys[:-100]:
            del all_metrics[key]
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)


def check_metric_thresholds(
    current_metrics: Dict[str, Any],
    thresholds: Dict[str, float]
) -> Dict[str, bool]:
    """
    Check if metrics meet thresholds
    
    Args:
        current_metrics: Current evaluation metrics
        thresholds: Dictionary of metric thresholds
        
    Returns:
        Dictionary indicating which thresholds are met
    """
    results = {}
    
    # OCR thresholds
    if "ocr_wer_max" in thresholds:
        ocr = current_metrics.get("evaluations", {}).get("ocr", {})
        summary = ocr.get("summary", {})
        wer = summary.get("overall_avg_wer", 1.0)
        results["ocr_wer"] = wer <= thresholds["ocr_wer_max"]
    
    # Entity/Relation thresholds
    if "entity_f1_min" in thresholds:
        er = current_metrics.get("evaluations", {}).get("entity_relation", {})
        summary = er.get("summary", {})
        f1 = summary.get("avg_entity_f1", 0.0)
        results["entity_f1"] = f1 >= thresholds["entity_f1_min"]
    
    if "relation_f1_min" in thresholds:
        er = current_metrics.get("evaluations", {}).get("entity_relation", {})
        summary = er.get("summary", {})
        f1 = summary.get("avg_relation_f1", 0.0)
        results["relation_f1"] = f1 >= thresholds["relation_f1_min"]
    
    # Retrieval thresholds
    if "mrr_min" in thresholds:
        ret = current_metrics.get("evaluations", {}).get("retrieval", {})
        mrr = ret.get("mrr", 0.0)
        results["mrr"] = mrr >= thresholds["mrr_min"]
    
    if "response_time_max" in thresholds:
        ret = current_metrics.get("evaluations", {}).get("retrieval", {})
        response_time = ret.get("avg_response_time", float('inf'))
        results["response_time"] = response_time <= thresholds["response_time_max"]
    
    return results


def run_ci_evaluation(
    company_id: str,
    thresholds_file: Path = None,
    metrics_file: Path = None
) -> int:
    """
    Run evaluation in CI/CD context
    
    Args:
        company_id: Company identifier
        thresholds_file: Path to thresholds JSON file
        metrics_file: Path to metrics history file
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup paths
    if metrics_file is None:
        metrics_file = Path("evaluations/metrics_history.json")
    
    if thresholds_file is None:
        thresholds_file = Path("evaluations/thresholds.json")
    
    # Load thresholds
    thresholds = {}
    if thresholds_file.exists():
        with open(thresholds_file, 'r', encoding='utf-8') as f:
            thresholds = json.load(f)
    
    # Initialize runner
    runner = EvaluationRunner()
    
    # Get configuration from environment or defaults
    extracted_files_dir = Path(os.getenv("EXTRACTED_FILES_DIR", "data/processed"))
    test_queries_file = Path(os.getenv("TEST_QUERIES_FILE", "evaluations/ground_truth/test_queries.json"))
    
    # Find extracted files
    extracted_files = list((extracted_files_dir / company_id).glob("*.json")) if (extracted_files_dir / company_id).exists() else []
    
    # Get doc IDs from environment or use defaults
    doc_ids = os.getenv("DOC_IDS", "").split() if os.getenv("DOC_IDS") else None
    
    # Import dependencies
    from neo4j import GraphDatabase
    from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from app.rag_engine import RAGEngine
    
    neo4j_driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    rag_engine = RAGEngine()
    
    try:
        # Run full evaluation
        logger.info("Running CI/CD evaluation pipeline...")
        
        results = runner.run_full_evaluation(
            company_id=company_id,
            extracted_files=extracted_files if extracted_files else None,
            doc_ids=doc_ids,
            test_queries_file=test_queries_file if test_queries_file.exists() else None,
            neo4j_driver=neo4j_driver,
            rag_engine=rag_engine
        )
        
        # Save metrics
        save_metrics(results, metrics_file)
        
        # Check thresholds
        if thresholds:
            threshold_results = check_metric_thresholds(results, thresholds)
            
            logger.info("Threshold Check Results:")
            all_passed = True
            for metric, passed in threshold_results.items():
                status = "PASS" if passed else "FAIL"
                logger.info(f"  {metric}: {status}")
                if not passed:
                    all_passed = False
            
            if not all_passed:
                logger.error("Some metrics did not meet thresholds!")
                return 1
        
        logger.info("Evaluation pipeline completed successfully")
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}", exc_info=True)
        return 1
    
    finally:
        neo4j_driver.close()
        rag_engine.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    company_id = os.getenv("COMPANY_ID", "company_1")
    
    exit_code = run_ci_evaluation(company_id)
    sys.exit(exit_code)

