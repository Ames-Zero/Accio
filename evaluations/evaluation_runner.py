"""
Main Evaluation Runner
Orchestrates all evaluation modules and provides CI/CD integration
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import argparse

from .ocr_evaluation import OCREvaluator
from .entity_evaluation import EntityRelationEvaluator
from .kg_evaluation import KnowledgeGraphEvaluator
from .retrieval_evaluation import RetrievalEvaluator

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Main evaluation runner that orchestrates all evaluation modules.
    Supports CI/CD integration and continuous evaluation.
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        ground_truth_dir: Path = None
    ):
        """
        Initialize evaluation runner
        
        Args:
            output_dir: Directory to save evaluation results
            ground_truth_dir: Base directory for ground truth data
        """
        self.output_dir = output_dir or Path("evaluations/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ground_truth_dir = ground_truth_dir or Path("evaluations/ground_truth")
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Evaluation Runner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Ground truth directory: {self.ground_truth_dir}")
    
    def run_ocr_evaluation(
        self,
        extracted_files: List[Path],
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Run OCR and text extraction evaluation
        
        Args:
            extracted_files: List of paths to extracted JSON files
            output_file: Optional output file path
            
        Returns:
            Evaluation results
        """
        logger.info("="*60)
        logger.info("Running OCR Evaluation")
        logger.info("="*60)
        
        evaluator = OCREvaluator(
            ground_truth_dir=self.ground_truth_dir / "ocr"
        )
        
        if output_file is None:
            output_file = self.output_dir / f"ocr_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = evaluator.evaluate_batch(extracted_files, output_file)
        
        logger.info(f"OCR Evaluation complete. Avg WER: {results['summary']['overall_avg_wer']:.4f}, "
                   f"Avg CER: {results['summary']['overall_avg_cer']:.4f}")
        
        return results
    
    def run_entity_evaluation(
        self,
        doc_ids: List[str],
        neo4j_driver = None,
        cross_validate: bool = False,
        k_folds: int = 5,
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Run entity and relation extraction evaluation
        
        Args:
            doc_ids: List of document identifiers
            neo4j_driver: Neo4j driver instance
            cross_validate: Whether to perform cross-validation
            k_folds: Number of folds for cross-validation
            output_file: Optional output file path
            
        Returns:
            Evaluation results
        """
        logger.info("="*60)
        logger.info("Running Entity/Relation Evaluation")
        logger.info("="*60)
        
        evaluator = EntityRelationEvaluator(
            ground_truth_dir=self.ground_truth_dir / "entities"
        )
        
        if output_file is None:
            output_file = self.output_dir / f"entity_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if cross_validate and neo4j_driver:
            results = evaluator.cross_validate(doc_ids, neo4j_driver, k_folds)
        else:
            # Evaluate individual documents
            results = {
                "documents": [],
                "summary": {}
            }
            
            for doc_id in doc_ids:
                if neo4j_driver:
                    doc_result = evaluator.evaluate_from_neo4j(doc_id, neo4j_driver)
                else:
                    logger.warning(f"No Neo4j driver provided, skipping {doc_id}")
                    continue
                
                results["documents"].append(doc_result)
            
            # Calculate summary
            if results["documents"]:
                results["summary"] = {
                    "avg_entity_precision": sum(d["entities"]["precision"] for d in results["documents"]) / len(results["documents"]),
                    "avg_entity_recall": sum(d["entities"]["recall"] for d in results["documents"]) / len(results["documents"]),
                    "avg_entity_f1": sum(d["entities"]["f1_score"] for d in results["documents"]) / len(results["documents"]),
                    "avg_relation_precision": sum(d["relations"]["precision"] for d in results["documents"]) / len(results["documents"]),
                    "avg_relation_recall": sum(d["relations"]["recall"] for d in results["documents"]) / len(results["documents"]),
                    "avg_relation_f1": sum(d["relations"]["f1_score"] for d in results["documents"]) / len(results["documents"])
                }
        
        # Save results
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Entity/Relation Evaluation complete. Results saved to {output_file}")
        
        return results
    
    def run_kg_evaluation(
        self,
        company_id: str,
        expected_entities: List[str] = None,
        expected_relations: List[tuple] = None,
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Run knowledge graph construction evaluation
        
        Args:
            company_id: Company identifier
            expected_entities: Optional list of expected entities
            expected_relations: Optional list of expected relations
            output_file: Optional output file path
            
        Returns:
            Evaluation results
        """
        logger.info("="*60)
        logger.info("Running Knowledge Graph Evaluation")
        logger.info("="*60)
        
        evaluator = KnowledgeGraphEvaluator()
        
        try:
            results = evaluator.evaluate_company(
                company_id,
                expected_entities,
                expected_relations
            )
            
            if output_file is None:
                output_file = self.output_dir / f"kg_evaluation_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"KG Evaluation complete. Results saved to {output_file}")
            
            return results
        
        finally:
            evaluator.close()
    
    def run_retrieval_evaluation(
        self,
        test_queries_file: Path,
        company_id: str,
        rag_engine = None,
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Run retrieval and query relevance evaluation
        
        Args:
            test_queries_file: Path to test queries JSON file
            company_id: Company identifier
            rag_engine: RAGEngine instance
            output_file: Optional output file path
            
        Returns:
            Evaluation results
        """
        logger.info("="*60)
        logger.info("Running Retrieval Evaluation")
        logger.info("="*60)
        
        evaluator = RetrievalEvaluator(rag_engine=rag_engine)
        
        if output_file is None:
            output_file = self.output_dir / f"retrieval_evaluation_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results = evaluator.evaluate_test_set(test_queries_file, company_id, output_file)
        
        logger.info(f"Retrieval Evaluation complete. MRR: {results['mrr']:.4f}, "
                   f"Avg response time: {results['avg_response_time']:.2f}s")
        
        return results
    
    def run_full_evaluation(
        self,
        company_id: str,
        extracted_files: List[Path] = None,
        doc_ids: List[str] = None,
        test_queries_file: Path = None,
        neo4j_driver = None,
        rag_engine = None,
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline
        
        Args:
            company_id: Company identifier
            extracted_files: List of extracted JSON files for OCR evaluation
            doc_ids: List of document IDs for entity evaluation
            test_queries_file: Path to test queries file
            neo4j_driver: Neo4j driver instance
            rag_engine: RAGEngine instance
            output_file: Optional output file path
            
        Returns:
            Complete evaluation results
        """
        logger.info("="*60)
        logger.info("Running Full Evaluation Pipeline")
        logger.info("="*60)
        
        all_results = {
            "company_id": company_id,
            "timestamp": datetime.now().isoformat(),
            "evaluations": {}
        }
        
        # 1. OCR Evaluation
        if extracted_files:
            try:
                ocr_results = self.run_ocr_evaluation(extracted_files)
                all_results["evaluations"]["ocr"] = ocr_results
            except Exception as e:
                logger.error(f"OCR evaluation failed: {e}", exc_info=True)
                all_results["evaluations"]["ocr"] = {"error": str(e)}
        
        # 2. Entity/Relation Evaluation
        if doc_ids and neo4j_driver:
            try:
                entity_results = self.run_entity_evaluation(doc_ids, neo4j_driver)
                all_results["evaluations"]["entity_relation"] = entity_results
            except Exception as e:
                logger.error(f"Entity/Relation evaluation failed: {e}", exc_info=True)
                all_results["evaluations"]["entity_relation"] = {"error": str(e)}
        
        # 3. Knowledge Graph Evaluation
        try:
            kg_results = self.run_kg_evaluation(company_id)
            all_results["evaluations"]["knowledge_graph"] = kg_results
        except Exception as e:
            logger.error(f"KG evaluation failed: {e}", exc_info=True)
            all_results["evaluations"]["knowledge_graph"] = {"error": str(e)}
        
        # 4. Retrieval Evaluation
        if test_queries_file and rag_engine:
            try:
                retrieval_results = self.run_retrieval_evaluation(
                    test_queries_file,
                    company_id,
                    rag_engine
                )
                all_results["evaluations"]["retrieval"] = retrieval_results
            except Exception as e:
                logger.error(f"Retrieval evaluation failed: {e}", exc_info=True)
                all_results["evaluations"]["retrieval"] = {"error": str(e)}
        
        # Save complete results
        if output_file is None:
            output_file = self.output_dir / f"full_evaluation_{company_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Full evaluation complete. Results saved to {output_file}")
        
        return all_results
    
    def generate_evaluation_report(
        self,
        results_file: Path,
        output_file: Path = None
    ) -> str:
        """
        Generate human-readable evaluation report
        
        Args:
            results_file: Path to evaluation results JSON file
            output_file: Optional path to save report
            
        Returns:
            Report text
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        report_lines = [
            "="*80,
            "EVALUATION REPORT",
            "="*80,
            f"Company ID: {results.get('company_id', 'N/A')}",
            f"Timestamp: {results.get('timestamp', 'N/A')}",
            "",
        ]
        
        evaluations = results.get("evaluations", {})
        
        # OCR Evaluation
        if "ocr" in evaluations:
            ocr = evaluations["ocr"]
            if "error" not in ocr:
                summary = ocr.get("summary", {})
                report_lines.extend([
                    "OCR and Text Extraction",
                    "-"*80,
                    f"Total Documents: {summary.get('total_documents', 0)}",
                    f"Total Pages: {summary.get('total_pages', 0)}",
                    f"Average WER: {summary.get('overall_avg_wer', 0):.4f}",
                    f"Average CER: {summary.get('overall_avg_cer', 0):.4f}",
                    ""
                ])
        
        # Entity/Relation Evaluation
        if "entity_relation" in evaluations:
            er = evaluations["entity_relation"]
            if "error" not in er:
                summary = er.get("summary", {})
                if summary:
                    report_lines.extend([
                        "Entity and Relation Extraction",
                        "-"*80,
                        f"Entity Precision: {summary.get('avg_entity_precision', 0):.4f}",
                        f"Entity Recall: {summary.get('avg_entity_recall', 0):.4f}",
                        f"Entity F1-Score: {summary.get('avg_entity_f1', 0):.4f}",
                        f"Relation Precision: {summary.get('avg_relation_precision', 0):.4f}",
                        f"Relation Recall: {summary.get('avg_relation_recall', 0):.4f}",
                        f"Relation F1-Score: {summary.get('avg_relation_f1', 0):.4f}",
                        ""
                    ])
        
        # Knowledge Graph Evaluation
        if "knowledge_graph" in evaluations:
            kg = evaluations["knowledge_graph"]
            if "error" not in kg:
                report_lines.extend([
                    "Knowledge Graph Construction",
                    "-"*80,
                ])
                
                completeness = kg.get("completeness", {})
                provenance = kg.get("provenance", {})
                graph_structure = kg.get("graph_structure", {})
                
                report_lines.extend([
                    f"Entity Count: {completeness.get('entity_count', 0)}",
                    f"Relation Count: {completeness.get('relation_count', 0)}",
                    f"Entity Provenance Coverage: {provenance.get('entity_provenance_coverage', 0):.4f}",
                    f"Relation Evidence Coverage: {provenance.get('relation_evidence_coverage', 0):.4f}",
                    f"Graph Density: {graph_structure.get('graph_density', 0):.4f}",
                    f"Connectivity Ratio: {graph_structure.get('connectivity_ratio', 0):.4f}",
                    ""
                ])
        
        # Retrieval Evaluation
        if "retrieval" in evaluations:
            ret = evaluations["retrieval"]
            if "error" not in ret:
                report_lines.extend([
                    "Retrieval and Query Relevance",
                    "-"*80,
                    f"Total Queries: {ret.get('total_queries', 0)}",
                    f"Mean Reciprocal Rank (MRR): {ret.get('mrr', 0):.4f}",
                    f"Average Response Time: {ret.get('avg_response_time', 0):.2f}s",
                    ""
                ])
                
                avg_metrics = ret.get("avg_metrics_at_k", {})
                for k, metrics in sorted(avg_metrics.items()):
                    report_lines.append(
                        f"  @{k}: Precision={metrics.get('precision', 0):.4f}, "
                        f"Recall={metrics.get('recall', 0):.4f}, "
                        f"NDCG={metrics.get('ndcg', 0):.4f}"
                    )
        
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        
        return report_text


def main():
    """Command-line interface for evaluation runner"""
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--company-id", required=True, help="Company identifier")
    parser.add_argument("--evaluation-type", choices=["ocr", "entity", "kg", "retrieval", "full"],
                       default="full", help="Type of evaluation to run")
    parser.add_argument("--extracted-files", nargs="+", help="Paths to extracted JSON files")
    parser.add_argument("--doc-ids", nargs="+", help="Document IDs for evaluation")
    parser.add_argument("--test-queries", help="Path to test queries JSON file")
    parser.add_argument("--output-dir", default="evaluations/results", help="Output directory")
    parser.add_argument("--cross-validate", action="store_true", help="Perform cross-validation")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    runner = EvaluationRunner(output_dir=Path(args.output_dir))
    
    # Import here to avoid circular dependencies
    from neo4j import GraphDatabase
    from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from app.rag_engine import RAGEngine
    
    neo4j_driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    
    rag_engine = RAGEngine()
    
    try:
        if args.evaluation_type == "ocr":
            if not args.extracted_files:
                logger.error("--extracted-files required for OCR evaluation")
                sys.exit(1)
            extracted_files = [Path(f) for f in args.extracted_files]
            runner.run_ocr_evaluation(extracted_files)
        
        elif args.evaluation_type == "entity":
            if not args.doc_ids:
                logger.error("--doc-ids required for entity evaluation")
                sys.exit(1)
            runner.run_entity_evaluation(args.doc_ids, neo4j_driver, cross_validate=args.cross_validate)
        
        elif args.evaluation_type == "kg":
            runner.run_kg_evaluation(args.company_id)
        
        elif args.evaluation_type == "retrieval":
            if not args.test_queries:
                logger.error("--test-queries required for retrieval evaluation")
                sys.exit(1)
            runner.run_retrieval_evaluation(Path(args.test_queries), args.company_id, rag_engine)
        
        elif args.evaluation_type == "full":
            extracted_files = [Path(f) for f in args.extracted_files] if args.extracted_files else None
            test_queries = Path(args.test_queries) if args.test_queries else None
            
            runner.run_full_evaluation(
                args.company_id,
                extracted_files,
                args.doc_ids,
                test_queries,
                neo4j_driver,
                rag_engine
            )
    
    finally:
        neo4j_driver.close()
        rag_engine.close()


if __name__ == "__main__":
    main()

