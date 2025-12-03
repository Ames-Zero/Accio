"""
Knowledge Graph Construction Evaluation
Metrics: Completeness, Correctness, Provenance, Graph structure quality
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


@dataclass
class KGMetrics:
    """Knowledge graph evaluation metrics"""
    completeness: float  # Percentage of expected entities/relations present
    correctness: float  # Percentage of correct entities/relations (manual review)
    provenance_coverage: float  # Percentage of entities/relations with source tracking
    graph_quality: Dict[str, float]  # Graph structure metrics (connectivity, density, etc.)


class KnowledgeGraphEvaluator:
    """
    Evaluates knowledge graph construction quality.
    Measures completeness, correctness, provenance, and graph structure.
    """
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        """
        Initialize KG evaluator
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        """
        from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        
        self.neo4j_uri = neo4j_uri or NEO4J_URI
        self.neo4j_user = neo4j_user or NEO4J_USER
        self.neo4j_password = neo4j_password or NEO4J_PASSWORD
        
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        logger.info(f"KG Evaluator initialized with Neo4j at {self.neo4j_uri}")
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()
    
    def evaluate_completeness(
        self,
        company_id: str,
        expected_entities: List[str] = None,
        expected_relations: List[Tuple[str, str, str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate graph completeness
        
        Args:
            company_id: Company identifier
            expected_entities: Optional list of expected entity names
            expected_relations: Optional list of expected (subject, predicate, object) tuples
            
        Returns:
            Dictionary with completeness metrics
        """
        with self.driver.session() as session:
            # Count actual entities
            entity_query = """
            MATCH (e:Entity)
            WHERE e.company_id = $company_id
            RETURN count(DISTINCT e.name) AS entity_count
            """
            entity_result = session.run(entity_query, company_id=company_id)
            actual_entity_count = entity_result.single()["entity_count"]
            
            # Count actual relations
            relation_query = """
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE r.company_id = $company_id
            RETURN count(r) AS relation_count
            """
            relation_result = session.run(relation_query, company_id=company_id)
            actual_relation_count = relation_result.single()["relation_count"]
            
            # Calculate completeness if expected values provided
            entity_completeness = None
            relation_completeness = None
            
            if expected_entities:
                # Check which expected entities are present
                found_entities = set()
                entity_check_query = """
                MATCH (e:Entity)
                WHERE e.company_id = $company_id AND e.name IN $expected
                RETURN e.name AS name
                """
                check_result = session.run(entity_check_query, company_id=company_id, expected=expected_entities)
                for record in check_result:
                    found_entities.add(record["name"])
                
                entity_completeness = len(found_entities) / len(expected_entities) if expected_entities else 0.0
            
            if expected_relations:
                # Check which expected relations are present
                found_relations = set()
                for subject, predicate, obj in expected_relations:
                    relation_check_query = """
                    MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
                    WHERE s.company_id = $company_id
                        AND s.name = $subject
                        AND r.relation = $predicate
                        AND o.name = $object
                    RETURN count(r) AS count
                    """
                    check_result = session.run(
                        relation_check_query,
                        company_id=company_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj
                    )
                    if check_result.single()["count"] > 0:
                        found_relations.add((subject, predicate, obj))
                
                relation_completeness = len(found_relations) / len(expected_relations) if expected_relations else 0.0
        
        return {
            "entity_count": actual_entity_count,
            "relation_count": actual_relation_count,
            "entity_completeness": entity_completeness,
            "relation_completeness": relation_completeness
        }
    
    def evaluate_provenance(
        self,
        company_id: str
    ) -> Dict[str, float]:
        """
        Evaluate provenance coverage (source tracking)
        
        Args:
            company_id: Company identifier
            
        Returns:
            Dictionary with provenance metrics
        """
        with self.driver.session() as session:
            # Check entities with source chunks
            entity_provenance_query = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(ch:Chunk)
            WHERE e.company_id = $company_id
            WITH e, count(DISTINCT ch) AS chunk_count
            RETURN count(DISTINCT e) AS entities_with_provenance,
                   count(e) AS total_entity_mentions
            """
            entity_result = session.run(entity_provenance_query, company_id=company_id)
            entity_record = entity_result.single()
            entities_with_provenance = entity_record["entities_with_provenance"]
            total_entity_mentions = entity_record["total_entity_mentions"]
            
            # Check relations with evidence
            relation_provenance_query = """
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE r.company_id = $company_id
            RETURN count(r) AS total_relations,
                   sum(CASE WHEN r.evidence IS NOT NULL AND r.evidence <> '' THEN 1 ELSE 0 END) AS relations_with_evidence,
                   sum(CASE WHEN r.doc_id IS NOT NULL THEN 1 ELSE 0 END) AS relations_with_doc_id
            """
            relation_result = session.run(relation_provenance_query, company_id=company_id)
            relation_record = relation_result.single()
            total_relations = relation_record["total_relations"]
            relations_with_evidence = relation_record["relations_with_evidence"]
            relations_with_doc_id = relation_record["relations_with_doc_id"]
            
            # Calculate total entities
            total_entities_query = """
            MATCH (e:Entity)
            WHERE e.company_id = $company_id
            RETURN count(DISTINCT e.name) AS total_entities
            """
            total_entities = session.run(total_entities_query, company_id=company_id).single()["total_entities"]
        
        entity_provenance_coverage = entities_with_provenance / total_entities if total_entities > 0 else 0.0
        relation_evidence_coverage = relations_with_evidence / total_relations if total_relations > 0 else 0.0
        relation_doc_coverage = relations_with_doc_id / total_relations if total_relations > 0 else 0.0
        
        return {
            "entity_provenance_coverage": entity_provenance_coverage,
            "relation_evidence_coverage": relation_evidence_coverage,
            "relation_doc_coverage": relation_doc_coverage,
            "entities_with_provenance": entities_with_provenance,
            "total_entities": total_entities,
            "relations_with_evidence": relations_with_evidence,
            "total_relations": total_relations
        }
    
    def evaluate_graph_structure(
        self,
        company_id: str
    ) -> Dict[str, float]:
        """
        Evaluate graph structure quality (connectivity, density, etc.)
        
        Args:
            company_id: Company identifier
            
        Returns:
            Dictionary with graph structure metrics
        """
        with self.driver.session() as session:
            # Get basic graph statistics
            stats_query = """
            MATCH (e:Entity)
            WHERE e.company_id = $company_id
            WITH count(DISTINCT e) AS entity_count
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE r.company_id = $company_id
            WITH entity_count, count(r) AS relation_count
            MATCH (e:Entity)-[:RELATES_TO*1..2]-(connected:Entity)
            WHERE e.company_id = $company_id
            WITH entity_count, relation_count, count(DISTINCT e) AS connected_entities
            RETURN entity_count, relation_count, connected_entities,
                   CASE WHEN entity_count > 0 THEN toFloat(relation_count) / entity_count ELSE 0.0 END AS avg_degree,
                   CASE WHEN entity_count > 1 THEN toFloat(relation_count) / (entity_count * (entity_count - 1)) ELSE 0.0 END AS graph_density
            """
            stats_result = session.run(stats_query, company_id=company_id)
            stats = stats_result.single()
            
            # Calculate component sizes (connected subgraphs)
            component_query = """
            MATCH (e:Entity)
            WHERE e.company_id = $company_id
            WITH e
            CALL {
                WITH e
                MATCH path = (e)-[:RELATES_TO*]-(connected:Entity)
                WHERE connected.company_id = $company_id
                RETURN count(DISTINCT connected) AS component_size
            }
            RETURN avg(component_size) AS avg_component_size,
                   max(component_size) AS max_component_size,
                   min(component_size) AS min_component_size
            """
            component_result = session.run(component_query, company_id=company_id)
            component_stats = component_result.single()
        
        return {
            "entity_count": stats["entity_count"],
            "relation_count": stats["relation_count"],
            "connected_entities": stats["connected_entities"],
            "connectivity_ratio": stats["connected_entities"] / stats["entity_count"] if stats["entity_count"] > 0 else 0.0,
            "average_degree": stats["avg_degree"],
            "graph_density": stats["graph_density"],
            "avg_component_size": component_stats["avg_component_size"] or 0.0,
            "max_component_size": component_stats["max_component_size"] or 0.0,
            "min_component_size": component_stats["min_component_size"] or 0.0
        }
    
    def manual_review_correctness(
        self,
        company_id: str,
        review_file: Path = None
    ) -> Dict[str, float]:
        """
        Load manual review results and calculate correctness
        
        Args:
            company_id: Company identifier
            review_file: Path to manual review JSON file
            
        Returns:
            Dictionary with correctness metrics
        """
        if review_file is None:
            review_file = Path(f"evaluations/ground_truth/manual_reviews/{company_id}.json")
        
        if not review_file.exists():
            logger.warning(f"Manual review file not found: {review_file}")
            return {
                "correctness": None,
                "reviewed_entities": 0,
                "correct_entities": 0,
                "reviewed_relations": 0,
                "correct_relations": 0
            }
        
        with open(review_file, 'r', encoding='utf-8') as f:
            review_data = json.load(f)
        
        # Count correct vs incorrect
        correct_entities = sum(1 for e in review_data.get("entities", []) if e.get("correct", False))
        reviewed_entities = len(review_data.get("entities", []))
        
        correct_relations = sum(1 for r in review_data.get("relations", []) if r.get("correct", False))
        reviewed_relations = len(review_data.get("relations", []))
        
        entity_correctness = correct_entities / reviewed_entities if reviewed_entities > 0 else 0.0
        relation_correctness = correct_relations / reviewed_relations if reviewed_relations > 0 else 0.0
        overall_correctness = (entity_correctness + relation_correctness) / 2 if (reviewed_entities > 0 or reviewed_relations > 0) else 0.0
        
        return {
            "correctness": overall_correctness,
            "entity_correctness": entity_correctness,
            "relation_correctness": relation_correctness,
            "reviewed_entities": reviewed_entities,
            "correct_entities": correct_entities,
            "reviewed_relations": reviewed_relations,
            "correct_relations": correct_relations
        }
    
    def evaluate_company(
        self,
        company_id: str,
        expected_entities: List[str] = None,
        expected_relations: List[Tuple[str, str, str]] = None,
        review_file: Path = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation for a company's knowledge graph
        
        Args:
            company_id: Company identifier
            expected_entities: Optional list of expected entities
            expected_relations: Optional list of expected relations
            review_file: Optional path to manual review file
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Evaluating knowledge graph for {company_id}")
        
        completeness = self.evaluate_completeness(company_id, expected_entities, expected_relations)
        provenance = self.evaluate_provenance(company_id)
        graph_structure = self.evaluate_graph_structure(company_id)
        correctness = self.manual_review_correctness(company_id, review_file)
        
        return {
            "company_id": company_id,
            "completeness": completeness,
            "provenance": provenance,
            "graph_structure": graph_structure,
            "correctness": correctness
        }
    
    def compare_schema_variants(
        self,
        company_id: str,
        schema_variants: List[str]
    ) -> Dict[str, Any]:
        """
        Compare different schema variants to optimize explainability and retrieval
        
        Args:
            company_id: Company identifier
            schema_variants: List of schema variant names to compare
            
        Returns:
            Dictionary comparing schema variants
        """
        results = {}
        
        for variant in schema_variants:
            # Query graph with different schema assumptions
            # This is a placeholder - actual implementation would query different node/relationship types
            logger.info(f"Evaluating schema variant: {variant}")
            
            with self.driver.session() as session:
                # Example: evaluate retrieval accuracy for this schema
                # This would require test queries and expected results
                variant_stats = {
                    "entity_count": 0,
                    "relation_count": 0,
                    "avg_path_length": 0.0
                }
                
                results[variant] = variant_stats
        
        return {
            "company_id": company_id,
            "variants": results
        }

