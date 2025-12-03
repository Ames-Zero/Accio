"""
Entity and Relation Extraction Evaluation
Metrics: Precision, Recall, F1-score
"""

import logging
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EntityMetrics:
    """Entity extraction evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class RelationMetrics:
    """Relation extraction evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int


class EntityRelationEvaluator:
    """
    Evaluates entity and relation extraction using Precision, Recall, and F1-score.
    Compares extracted entities/relations to annotated ground truth.
    """
    
    def __init__(self, ground_truth_dir: Path = None):
        """
        Initialize entity/relation evaluator
        
        Args:
            ground_truth_dir: Directory containing ground truth annotations
        """
        self.ground_truth_dir = ground_truth_dir or Path("evaluations/ground_truth/entities")
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Entity/Relation Evaluator initialized with ground truth dir: {self.ground_truth_dir}")
    
    def normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name for comparison
        
        Args:
            name: Entity name
            
        Returns:
            Normalized entity name
        """
        return " ".join(name.lower().strip().split())
    
    def normalize_relation(self, relation: str) -> str:
        """
        Normalize relation/predicate for comparison
        
        Args:
            relation: Relation/predicate name
            
        Returns:
            Normalized relation
        """
        return " ".join(relation.lower().strip().split())
    
    def evaluate_entities(
        self,
        predicted_entities: List[Dict[str, Any]],
        ground_truth_entities: List[Dict[str, Any]]
    ) -> EntityMetrics:
        """
        Evaluate entity extraction
        
        Args:
            predicted_entities: List of extracted entities with 'name' and optional 'type'
            ground_truth_entities: List of ground truth entities with 'name' and optional 'type'
            
        Returns:
            EntityMetrics object
        """
        # Normalize and create sets for comparison
        pred_set = set()
        gt_set = set()
        
        for entity in predicted_entities:
            name = self.normalize_entity_name(entity.get("name", ""))
            entity_type = entity.get("type", "").lower() if entity.get("type") else ""
            key = (name, entity_type) if entity_type else name
            pred_set.add(key)
        
        for entity in ground_truth_entities:
            name = self.normalize_entity_name(entity.get("name", ""))
            entity_type = entity.get("type", "").lower() if entity.get("type") else ""
            key = (name, entity_type) if entity_type else name
            gt_set.add(key)
        
        # Calculate metrics
        true_positives = len(pred_set & gt_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EntityMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
    
    def evaluate_relations(
        self,
        predicted_relations: List[Dict[str, Any]],
        ground_truth_relations: List[Dict[str, Any]]
    ) -> RelationMetrics:
        """
        Evaluate relation extraction
        
        Args:
            predicted_relations: List of extracted relations with 'subject', 'predicate', 'object'
            ground_truth_relations: List of ground truth relations with 'subject', 'predicate', 'object'
            
        Returns:
            RelationMetrics object
        """
        # Normalize and create sets for comparison
        pred_set = set()
        gt_set = set()
        
        for rel in predicted_relations:
            subject = self.normalize_entity_name(rel.get("subject", ""))
            predicate = self.normalize_relation(rel.get("predicate", ""))
            obj = self.normalize_entity_name(rel.get("object", ""))
            
            if subject and predicate and obj:
                pred_set.add((subject, predicate, obj))
        
        for rel in ground_truth_relations:
            subject = self.normalize_entity_name(rel.get("subject", ""))
            predicate = self.normalize_relation(rel.get("predicate", ""))
            obj = self.normalize_entity_name(rel.get("object", ""))
            
            if subject and predicate and obj:
                gt_set.add((subject, predicate, obj))
        
        # Calculate metrics
        true_positives = len(pred_set & gt_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return RelationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )
    
    def load_ground_truth(self, doc_id: str) -> Dict[str, Any]:
        """
        Load ground truth annotations for a document
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with 'entities' and 'relations' lists
        """
        gt_file = self.ground_truth_dir / f"{doc_id}.json"
        
        if not gt_file.exists():
            logger.warning(f"Ground truth file not found: {gt_file}")
            return {"entities": [], "relations": []}
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_entities_from_kg(self, kg_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from knowledge graph data
        
        Args:
            kg_data: Knowledge graph data (from Neo4j or extraction output)
            
        Returns:
            List of entities
        """
        entities = []
        
        # Extract from triplets if available
        for triplet in kg_data.get("triplets", []):
            subject = triplet.get("subject", "")
            obj = triplet.get("object", "")
            
            if subject:
                entities.append({"name": subject, "type": triplet.get("subject_type", "")})
            if obj:
                entities.append({"name": obj, "type": triplet.get("object_type", "")})
        
        # Deduplicate
        seen = set()
        unique_entities = []
        for entity in entities:
            key = self.normalize_entity_name(entity["name"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_relations_from_kg(self, kg_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relations from knowledge graph data
        
        Args:
            kg_data: Knowledge graph data (from Neo4j or extraction output)
            
        Returns:
            List of relations
        """
        relations = []
        
        # Extract from triplets
        for triplet in kg_data.get("triplets", []):
            subject = triplet.get("subject", "")
            predicate = triplet.get("predicate", "")
            obj = triplet.get("object", "")
            
            if subject and predicate and obj:
                relations.append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj
                })
        
        return relations
    
    def evaluate_document(
        self,
        doc_id: str,
        extracted_kg_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate entity and relation extraction for a document
        
        Args:
            doc_id: Document identifier
            extracted_kg_data: Extracted knowledge graph data
            
        Returns:
            Dictionary with evaluation results
        """
        # Load ground truth
        ground_truth = self.load_ground_truth(doc_id)
        
        # Extract predicted entities and relations
        predicted_entities = self.extract_entities_from_kg(extracted_kg_data)
        predicted_relations = self.extract_relations_from_kg(extracted_kg_data)
        
        # Evaluate
        entity_metrics = self.evaluate_entities(
            predicted_entities,
            ground_truth.get("entities", [])
        )
        
        relation_metrics = self.evaluate_relations(
            predicted_relations,
            ground_truth.get("relations", [])
        )
        
        return {
            "doc_id": doc_id,
            "entities": {
                "precision": entity_metrics.precision,
                "recall": entity_metrics.recall,
                "f1_score": entity_metrics.f1_score,
                "true_positives": entity_metrics.true_positives,
                "false_positives": entity_metrics.false_positives,
                "false_negatives": entity_metrics.false_negatives,
                "predicted_count": len(predicted_entities),
                "ground_truth_count": len(ground_truth.get("entities", []))
            },
            "relations": {
                "precision": relation_metrics.precision,
                "recall": relation_metrics.recall,
                "f1_score": relation_metrics.f1_score,
                "true_positives": relation_metrics.true_positives,
                "false_positives": relation_metrics.false_positives,
                "false_negatives": relation_metrics.false_negatives,
                "predicted_count": len(predicted_relations),
                "ground_truth_count": len(ground_truth.get("relations", []))
            }
        }
    
    def evaluate_from_neo4j(
        self,
        doc_id: str,
        neo4j_driver
    ) -> Dict[str, Any]:
        """
        Evaluate entities and relations by querying Neo4j
        
        Args:
            doc_id: Document identifier
            neo4j_driver: Neo4j driver instance
            
        Returns:
            Dictionary with evaluation results
        """
        # Query Neo4j for entities and relations
        with neo4j_driver.session() as session:
            # Get entities
            entity_query = """
            MATCH (e:Entity)-[:MENTIONED_IN]->(ch:Chunk)-[:FROM_PAGE]->(p:Page)-[:IN_DOCUMENT]->(d:Document)
            WHERE d.doc_id = $doc_id
            RETURN DISTINCT e.name AS name, e.type AS type
            """
            entity_result = session.run(entity_query, doc_id=doc_id)
            predicted_entities = [{"name": record["name"], "type": record.get("type", "")} for record in entity_result]
            
            # Get relations
            relation_query = """
            MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
            WHERE r.doc_id = $doc_id
            RETURN s.name AS subject, r.relation AS predicate, o.name AS object
            """
            relation_result = session.run(relation_query, doc_id=doc_id)
            predicted_relations = [
                {
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"]
                }
                for record in relation_result
            ]
        
        # Load ground truth
        ground_truth = self.load_ground_truth(doc_id)
        
        # Evaluate
        entity_metrics = self.evaluate_entities(
            predicted_entities,
            ground_truth.get("entities", [])
        )
        
        relation_metrics = self.evaluate_relations(
            predicted_relations,
            ground_truth.get("relations", [])
        )
        
        return {
            "doc_id": doc_id,
            "entities": {
                "precision": entity_metrics.precision,
                "recall": entity_metrics.recall,
                "f1_score": entity_metrics.f1_score,
                "true_positives": entity_metrics.true_positives,
                "false_positives": entity_metrics.false_positives,
                "false_negatives": entity_metrics.false_negatives,
                "predicted_count": len(predicted_entities),
                "ground_truth_count": len(ground_truth.get("entities", []))
            },
            "relations": {
                "precision": relation_metrics.precision,
                "recall": relation_metrics.recall,
                "f1_score": relation_metrics.f1_score,
                "true_positives": relation_metrics.true_positives,
                "false_positives": relation_metrics.false_positives,
                "false_negatives": relation_metrics.false_negatives,
                "predicted_count": len(predicted_relations),
                "ground_truth_count": len(ground_truth.get("relations", []))
            }
        }
    
    def cross_validate(
        self,
        doc_ids: List[str],
        neo4j_driver,
        k_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation on entity/relation extraction
        
        Args:
            doc_ids: List of document identifiers
            neo4j_driver: Neo4j driver instance
            k_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary with cross-validation results
        """
        import numpy as np
        
        # Split documents into folds
        fold_size = len(doc_ids) // k_folds
        folds = []
        for i in range(k_folds):
            start = i * fold_size
            end = start + fold_size if i < k_folds - 1 else len(doc_ids)
            folds.append(doc_ids[start:end])
        
        fold_results = []
        
        for fold_idx, test_fold in enumerate(folds):
            logger.info(f"Processing fold {fold_idx + 1}/{k_folds}")
            
            # Evaluate on test fold
            fold_metrics = {
                "fold": fold_idx + 1,
                "test_documents": test_fold,
                "entity_precision": [],
                "entity_recall": [],
                "entity_f1": [],
                "relation_precision": [],
                "relation_recall": [],
                "relation_f1": []
            }
            
            for doc_id in test_fold:
                try:
                    results = self.evaluate_from_neo4j(doc_id, neo4j_driver)
                    fold_metrics["entity_precision"].append(results["entities"]["precision"])
                    fold_metrics["entity_recall"].append(results["entities"]["recall"])
                    fold_metrics["entity_f1"].append(results["entities"]["f1_score"])
                    fold_metrics["relation_precision"].append(results["relations"]["precision"])
                    fold_metrics["relation_recall"].append(results["relations"]["recall"])
                    fold_metrics["relation_f1"].append(results["relations"]["f1_score"])
                except Exception as e:
                    logger.error(f"Error evaluating {doc_id}: {e}")
            
            # Calculate fold averages
            for metric in ["entity_precision", "entity_recall", "entity_f1",
                          "relation_precision", "relation_recall", "relation_f1"]:
                if fold_metrics[metric]:
                    fold_metrics[f"avg_{metric}"] = np.mean(fold_metrics[metric])
                    fold_metrics[f"std_{metric}"] = np.std(fold_metrics[metric])
            
            fold_results.append(fold_metrics)
        
        # Calculate overall cross-validation metrics
        cv_results = {
            "k_folds": k_folds,
            "folds": fold_results,
            "overall": {
                "entity_precision": np.mean([f.get("avg_entity_precision", 0) for f in fold_results]),
                "entity_recall": np.mean([f.get("avg_entity_recall", 0) for f in fold_results]),
                "entity_f1": np.mean([f.get("avg_entity_f1", 0) for f in fold_results]),
                "relation_precision": np.mean([f.get("avg_relation_precision", 0) for f in fold_results]),
                "relation_recall": np.mean([f.get("avg_relation_recall", 0) for f in fold_results]),
                "relation_f1": np.mean([f.get("avg_relation_f1", 0) for f in fold_results])
            }
        }
        
        return cv_results

