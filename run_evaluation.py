"""
Comprehensive System Evaluation
Runs all evaluation metrics and generates a detailed report
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """Test Neo4j connection and get database stats"""
    try:
        from neo4j import GraphDatabase
        from backend.app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            # Get node counts
            node_result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS type, count(n) AS count
                ORDER BY count DESC
            """)
            
            nodes = {record["type"]: record["count"] for record in node_result}
            
            # Get relationship count
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            rel_count = rel_result.single()["count"]
            
            # Get chunks with embeddings
            emb_result = session.run("""
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN count(c) AS count
            """)
            emb_count = emb_result.single()["count"]
            
            # Get entities
            entity_result = session.run("""
                MATCH (e:Entity)
                RETURN count(DISTINCT e.name) AS count
            """)
            entity_count = entity_result.single()["count"]
            
            # Get relations
            relation_result = session.run("""
                MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity)
                RETURN count(r) AS count
            """)
            relation_count = relation_result.single()["count"]
            
        driver.close()
        
        return {
            "connected": True,
            "nodes": nodes,
            "total_nodes": sum(nodes.values()),
            "total_relationships": rel_count,
            "chunks_with_embeddings": emb_count,
            "unique_entities": entity_count,
            "entity_relations": relation_count
        }
    
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return {"connected": False, "error": str(e)}


def test_rag_system():
    """Test RAG system with sample queries"""
    try:
        from backend.app.rag_engine import RAGEngine
        
        rag_engine = RAGEngine()
        
        test_queries = [
            "What happened to NSF's Total Liabilities in FY 2011?",
            "What is the NNI's approach to addressing ethical questions?",
            "Where must accessible parking spaces be located?"
        ]
        
        results = []
        
        for query in test_queries:
            logger.info(f"Testing query: {query[:50]}...")
            
            try:
                import time
                start_time = time.time()
                
                answer, citations = rag_engine.query(query, "company_1")
                
                response_time = time.time() - start_time
                
                results.append({
                    "query": query,
                    "answer": answer[:200] + "..." if len(answer) > 200 else answer,
                    "citation_count": len(citations),
                    "response_time": response_time,
                    "success": True
                })
            
            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        rag_engine.close()
        
        return {
            "tested": True,
            "queries": results,
            "avg_response_time": sum(r.get("response_time", 0) for r in results if r["success"]) / len([r for r in results if r["success"]]) if any(r["success"] for r in results) else 0,
            "success_rate": sum(1 for r in results if r["success"]) / len(results)
        }
    
    except Exception as e:
        logger.error(f"RAG system test failed: {e}")
        return {"tested": False, "error": str(e)}


def evaluate_knowledge_graph():
    """Evaluate knowledge graph quality"""
    try:
        from evaluations.kg_evaluation import KnowledgeGraphEvaluator
        
        evaluator = KnowledgeGraphEvaluator()
        
        # Evaluate completeness and structure
        completeness = evaluator.evaluate_completeness("company_1")
        provenance = evaluator.evaluate_provenance("company_1")
        graph_structure = evaluator.evaluate_graph_structure("company_1")
        
        evaluator.close()
        
        return {
            "evaluated": True,
            "completeness": completeness,
            "provenance": provenance,
            "graph_structure": graph_structure
        }
    
    except Exception as e:
        logger.error(f"KG evaluation failed: {e}")
        return {"evaluated": False, "error": str(e)}


def analyze_extraction_quality():
    """Analyze extraction quality from processed files"""
    try:
        processed_dir = Path("data/processed/company_1")
        json_files = list(processed_dir.glob("*.json"))
        
        total_pages = 0
        total_blocks = 0
        total_chunks_estimated = 0
        
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            total_pages += len(data.get("pages", []))
            
            for page in data.get("pages", []):
                blocks = page.get("layout_blocks", [])
                total_blocks += len(blocks)
                
                # Estimate chunks (512 words per chunk)
                for block in blocks:
                    text = block.get("text", "")
                    word_count = len(text.split())
                    total_chunks_estimated += max(1, word_count // 512)
        
        return {
            "analyzed": True,
            "total_documents": len(json_files),
            "total_pages": total_pages,
            "total_text_blocks": total_blocks,
            "estimated_chunks": total_chunks_estimated,
            "avg_blocks_per_page": total_blocks / total_pages if total_pages > 0 else 0
        }
    
    except Exception as e:
        logger.error(f"Extraction analysis failed: {e}")
        return {"analyzed": False, "error": str(e)}


def generate_report(results):
    """Generate comprehensive evaluation report"""
    
    report_lines = [
        "="*80,
        "COMPREHENSIVE SYSTEM EVALUATION REPORT",
        "="*80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "="*80,
        "1. NEO4J DATABASE STATUS",
        "="*80,
    ]
    
    neo4j = results.get("neo4j", {})
    if neo4j.get("connected"):
        report_lines.extend([
            f"[OK] Connected to Neo4j",
            f"",
            f"Node Statistics:",
            f"  - Total Nodes: {neo4j.get('total_nodes', 0):,}",
        ])
        
        for node_type, count in neo4j.get("nodes", {}).items():
            report_lines.append(f"    • {node_type}: {count:,}")
        
        report_lines.extend([
            f"",
            f"Relationship Statistics:",
            f"  - Total Relationships: {neo4j.get('total_relationships', 0):,}",
            f"  - Entity Relations: {neo4j.get('entity_relations', 0):,}",
            f"",
            f"Embedding Statistics:",
            f"  - Chunks with Embeddings: {neo4j.get('chunks_with_embeddings', 0):,}",
            f"  - Unique Entities: {neo4j.get('unique_entities', 0):,}",
        ])
    else:
        report_lines.append(f"[FAIL] Neo4j connection failed: {neo4j.get('error', 'Unknown error')}")
    
    report_lines.extend([
        "",
        "="*80,
        "2. EXTRACTION QUALITY ANALYSIS",
        "="*80,
    ])
    
    extraction = results.get("extraction", {})
    if extraction.get("analyzed"):
        report_lines.extend([
            f"[OK] Extraction analysis completed",
            f"",
            f"Document Statistics:",
            f"  - Total Documents: {extraction.get('total_documents', 0)}",
            f"  - Total Pages: {extraction.get('total_pages', 0)}",
            f"  - Total Text Blocks: {extraction.get('total_text_blocks', 0):,}",
            f"  - Estimated Chunks: {extraction.get('estimated_chunks', 0):,}",
            f"  - Avg Blocks per Page: {extraction.get('avg_blocks_per_page', 0):.2f}",
        ])
    else:
        report_lines.append(f"[FAIL] Extraction analysis failed: {extraction.get('error', 'Unknown error')}")
    
    report_lines.extend([
        "",
        "="*80,
        "3. KNOWLEDGE GRAPH EVALUATION",
        "="*80,
    ])
    
    kg = results.get("knowledge_graph", {})
    if kg.get("evaluated"):
        completeness = kg.get("completeness", {})
        provenance = kg.get("provenance", {})
        graph_structure = kg.get("graph_structure", {})
        
        report_lines.extend([
            f"[OK] Knowledge graph evaluation completed",
            f"",
            f"Completeness:",
            f"  - Entity Count: {completeness.get('entity_count', 0):,}",
            f"  - Relation Count: {completeness.get('relation_count', 0):,}",
            f"",
            f"Provenance Coverage:",
            f"  - Entities with Source: {provenance.get('entity_provenance_coverage', 0):.2%}",
            f"  - Relations with Evidence: {provenance.get('relation_evidence_coverage', 0):.2%}",
            f"  - Relations with Doc ID: {provenance.get('relation_doc_coverage', 0):.2%}",
            f"",
            f"Graph Structure Quality:",
            f"  - Connected Entities: {graph_structure.get('connected_entities', 0):,}",
            f"  - Connectivity Ratio: {graph_structure.get('connectivity_ratio', 0):.2%}",
            f"  - Average Degree: {graph_structure.get('average_degree', 0):.2f}",
            f"  - Graph Density: {graph_structure.get('graph_density', 0):.4f}",
        ])
    else:
        report_lines.append(f"[FAIL] KG evaluation failed: {kg.get('error', 'Unknown error')}")
    
    report_lines.extend([
        "",
        "="*80,
        "4. RAG SYSTEM PERFORMANCE",
        "="*80,
    ])
    
    rag = results.get("rag_system", {})
    if rag.get("tested"):
        report_lines.extend([
            f"[OK] RAG system test completed",
            f"",
            f"Performance Metrics:",
            f"  - Success Rate: {rag.get('success_rate', 0):.2%}",
            f"  - Avg Response Time: {rag.get('avg_response_time', 0):.2f}s",
            f"",
            f"Query Results:",
        ])
        
        for i, query_result in enumerate(rag.get("queries", []), 1):
            if query_result.get("success"):
                report_lines.extend([
                    f"",
                    f"  Query {i}: {query_result['query'][:60]}...",
                    f"    - Citations: {query_result['citation_count']}",
                    f"    - Response Time: {query_result['response_time']:.2f}s",
                    f"    - Answer: {query_result['answer'][:100]}...",
                ])
            else:
                report_lines.extend([
                    f"",
                    f"  Query {i}: {query_result['query'][:60]}...",
                    f"    - Status: FAILED",
                    f"    - Error: {query_result.get('error', 'Unknown')}",
                ])
    else:
        report_lines.append(f"[FAIL] RAG system test failed: {rag.get('error', 'Unknown error')}")
    
    report_lines.extend([
        "",
        "="*80,
        "5. OVERALL SYSTEM HEALTH",
        "="*80,
    ])
    
    # Calculate overall health score
    health_checks = [
        neo4j.get("connected", False),
        extraction.get("analyzed", False),
        kg.get("evaluated", False),
        rag.get("tested", False) and rag.get("success_rate", 0) > 0.5
    ]
    
    health_score = sum(health_checks) / len(health_checks)
    
    report_lines.extend([
        f"Health Score: {health_score:.0%}",
        f"",
        f"Component Status:",
        f"  {'[OK]' if neo4j.get('connected') else '[FAIL]'} Neo4j Database",
        f"  {'[OK]' if extraction.get('analyzed') else '[FAIL]'} Data Extraction",
        f"  {'[OK]' if kg.get('evaluated') else '[FAIL]'} Knowledge Graph",
        f"  {'[OK]' if rag.get('tested') and rag.get('success_rate', 0) > 0.5 else '[FAIL]'} RAG System",
        "",
    ])
    
    if health_score == 1.0:
        report_lines.append("[SUCCESS] All systems operational!")
    elif health_score >= 0.75:
        report_lines.append("[WARNING] System mostly operational with minor issues")
    elif health_score >= 0.5:
        report_lines.append("[WARNING] System partially operational - attention needed")
    else:
        report_lines.append("[ERROR] System has critical issues - immediate attention required")
    
    report_lines.extend([
        "",
        "="*80,
        "END OF REPORT",
        "="*80,
    ])
    
    return "\n".join(report_lines)


def main():
    """Run comprehensive evaluation"""
    logger.info("="*80)
    logger.info("Starting Comprehensive System Evaluation")
    logger.info("="*80)
    
    results = {}
    
    # 1. Test Neo4j connection
    logger.info("\n1. Testing Neo4j connection...")
    results["neo4j"] = test_neo4j_connection()
    
    # 2. Analyze extraction quality
    logger.info("\n2. Analyzing extraction quality...")
    results["extraction"] = analyze_extraction_quality()
    
    # 3. Evaluate knowledge graph
    logger.info("\n3. Evaluating knowledge graph...")
    results["knowledge_graph"] = evaluate_knowledge_graph()
    
    # 4. Test RAG system
    logger.info("\n4. Testing RAG system...")
    results["rag_system"] = test_rag_system()
    
    # Generate report
    logger.info("\n5. Generating report...")
    report = generate_report(results)
    
    # Save results
    output_dir = Path("evaluations/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_file = output_dir / f"evaluation_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save text report
    report_file = output_dir / f"evaluation_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print report
    print("\n")
    print(report.encode('utf-8', errors='replace').decode('utf-8'))
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - JSON: {json_file}")
    logger.info(f"  - Report: {report_file}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

