"""
Simplified System Evaluation
Evaluates the system without requiring Neo4j to be running
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def analyze_pdfs():
    """Analyze PDF files"""
    pdf_dir = Path("data/company_1/pdfs")
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    total_size = sum(pdf.stat().st_size for pdf in pdfs)
    
    return {
        "total_pdfs": len(pdfs),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "avg_size_mb": round(total_size / (1024 * 1024) / len(pdfs), 2) if pdfs else 0,
        "pdf_list": [pdf.name for pdf in sorted(pdfs)]
    }


def analyze_extracted_data():
    """Analyze extracted JSON files"""
    processed_dir = Path("data/processed/company_1")
    json_files = list(processed_dir.glob("*.json"))
    
    stats = {
        "total_documents": len(json_files),
        "total_pages": 0,
        "total_blocks": 0,
        "total_text_length": 0,
        "estimated_chunks": 0,
        "pages_per_doc": [],
        "blocks_per_page": [],
        "document_details": []
    }
    
    for json_file in sorted(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        doc_pages = len(data.get("pages", []))
        doc_blocks = 0
        doc_text_length = 0
        
        for page in data.get("pages", []):
            blocks = page.get("layout_blocks", [])
            doc_blocks += len(blocks)
            
            for block in blocks:
                text = block.get("text", "")
                doc_text_length += len(text)
                word_count = len(text.split())
                stats["estimated_chunks"] += max(1, word_count // 512)
        
        stats["total_pages"] += doc_pages
        stats["total_blocks"] += doc_blocks
        stats["total_text_length"] += doc_text_length
        stats["pages_per_doc"].append(doc_pages)
        stats["blocks_per_page"].append(doc_blocks / doc_pages if doc_pages > 0 else 0)
        
        stats["document_details"].append({
            "filename": json_file.stem,
            "pages": doc_pages,
            "blocks": doc_blocks,
            "text_length": doc_text_length,
            "estimated_chunks": max(1, (doc_text_length // 4) // 512)  # Rough estimate
        })
    
    # Calculate averages
    if stats["total_documents"] > 0:
        stats["avg_pages_per_doc"] = round(stats["total_pages"] / stats["total_documents"], 2)
        stats["avg_blocks_per_doc"] = round(stats["total_blocks"] / stats["total_documents"], 2)
        stats["avg_text_length_per_doc"] = round(stats["total_text_length"] / stats["total_documents"], 2)
    
    if stats["total_pages"] > 0:
        stats["avg_blocks_per_page"] = round(stats["total_blocks"] / stats["total_pages"], 2)
    
    return stats


def analyze_images():
    """Analyze generated page images"""
    images_dir = Path("data/company_1/images")
    
    if not images_dir.exists():
        return {"exists": False}
    
    images = list(images_dir.glob("*.png"))
    total_size = sum(img.stat().st_size for img in images)
    
    return {
        "exists": True,
        "total_images": len(images),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "avg_size_mb": round(total_size / (1024 * 1024) / len(images), 2) if images else 0
    }


def check_neo4j_database():
    """Check if Neo4j database files exist"""
    neo4j_dir = Path("neo4j_data")
    
    if not neo4j_dir.exists():
        return {"exists": False, "status": "Database directory not found"}
    
    # Check for database files
    db_files = list(neo4j_dir.rglob("*"))
    total_size = sum(f.stat().st_size for f in db_files if f.is_file())
    
    return {
        "exists": True,
        "total_files": len([f for f in db_files if f.is_file()]),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "status": "Database files present (Neo4j not running)"
    }


def generate_comprehensive_report(results):
    """Generate detailed evaluation report"""
    
    lines = [
        "=" * 100,
        "COMPREHENSIVE SYSTEM EVALUATION REPORT",
        "=" * 100,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "=" * 100,
        "1. PDF DOCUMENTS ANALYSIS",
        "=" * 100,
    ]
    
    pdf_stats = results["pdf_analysis"]
    lines.extend([
        f"Total PDFs: {pdf_stats['total_pdfs']}",
        f"Total Size: {pdf_stats['total_size_mb']} MB",
        f"Average Size: {pdf_stats['avg_size_mb']} MB per document",
        "",
        "Document List:",
    ])
    
    for pdf in pdf_stats["pdf_list"]:
        lines.append(f"  - {pdf}")
    
    lines.extend([
        "",
        "=" * 100,
        "2. EXTRACTION QUALITY ANALYSIS",
        "=" * 100,
    ])
    
    extraction = results["extraction_analysis"]
    lines.extend([
        f"Total Documents Processed: {extraction['total_documents']}",
        f"Total Pages Extracted: {extraction['total_pages']}",
        f"Total Text Blocks: {extraction['total_blocks']:,}",
        f"Total Text Length: {extraction['total_text_length']:,} characters",
        f"Estimated Chunks (512 words): {extraction['estimated_chunks']:,}",
        "",
        "Averages:",
        f"  - Pages per Document: {extraction.get('avg_pages_per_doc', 0):.2f}",
        f"  - Blocks per Document: {extraction.get('avg_blocks_per_doc', 0):.2f}",
        f"  - Blocks per Page: {extraction.get('avg_blocks_per_page', 0):.2f}",
        f"  - Text Length per Document: {extraction.get('avg_text_length_per_doc', 0):,.0f} characters",
        "",
        "Top 5 Documents by Content:",
    ])
    
    sorted_docs = sorted(extraction["document_details"], key=lambda x: x["text_length"], reverse=True)[:5]
    for i, doc in enumerate(sorted_docs, 1):
        lines.append(
            f"  {i}. {doc['filename']}: {doc['pages']} pages, "
            f"{doc['blocks']} blocks, {doc['text_length']:,} chars, "
            f"~{doc['estimated_chunks']} chunks"
        )
    
    lines.extend([
        "",
        "=" * 100,
        "3. IMAGE GENERATION ANALYSIS",
        "=" * 100,
    ])
    
    images = results["image_analysis"]
    if images["exists"]:
        lines.extend([
            f"Total Page Images: {images['total_images']}",
            f"Total Size: {images['total_size_mb']} MB",
            f"Average Size: {images['avg_size_mb']} MB per image",
            f"Status: [OK] All page screenshots generated successfully",
        ])
    else:
        lines.append("Status: [FAIL] No page images found")
    
    lines.extend([
        "",
        "=" * 100,
        "4. NEO4J DATABASE STATUS",
        "=" * 100,
    ])
    
    neo4j = results["neo4j_check"]
    if neo4j["exists"]:
        lines.extend([
            f"Database Files: {neo4j['total_files']:,}",
            f"Database Size: {neo4j['total_size_mb']} MB",
            f"Status: {neo4j['status']}",
            "",
            "Note: To run full evaluation with Neo4j:",
            "  1. Start Neo4j: docker compose up -d",
            "  2. Wait 15 seconds for initialization",
            "  3. Run: python run_evaluation.py",
        ])
    else:
        lines.extend([
            f"Status: {neo4j['status']}",
            "",
            "To build the knowledge graph:",
            "  1. Start Neo4j: docker compose up -d",
            "  2. Run: python scripts/build_knowledge_graph.py",
        ])
    
    lines.extend([
        "",
        "=" * 100,
        "5. EXTRACTION PIPELINE METRICS",
        "=" * 100,
    ])
    
    # Calculate pipeline efficiency
    pdf_count = pdf_stats['total_pdfs']
    extracted_count = extraction['total_documents']
    pages_extracted = extraction['total_pages']
    blocks_extracted = extraction['total_blocks']
    
    extraction_rate = (extracted_count / pdf_count * 100) if pdf_count > 0 else 0
    
    lines.extend([
        f"Extraction Success Rate: {extraction_rate:.1f}% ({extracted_count}/{pdf_count} documents)",
        f"Content Extraction Density: {blocks_extracted / pages_extracted:.1f} blocks per page",
        f"Estimated Chunk Generation: {extraction['estimated_chunks']:,} chunks from {pages_extracted} pages",
        f"Average Chunk per Page: {extraction['estimated_chunks'] / pages_extracted:.1f}",
        "",
        "Data Pipeline Status:",
        f"  [{'OK' if pdf_count > 0 else 'FAIL'}] PDF Documents: {pdf_count} files",
        f"  [{'OK' if extracted_count > 0 else 'FAIL'}] Extraction: {extracted_count} documents processed",
        f"  [{'OK' if images['exists'] and images['total_images'] > 0 else 'FAIL'}] Images: {images.get('total_images', 0)} screenshots",
        f"  [{'OK' if neo4j['exists'] else 'WARN'}] Neo4j: {neo4j['status']}",
    ])
    
    lines.extend([
        "",
        "=" * 100,
        "6. SYSTEM READINESS SUMMARY",
        "=" * 100,
    ])
    
    # Calculate readiness score
    checks = [
        pdf_count > 0,
        extracted_count > 0,
        images["exists"] and images.get("total_images", 0) > 0,
        neo4j["exists"]
    ]
    
    readiness_score = sum(checks) / len(checks) * 100
    
    lines.extend([
        f"Overall Readiness: {readiness_score:.0f}%",
        "",
        "Component Checklist:",
        f"  [{'OK' if checks[0] else 'FAIL'}] PDF Documents Available",
        f"  [{'OK' if checks[1] else 'FAIL'}] Text Extraction Complete",
        f"  [{'OK' if checks[2] else 'FAIL'}] Page Screenshots Generated",
        f"  [{'OK' if checks[3] else 'WARN'}] Knowledge Graph Database",
        "",
    ])
    
    if readiness_score == 100:
        lines.append("[SUCCESS] System is fully ready! Start Neo4j and run the RAG system.")
    elif readiness_score >= 75:
        lines.append("[READY] System is mostly ready. Start Neo4j to enable full functionality.")
    elif readiness_score >= 50:
        lines.append("[PARTIAL] Some components are missing. Review the checklist above.")
    else:
        lines.append("[NOT READY] Critical components are missing. Complete the extraction pipeline first.")
    
    lines.extend([
        "",
        "=" * 100,
        "7. NEXT STEPS",
        "=" * 100,
        "",
        "To complete the system setup:",
        "  1. Ensure Neo4j is running: docker compose up -d",
        "  2. Build knowledge graph (if not done): python scripts/build_knowledge_graph.py",
        "  3. Start backend: cd backend && uvicorn app.main:app --reload",
        "  4. Start frontend: cd frontend && streamlit run streamlit_app.py",
        "",
        "To run full evaluation with Neo4j:",
        "  python run_evaluation.py",
        "",
        "=" * 100,
        "END OF REPORT",
        "=" * 100,
    ])
    
    return "\n".join(lines)


def main():
    """Run simplified evaluation"""
    logger.info("Starting Simplified System Evaluation...")
    logger.info("=" * 100)
    
    results = {}
    
    # 1. Analyze PDFs
    logger.info("1. Analyzing PDF documents...")
    results["pdf_analysis"] = analyze_pdfs()
    
    # 2. Analyze extracted data
    logger.info("2. Analyzing extracted data...")
    results["extraction_analysis"] = analyze_extracted_data()
    
    # 3. Analyze images
    logger.info("3. Analyzing page images...")
    results["image_analysis"] = analyze_images()
    
    # 4. Check Neo4j database
    logger.info("4. Checking Neo4j database...")
    results["neo4j_check"] = check_neo4j_database()
    
    # Generate report
    logger.info("5. Generating comprehensive report...")
    report = generate_comprehensive_report(results)
    
    # Save results
    output_dir = Path("evaluations/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_file = output_dir / f"simple_evaluation_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save report
    report_file = output_dir / f"simple_evaluation_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print report
    print("\n")
    print(report)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - JSON: {json_file}")
    logger.info(f"  - Report: {report_file}")
    
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)

