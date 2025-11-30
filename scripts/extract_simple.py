"""
Phase 1: Simple PDF Extraction
Extracts text and generates page screenshots from PDFs using PyMuPDF.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import logging
from pdf2image import convert_from_path
import fitz  # PyMuPDF

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# DPI for screenshots
SCREENSHOT_DPI = 300


class SimplePDFExtractor:
    """Extract content from PDFs using PyMuPDF"""
    
    def extract_pdf(self, pdf_path: Path, company_id: str) -> Dict[str, Any]:
        """
        Extract content from a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            company_id: Company identifier
        
        Returns:
            Dictionary with extracted content
        """
        logger.info(f"Processing {pdf_path.name} for {company_id}")
        
        # Generate document ID from filename
        doc_id = pdf_path.stem
        
        # Setup paths
        company_images_dir = DATA_DIR / company_id / "images"
        company_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Generate page screenshots
        logger.info(f"Generating screenshots for {pdf_path.name}")
        page_images = self._generate_screenshots(pdf_path, company_images_dir, doc_id)
        
        # Extract content page by page
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_data = {
                "page_num": page_num + 1,
                "image_path": str(page_images[page_num]),
                "layout_blocks": [],
                "tables": [],
                "figures": []
            }
            
            # Extract text blocks
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:  # Text block
                    text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"] + " "
                    
                    if text.strip():
                        page_data["layout_blocks"].append({
                            "type": "paragraph",
                            "text": text.strip(),
                            "bbox": block["bbox"],
                            "confidence": 0.95
                        })
            
            pages_data.append(page_data)
            logger.info(f"Extracted page {page_num + 1}/{len(doc)} - {len(page_data['layout_blocks'])} blocks")
        
        doc.close()
        
        # Build final output structure
        output = {
            "doc_id": doc_id,
            "company_id": company_id,
            "filename": pdf_path.name,
            "total_pages": len(pages_data),
            "pages": pages_data
        }
        
        return output
    
    def _generate_screenshots(
        self, 
        pdf_path: Path, 
        output_dir: Path, 
        doc_id: str
    ) -> List[str]:
        """
        Generate page screenshots from PDF
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save screenshots
            doc_id: Document identifier
        
        Returns:
            List of image paths (relative to project root)
        """
        image_paths = []
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                str(pdf_path), 
                dpi=SCREENSHOT_DPI,
                fmt='png'
            )
            
            for page_num, image in enumerate(images, start=1):
                # Build filename
                image_filename = f"{doc_id}_page_{page_num}.png"
                image_path = output_dir / image_filename
                
                # Save image
                image.save(str(image_path), 'PNG')
                
                # Store relative path from project root
                relative_path = str(image_path.relative_to(PROJECT_ROOT))
                image_paths.append(relative_path)
            
            logger.info(f"Generated {len(images)} screenshots")
            
        except Exception as e:
            logger.error(f"Failed to generate screenshots: {e}")
            raise
        
        return image_paths


def process_company(company_id: str):
    """
    Process all PDFs for a specific company
    
    Args:
        company_id: Company identifier
    """
    logger.info(f"Starting processing for {company_id}")
    
    # Setup paths
    pdfs_dir = DATA_DIR / company_id / "pdfs"
    output_dir = PROCESSED_DIR / company_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if PDF directory exists
    if not pdfs_dir.exists():
        logger.error(f"PDF directory not found: {pdfs_dir}")
        return
    
    # Get all PDF files
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdfs_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files for {company_id}")
    
    # Initialize extractor
    extractor = SimplePDFExtractor()
    
    # Process each PDF
    for i, pdf_path in enumerate(pdf_files, start=1):
        logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_path.name}")
        
        try:
            # Extract content
            extracted_data = extractor.extract_pdf(pdf_path, company_id)
            
            # Save to JSON
            output_file = output_dir / f"{pdf_path.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved extraction to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
            continue
    
    logger.info(f"Completed processing for {company_id}")


def main():
    """Main execution function"""
    logger.info("="*60)
    logger.info("PDF EXTRACTION - Phase 1 (Simplified)")
    logger.info("="*60)
    
    # List of companies to process
    companies = ["company_1"]
    
    for company_id in companies:
        try:
            process_company(company_id)
        except Exception as e:
            logger.error(f"Failed to process {company_id}: {e}", exc_info=True)
    
    logger.info("="*60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Processed files saved to: {PROCESSED_DIR}")
    logger.info(f"Page screenshots saved to: data/{{company_id}}/images/")


if __name__ == "__main__":
    main()
