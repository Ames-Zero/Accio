#!/usr/bin/env python3
"""
PDF extraction using Docling library.
Extracts text, tables, and structure from PDFs with bounding boxes.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
except ImportError:
    print("ERROR: Docling is not installed. Install with:")
    print("pip install docling")
    sys.exit(1)


class DoclingPDFExtractor:
    """Extract content from PDFs using Docling."""
    
    def __init__(self, companies_file: str = "data/companies.yaml"):
        self.companies_file = Path(companies_file)
        self.companies = self._load_companies()
        
        # Configure Docling pipeline
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = False  # Set to True if PDFs are scanned images
        self.pipeline_options.do_table_structure = True  # Extract table structure
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: self.pipeline_options,
            }
        )
    
    def _load_companies(self) -> Dict[str, Any]:
        """Load company configuration."""
        if not self.companies_file.exists():
            raise FileNotFoundError(f"Companies file not found: {self.companies_file}")
        
        with open(self.companies_file, 'r') as f:
            data = yaml.safe_load(f)
            return {c['id']: c for c in data['companies']}
    
    def extract_pdf(self, pdf_path: Path, company_id: str, doc_id: str) -> Dict[str, Any]:
        """
        Extract content from a single PDF using Docling.
        
        Args:
            pdf_path: Path to PDF file
            company_id: Company identifier
            doc_id: Document identifier
            
        Returns:
            Dictionary with extracted content
        """
        print(f"  Processing: {pdf_path.name}")
        
        # Convert PDF using Docling
        result = self.converter.convert(str(pdf_path))
        
        # Extract document structure
        doc_data = {
            "document_id": doc_id,
            "company": company_id,
            "pdf_filename": pdf_path.name,
            "pages": []
        }
        
        # Process each page
        for page_num, page in enumerate(result.pages, start=1):
            page_data = {
                "page_num": page_num,
                "width": page.size.width if hasattr(page, 'size') else None,
                "height": page.size.height if hasattr(page, 'size') else None,
                "blocks": []
            }
            
            # Extract text blocks with bounding boxes
            block_idx = 0
            for item in page.items:
                # Get bounding box if available
                bbox = None
                if hasattr(item, 'prov') and item.prov:
                    for prov in item.prov:
                        if hasattr(prov, 'bbox'):
                            # Docling bbox format: [x0, y0, x1, y1]
                            bbox = [
                                prov.bbox.l,  # left (x0)
                                prov.bbox.t,  # top (y0)
                                prov.bbox.r,  # right (x1)
                                prov.bbox.b   # bottom (y1)
                            ]
                            break
                
                # Extract text content
                text = ""
                if hasattr(item, 'text'):
                    text = item.text
                elif hasattr(item, 'export_to_markdown'):
                    text = item.export_to_markdown()
                
                if text.strip():
                    block_data = {
                        "block_idx": block_idx,
                        "block_type": item.label if hasattr(item, 'label') else "text",
                        "text": text.strip(),
                        "bbox": bbox
                    }
                    page_data["blocks"].append(block_data)
                    block_idx += 1
            
            if page_data["blocks"]:
                doc_data["pages"].append(page_data)
        
        return doc_data
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # At least 50% through chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def process_company(self, company_id: str):
        """Process all PDFs for a company."""
        company = self.companies.get(company_id)
        if not company:
            print(f"ERROR: Company '{company_id}' not found")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing Company: {company['name']}")
        print(f"{'='*60}")
        
        # Setup paths
        pdf_dir = Path(f"data/{company_id}/pdfs")
        output_dir = Path(f"data/processed/{company_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all PDF files
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return
        
        print(f"Found {len(pdf_files)} PDF files")
        
        # Process each PDF
        for idx, pdf_path in enumerate(tqdm(pdf_files, desc="Extracting PDFs"), start=1):
            doc_id = f"doc_{idx:03d}"
            
            try:
                # Extract content
                doc_data = self.extract_pdf(pdf_path, company_id, doc_id)
                
                # Add chunks to blocks
                for page in doc_data["pages"]:
                    for block in page["blocks"]:
                        block["chunks"] = self.chunk_text(block["text"])
                
                # Save to JSON
                output_file = output_dir / f"{doc_id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(doc_data, f, indent=2, ensure_ascii=False)
                
                print(f"Saved: {output_file.name} ({len(doc_data['pages'])} pages)")
                
            except Exception as e:
                print(f"ERROR processing {pdf_path.name}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Extraction complete for {company['name']}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
    
    def process_all_companies(self):
        """Process PDFs for all companies."""
        for company_id in self.companies.keys():
            self.process_company(company_id)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract PDF content using Docling"
    )
    parser.add_argument(
        "--company",
        type=str,
        help="Company ID to process (e.g., company_1). If not specified, processes all companies."
    )
    parser.add_argument(
        "--companies-file",
        type=str,
        default="data/companies.yaml",
        help="Path to companies YAML file"
    )
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = DoclingPDFExtractor(companies_file=args.companies_file)
    
    # Process companies
    if args.company:
        extractor.process_company(args.company)
    else:
        extractor.process_all_companies()


if __name__ == "__main__":
    main()
