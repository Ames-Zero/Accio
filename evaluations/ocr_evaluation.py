"""
OCR and Text Extraction Evaluation
Metrics: Word Error Rate (WER), Character Error Rate (CER)
"""

import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
import difflib

logger = logging.getLogger(__name__)


@dataclass
class OCRMetrics:
    """OCR evaluation metrics"""
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    total_words: int
    total_chars: int
    word_errors: int
    char_errors: int


class OCREvaluator:
    """
    Evaluates OCR and text extraction quality using WER and CER metrics.
    Compares extracted text to manually transcribed ground truth.
    """
    
    def __init__(self, ground_truth_dir: Path = None):
        """
        Initialize OCR evaluator
        
        Args:
            ground_truth_dir: Directory containing ground truth transcriptions
        """
        self.ground_truth_dir = ground_truth_dir or Path("evaluations/ground_truth/ocr")
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"OCR Evaluator initialized with ground truth dir: {self.ground_truth_dir}")
    
    def calculate_wer(self, reference: str, hypothesis: str) -> Tuple[float, int, int]:
        """
        Calculate Word Error Rate (WER)
        
        Args:
            reference: Ground truth text
            hypothesis: Extracted text
            
        Returns:
            Tuple of (WER, word_errors, total_words)
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        
        # Use dynamic programming (Levenshtein distance) for word alignment
        # Initialize DP table
        n, m = len(ref_words), len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Base cases
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # deletion
                        dp[i][j-1] + 1,      # insertion
                        dp[i-1][j-1] + 1     # substitution
                    )
        
        word_errors = dp[n][m]
        total_words = len(ref_words)
        wer = word_errors / total_words if total_words > 0 else 0.0
        
        return wer, word_errors, total_words
    
    def calculate_cer(self, reference: str, hypothesis: str) -> Tuple[float, int, int]:
        """
        Calculate Character Error Rate (CER)
        
        Args:
            reference: Ground truth text
            hypothesis: Extracted text
            
        Returns:
            Tuple of (CER, char_errors, total_chars)
        """
        # Use difflib for character-level alignment
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        
        # Calculate Levenshtein distance at character level
        n, m = len(ref_chars), len(hyp_chars)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Base cases
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_chars[i-1] == hyp_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # deletion
                        dp[i][j-1] + 1,      # insertion
                        dp[i-1][j-1] + 1     # substitution
                    )
        
        char_errors = dp[n][m]
        total_chars = len(ref_chars)
        cer = char_errors / total_chars if total_chars > 0 else 0.0
        
        return cer, char_errors, total_chars
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison (lowercase, remove extra whitespace)
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase and normalize whitespace
        normalized = " ".join(text.lower().split())
        return normalized
    
    def evaluate_extraction(
        self,
        extracted_text: str,
        ground_truth_text: str,
        normalize: bool = True
    ) -> OCRMetrics:
        """
        Evaluate extracted text against ground truth
        
        Args:
            extracted_text: Text extracted from OCR
            ground_truth_text: Manually transcribed ground truth
            normalize: Whether to normalize text before comparison
            
        Returns:
            OCRMetrics object with WER, CER, and statistics
        """
        if normalize:
            extracted_text = self.normalize_text(extracted_text)
            ground_truth_text = self.normalize_text(ground_truth_text)
        
        wer, word_errors, total_words = self.calculate_wer(ground_truth_text, extracted_text)
        cer, char_errors, total_chars = self.calculate_cer(ground_truth_text, extracted_text)
        
        return OCRMetrics(
            wer=wer,
            cer=cer,
            total_words=total_words,
            total_chars=total_chars,
            word_errors=word_errors,
            char_errors=char_errors
        )
    
    def load_ground_truth(self, doc_id: str, page_num: int = None) -> str:
        """
        Load ground truth transcription for a document/page
        
        Args:
            doc_id: Document identifier
            page_num: Optional page number
            
        Returns:
            Ground truth text
        """
        if page_num:
            gt_file = self.ground_truth_dir / f"{doc_id}_page_{page_num}.txt"
        else:
            gt_file = self.ground_truth_dir / f"{doc_id}.txt"
        
        if not gt_file.exists():
            logger.warning(f"Ground truth file not found: {gt_file}")
            return ""
        
        with open(gt_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def evaluate_document(
        self,
        doc_id: str,
        extracted_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate OCR extraction for an entire document
        
        Args:
            doc_id: Document identifier
            extracted_data: Extracted document data (from extract_simple.py output)
            
        Returns:
            Dictionary with evaluation results per page and overall
        """
        results = {
            "doc_id": doc_id,
            "pages": [],
            "overall": {
                "total_pages": 0,
                "avg_wer": 0.0,
                "avg_cer": 0.0,
                "total_word_errors": 0,
                "total_char_errors": 0,
                "total_words": 0,
                "total_chars": 0
            }
        }
        
        total_wer = 0.0
        total_cer = 0.0
        page_count = 0
        
        for page_data in extracted_data.get("pages", []):
            page_num = page_data.get("page_num", 0)
            
            # Combine all text blocks from the page
            extracted_text = " ".join([
                block.get("text", "")
                for block in page_data.get("layout_blocks", [])
            ])
            
            # Load ground truth
            ground_truth = self.load_ground_truth(doc_id, page_num)
            
            if not ground_truth:
                logger.warning(f"No ground truth for {doc_id} page {page_num}, skipping")
                continue
            
            # Evaluate page
            metrics = self.evaluate_extraction(extracted_text, ground_truth)
            
            page_result = {
                "page_num": page_num,
                "wer": metrics.wer,
                "cer": metrics.cer,
                "word_errors": metrics.word_errors,
                "char_errors": metrics.char_errors,
                "total_words": metrics.total_words,
                "total_chars": metrics.total_chars
            }
            
            results["pages"].append(page_result)
            
            # Accumulate for overall metrics
            total_wer += metrics.wer
            total_cer += metrics.cer
            results["overall"]["total_word_errors"] += metrics.word_errors
            results["overall"]["total_char_errors"] += metrics.char_errors
            results["overall"]["total_words"] += metrics.total_words
            results["overall"]["total_chars"] += metrics.total_chars
            page_count += 1
        
        # Calculate overall averages
        if page_count > 0:
            results["overall"]["total_pages"] = page_count
            results["overall"]["avg_wer"] = total_wer / page_count
            results["overall"]["avg_cer"] = total_cer / page_count
        
        return results
    
    def evaluate_batch(
        self,
        extracted_files: List[Path],
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple extracted documents
        
        Args:
            extracted_files: List of paths to extracted JSON files
            output_file: Optional path to save results
            
        Returns:
            Dictionary with evaluation results for all documents
        """
        all_results = {
            "documents": [],
            "summary": {
                "total_documents": 0,
                "total_pages": 0,
                "overall_avg_wer": 0.0,
                "overall_avg_cer": 0.0
            }
        }
        
        total_wer = 0.0
        total_cer = 0.0
        total_docs = 0
        total_pages = 0
        
        for json_file in extracted_files:
            logger.info(f"Evaluating {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                extracted_data = json.load(f)
            
            doc_id = extracted_data.get("doc_id", json_file.stem)
            doc_results = self.evaluate_document(doc_id, extracted_data)
            
            all_results["documents"].append(doc_results)
            
            # Accumulate summary statistics
            if doc_results["overall"]["total_pages"] > 0:
                total_wer += doc_results["overall"]["avg_wer"]
                total_cer += doc_results["overall"]["avg_cer"]
                total_pages += doc_results["overall"]["total_pages"]
                total_docs += 1
        
        # Calculate overall summary
        if total_docs > 0:
            all_results["summary"]["total_documents"] = total_docs
            all_results["summary"]["total_pages"] = total_pages
            all_results["summary"]["overall_avg_wer"] = total_wer / total_docs
            all_results["summary"]["overall_avg_cer"] = total_cer / total_docs
        
        # Save results if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation results to {output_file}")
        
        return all_results

