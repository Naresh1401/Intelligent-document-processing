"""
src/processing/processor.py
=============================
Main IDP pipeline orchestrator.

Pipeline:
  Document (file) 
    → Text Extraction (PDF/DOCX/image OCR)
    → Document Classification
    → LLM Field Extraction  
    → Confidence Scoring & Validation
    → Human Review Queue (for low-confidence fields)
    → Structured JSON Output

This is what companies pay $50-200/document for from vendors like
ABBYY, Kofax, AWS Textract, and Azure Form Recogniser.
Building it yourself gives you full control + zero per-doc cost.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from loguru import logger

from src.processing.classifier import HybridClassifier
from src.processing.schemas import get_schema, SCHEMA_REGISTRY, ExtractedField, ConfidenceLevel
from src.extraction.llm_extractor import LLMExtractor


@dataclass
class ProcessingResult:
    """Full result of processing one document."""
    filename:            str
    doc_type:            str
    classification_conf: float
    extracted_fields:    Dict[str, Any]
    low_confidence_fields: List[str]    # Fields needing human review
    needs_review:        bool
    processing_time_ms:  float
    page_count:          int            = 0
    error:               Optional[str] = None

    def to_dict(self) -> Dict:
        def serialize(v):
            if isinstance(v, ExtractedField):
                return {"value": v.value, "confidence": v.confidence, "raw_text": v.raw_text}
            if isinstance(v, list):
                return [serialize(i) for i in v]
            if hasattr(v, "__dict__"):
                return v.__dict__
            return v

        return {
            "filename":               self.filename,
            "doc_type":               self.doc_type,
            "classification_confidence": round(self.classification_conf, 3),
            "extracted_fields":       {k: serialize(v) for k, v in self.extracted_fields.items()},
            "low_confidence_fields":  self.low_confidence_fields,
            "needs_review":           self.needs_review,
            "processing_time_ms":     round(self.processing_time_ms, 1),
            "page_count":             self.page_count,
            "error":                  self.error,
        }

    @property
    def summary(self) -> str:
        total = len(self.extracted_fields)
        low   = len(self.low_confidence_fields)
        return (
            f"{self.filename} → {self.doc_type} "
            f"({self.classification_conf:.0%} conf) | "
            f"{total - low}/{total} fields high-confidence | "
            f"{'⚠ REVIEW NEEDED' if self.needs_review else '✅ AUTO-PROCESSED'} | "
            f"{self.processing_time_ms:.0f}ms"
        )


class TextExtractor:
    """Extracts raw text from various file formats."""

    def extract(self, file_path: str) -> tuple[str, int]:
        """Returns (text, page_count)."""
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext in (".docx", ".doc"):
            return self._extract_docx(file_path), 1
        elif ext == ".txt" or ext == ".md":
            return Path(file_path).read_text(encoding="utf-8", errors="ignore"), 1
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
            return self._extract_image_ocr(file_path), 1
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf(self, path: str) -> tuple[str, int]:
        import pypdf
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            pages  = len(reader.pages)
            text   = "\n\n".join(
                p.extract_text() or "" for p in reader.pages
            )
        return text, pages

    def _extract_docx(self, path: str) -> str:
        from docx import Document
        doc   = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _extract_image_ocr(self, path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(path)
            return pytesseract.image_to_string(img)
        except ImportError:
            logger.warning("pytesseract/PIL not installed — OCR skipped")
            return ""


class ConfidenceAnalyzer:
    """
    Analyses extraction results to identify low-confidence fields
    and determine if human review is needed.
    """

    def __init__(
        self,
        high_threshold: float = 0.85,
        low_threshold:  float = 0.50,
        review_if_pct_low: float = 0.30,  # Review if >30% of fields are low confidence
    ):
        self.high_threshold     = high_threshold
        self.low_threshold      = low_threshold
        self.review_if_pct_low  = review_if_pct_low

    def analyze(self, extracted: Dict[str, Any]) -> tuple[List[str], bool]:
        """
        Returns (low_confidence_fields, needs_review).

        A document needs review if any critical field is low confidence
        or if more than `review_if_pct_low` of all fields are low confidence.
        """
        low_conf_fields = []
        total_scored    = 0

        for name, value in extracted.items():
            if isinstance(value, ExtractedField):
                total_scored += 1
                if value.confidence < self.low_threshold:
                    low_conf_fields.append(name)

        if total_scored == 0:
            return [], False

        pct_low      = len(low_conf_fields) / total_scored
        needs_review = pct_low >= self.review_if_pct_low

        return low_conf_fields, needs_review


class DocumentProcessor:
    """
    End-to-end IDP pipeline.

    Usage:
        processor = DocumentProcessor(llm=OpenAILLM())
        result = processor.process("invoice.pdf")
        print(result.summary)
        result.to_dict()  # full JSON output
    """

    def __init__(
        self,
        llm,
        high_conf_threshold: float = 0.85,
        low_conf_threshold:  float = 0.50,
        output_dir: Optional[str] = None,
    ):
        self.text_extractor   = TextExtractor()
        self.classifier       = HybridClassifier(llm=llm)
        self.extractor        = LLMExtractor(llm)
        self.conf_analyzer    = ConfidenceAnalyzer(high_conf_threshold, low_conf_threshold)
        self.output_dir       = output_dir

    def process(
        self,
        file_path: str,
        doc_type: Optional[str] = None,  # Override auto-classification
    ) -> ProcessingResult:
        """
        Process a single document end-to-end.

        Args:
            file_path: Path to the document
            doc_type:  Optional override for document type classification

        Returns:
            ProcessingResult with extracted fields and confidence scores
        """
        t0       = time.perf_counter()
        filename = Path(file_path).name
        logger.info(f"Processing: {filename}")

        try:
            # 1. Extract text
            text, page_count = self.text_extractor.extract(file_path)
            if not text.strip():
                raise ValueError("No text could be extracted from document")

            # 2. Classify (unless overridden)
            if doc_type:
                classification_conf = 1.0
            else:
                doc_type, classification_conf = self.classifier.classify(text)

            logger.info(f"Classified as '{doc_type}' ({classification_conf:.0%})")

            # 3. Extract fields
            schema_class    = get_schema(doc_type)
            extracted_fields = self.extractor.extract(text, doc_type, schema_class)

            # 4. Confidence analysis
            low_conf_fields, needs_review = self.conf_analyzer.analyze(extracted_fields)

            processing_time = (time.perf_counter() - t0) * 1000

            result = ProcessingResult(
                filename             = filename,
                doc_type             = doc_type,
                classification_conf  = classification_conf,
                extracted_fields     = extracted_fields,
                low_confidence_fields = low_conf_fields,
                needs_review         = needs_review,
                processing_time_ms   = processing_time,
                page_count           = page_count,
            )

            logger.info(result.summary)

            # 5. Save output
            if self.output_dir:
                self._save_result(result, filename)

            return result

        except Exception as e:
            logger.error(f"Processing failed for {filename}: {e}")
            return ProcessingResult(
                filename             = filename,
                doc_type             = doc_type or "unknown",
                classification_conf  = 0.0,
                extracted_fields     = {},
                low_confidence_fields = [],
                needs_review         = True,
                processing_time_ms   = (time.perf_counter() - t0) * 1000,
                error                = str(e),
            )

    def process_batch(
        self,
        file_paths: List[str],
        doc_type: Optional[str] = None,
    ) -> List[ProcessingResult]:
        """Process multiple documents and return results."""
        results = []
        for i, fp in enumerate(file_paths):
            logger.info(f"[{i+1}/{len(file_paths)}] {Path(fp).name}")
            results.append(self.process(fp, doc_type=doc_type))

        total     = len(results)
        reviewed  = sum(1 for r in results if r.needs_review)
        auto      = total - reviewed
        avg_time  = sum(r.processing_time_ms for r in results) / max(total, 1)

        logger.info(
            f"\nBatch complete: {total} docs | "
            f"{auto} auto-processed | {reviewed} need review | "
            f"avg {avg_time:.0f}ms/doc"
        )
        return results

    def _save_result(self, result: ProcessingResult, filename: str):
        """Save extraction result to JSON."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem    = Path(filename).stem
        out_path = out_dir / f"{stem}_extracted.json"
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.debug(f"Saved result → {out_path}")
