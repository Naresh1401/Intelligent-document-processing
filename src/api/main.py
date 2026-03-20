"""
src/api/main.py
================
FastAPI REST API for the Intelligent Document Processing system.

Endpoints:
  POST /process           — Upload and process a document
  POST /process/batch     — Process multiple documents
  GET  /document-types    — List supported document types
  POST /evaluate          — Run accuracy evaluation
  GET  /health            — Health check
  GET  /queue             — Get human review queue
  POST /queue/{id}/resolve — Resolve a review item
"""

import os
import json
import time
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from src.processing.schemas import SCHEMA_REGISTRY
from src.processing.processor import DocumentProcessor, ProcessingResult


# ── State ────────────────────────────────────────────────────────────────
_processor: Optional[DocumentProcessor] = None
_review_queue: Dict[str, Dict] = {}  # id → result dict
_stats = {"processed": 0, "auto": 0, "reviewed": 0, "total_time_ms": 0.0}


def get_processor() -> DocumentProcessor:
    global _processor
    if _processor is None:
        from src.extraction.llm_extractor import OpenAILLM, AnthropicLLM
        provider = os.getenv("LLM_PROVIDER", "openai")
        if provider == "anthropic":
            llm = AnthropicLLM(model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"))
        else:
            llm = OpenAILLM(model=os.getenv("LLM_MODEL", "gpt-4o"))
        _processor = DocumentProcessor(
            llm=llm,
            output_dir=os.getenv("OUTPUT_DIR", "./data/outputs"),
        )
    return _processor


# ── Pydantic models ───────────────────────────────────────────────────────

class ProcessResponse(BaseModel):
    id:                    str
    filename:              str
    doc_type:              str
    classification_conf:   float
    extracted_fields:      Dict[str, Any]
    low_confidence_fields: List[str]
    needs_review:          bool
    processing_time_ms:    float
    page_count:            int
    error:                 Optional[str] = None


class ReviewItem(BaseModel):
    id:       str
    filename: str
    doc_type: str
    low_confidence_fields: List[str]
    extracted_fields: Dict[str, Any]


class EvalRequest(BaseModel):
    ground_truths: List[Dict[str, Any]]   # [{filename, fields: {...}}]


# ── App ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting IDP API...")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Intelligent Document Processing API",
    description = "Extract structured data from PDFs, DOCX, and images using LLMs",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────

def result_to_response(result: ProcessingResult, doc_id: str) -> ProcessResponse:
    def serialize(v):
        from src.processing.schemas import ExtractedField
        if isinstance(v, ExtractedField):
            return {"value": v.value, "confidence": v.confidence, "raw_text": v.raw_text}
        if isinstance(v, list):
            return [serialize(i) for i in v]
        if hasattr(v, "__dict__"):
            return v.__dict__
        return v

    return ProcessResponse(
        id                    = doc_id,
        filename              = result.filename,
        doc_type              = result.doc_type,
        classification_conf   = result.classification_conf,
        extracted_fields      = {k: serialize(v) for k, v in result.extracted_fields.items()},
        low_confidence_fields = result.low_confidence_fields,
        needs_review          = result.needs_review,
        processing_time_ms    = result.processing_time_ms,
        page_count            = result.page_count,
        error                 = result.error,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/document-types")
async def document_types():
    return {
        "supported_types": list(SCHEMA_REGISTRY.keys()),
        "description": {
            "invoice":       "Invoices and bills with line items",
            "contract":      "Legal agreements and contracts",
            "resume":        "CVs and resumes",
            "medical_form":  "Patient records and medical intake forms",
            "bank_statement": "Bank statements with transactions",
        }
    }


@app.get("/stats")
async def stats():
    return {
        "total_processed":  _stats["processed"],
        "auto_processed":   _stats["auto"],
        "sent_to_review":   _stats["reviewed"],
        "review_queue_size": len(_review_queue),
        "avg_processing_ms": _stats["total_time_ms"] / max(_stats["processed"], 1),
    }


@app.post("/process", response_model=ProcessResponse)
async def process_document(
    file:     UploadFile = File(...),
    doc_type: Optional[str] = Form(None),
):
    """
    Upload and process a single document.
    Returns extracted fields with confidence scores.
    Automatically queues for human review if confidence is low.
    """
    processor = get_processor()
    doc_id    = str(uuid.uuid4())[:8]

    allowed_exts = {".pdf", ".docx", ".doc", ".txt", ".md", ".png", ".jpg", ".jpeg"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / file.filename
        tmp_path.write_bytes(await file.read())

        result = processor.process(str(tmp_path), doc_type=doc_type)

    # Update stats
    _stats["processed"]    += 1
    _stats["total_time_ms"] += result.processing_time_ms
    if result.needs_review:
        _stats["reviewed"] += 1
        _review_queue[doc_id] = result.to_dict()
    else:
        _stats["auto"] += 1

    return result_to_response(result, doc_id)


@app.post("/process/batch")
async def process_batch(
    files:    List[UploadFile] = File(...),
    doc_type: Optional[str]    = Form(None),
):
    """Process multiple documents in a batch."""
    processor = get_processor()
    responses = []

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for upload in files:
            tmp_path = Path(tmpdir) / upload.filename
            tmp_path.write_bytes(await upload.read())
            paths.append(str(tmp_path))

        results = processor.process_batch(paths, doc_type=doc_type)

    for result in results:
        doc_id = str(uuid.uuid4())[:8]
        _stats["processed"] += 1
        _stats["total_time_ms"] += result.processing_time_ms
        if result.needs_review:
            _stats["reviewed"] += 1
            _review_queue[doc_id] = result.to_dict()
        else:
            _stats["auto"] += 1
        responses.append(result_to_response(result, doc_id).model_dump())

    return {
        "total":        len(responses),
        "auto":         sum(1 for r in responses if not r["needs_review"]),
        "needs_review": sum(1 for r in responses if r["needs_review"]),
        "results":      responses,
    }


@app.get("/queue", response_model=List[ReviewItem])
async def get_review_queue():
    """Get all documents pending human review."""
    items = []
    for doc_id, data in _review_queue.items():
        items.append(ReviewItem(
            id                    = doc_id,
            filename              = data["filename"],
            doc_type              = data["doc_type"],
            low_confidence_fields = data["low_confidence_fields"],
            extracted_fields      = data["extracted_fields"],
        ))
    return items


@app.post("/queue/{doc_id}/resolve")
async def resolve_review(doc_id: str, corrections: Dict[str, Any]):
    """
    Submit human-corrected field values for a review item.
    Removes from queue and saves corrected result.
    """
    if doc_id not in _review_queue:
        raise HTTPException(404, f"Review item not found: {doc_id}")

    data = _review_queue.pop(doc_id)
    # Apply corrections
    for field, value in corrections.items():
        if field in data["extracted_fields"]:
            data["extracted_fields"][field]["value"]      = value
            data["extracted_fields"][field]["confidence"] = 1.0  # Human verified
            data["extracted_fields"][field]["raw_text"]   = f"[human corrected: {value}]"

    data["needs_review"]          = False
    data["low_confidence_fields"] = []

    # Save corrected result
    out_dir = Path(os.getenv("OUTPUT_DIR", "./data/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(data["filename"]).stem
    with open(out_dir / f"{stem}_corrected.json", "w") as f:
        json.dump(data, f, indent=2)

    return {"status": "resolved", "doc_id": doc_id}


@app.post("/evaluate")
async def evaluate(request: EvalRequest, files: List[UploadFile] = File(...)):
    """Run accuracy evaluation against ground truth labels."""
    from src.evaluation.accuracy import IDPEvaluator
    processor = get_processor()
    evaluator = IDPEvaluator()

    predictions = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            tmp_path = Path(tmpdir) / upload.filename
            tmp_path.write_bytes(await upload.read())
            result = processor.process(str(tmp_path))
            predictions.append(result.to_dict())

    report = evaluator.evaluate(predictions, request.ground_truths)
    return report.to_dict()
