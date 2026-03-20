"""
src/processing/classifier.py
==============================
Automatically classifies a document into its type (invoice, contract, resume, etc.)

Two strategies:
  1. Keyword classifier  — fast, zero cost, ~90% accurate on clean docs
  2. LLM classifier      — slower, API cost, ~98% accurate, handles ambiguous docs

Production pattern: run keyword classifier first.
If confidence < threshold, fall back to LLM classifier.
"""

import re
from typing import Dict, Tuple
from loguru import logger


# ── Keyword-based classifier ─────────────────────────────────────────────

KEYWORD_SIGNATURES: Dict[str, list] = {
    "invoice": [
        "invoice", "invoice number", "invoice #", "bill to", "due date",
        "subtotal", "tax", "total amount due", "payment terms",
        "remit to", "purchase order", "vendor",
    ],
    "contract": [
        "agreement", "contract", "whereas", "hereinafter", "party",
        "terms and conditions", "governing law", "termination", "indemnification",
        "liability", "confidentiality", "intellectual property", "force majeure",
    ],
    "resume": [
        "work experience", "education", "skills", "certifications",
        "references", "objective", "summary", "employment history",
        "curriculum vitae", "cv", "achievements", "projects",
    ],
    "medical_form": [
        "patient name", "date of birth", "diagnosis", "chief complaint",
        "blood pressure", "heart rate", "medication", "allergy",
        "physician", "medical record", "insurance", "symptoms",
    ],
    "bank_statement": [
        "account number", "account holder", "opening balance", "closing balance",
        "transaction", "debit", "credit", "statement period",
        "bank", "deposits", "withdrawals",
    ],
}


class KeywordClassifier:
    """
    Fast rule-based document classifier using keyword frequency.
    
    Each document type has a signature set of keywords.
    The class with the highest normalized keyword hit rate wins.
    """

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify document text.
        
        Returns:
            (doc_type, confidence) — confidence 0.0–1.0
        """
        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for doc_type, keywords in KEYWORD_SIGNATURES.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            scores[doc_type] = hits / len(keywords)

        if not scores or max(scores.values()) == 0:
            return "unknown", 0.0

        best_type  = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        # Normalize: map top score to a confidence
        # If the top score is > 3x the second-best, high confidence
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0)
        confidence = min(0.95, best_score * 2 + margin)

        logger.debug(f"Keyword classifier → {best_type} ({confidence:.2f})")
        return best_type, confidence


class LLMClassifier:
    """
    LLM-based document classifier for ambiguous or complex documents.
    More accurate than keyword matching, especially for mixed-language
    or non-standard document formats.
    """

    PROMPT = """You are a document classification expert.

Classify the following document into exactly one of these types:
- invoice
- contract
- resume
- medical_form
- bank_statement
- other

Respond with JSON only:
{{"type": "<type>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}

Document text (first 1000 chars):
{text}"""

    def __init__(self, llm):
        self.llm = llm

    def classify(self, text: str) -> Tuple[str, float]:
        import json
        prompt = self.PROMPT.format(text=text[:1000])
        try:
            response = self.llm.generate(
                "You are a precise document classifier. Always respond with valid JSON only.",
                prompt,
            )
            # Strip markdown fences if present
            clean = re.sub(r"```(?:json)?|```", "", response).strip()
            result = json.loads(clean)
            doc_type   = result.get("type", "other")
            confidence = float(result.get("confidence", 0.5))
            logger.debug(f"LLM classifier → {doc_type} ({confidence:.2f}): {result.get('reasoning','')}")
            return doc_type, confidence
        except Exception as e:
            logger.warning(f"LLM classifier failed: {e}")
            return "other", 0.0


class HybridClassifier:
    """
    Combines keyword and LLM classifiers.
    Uses keyword classifier first; falls back to LLM if confidence < threshold.
    """

    def __init__(self, llm=None, fallback_threshold: float = 0.30):
        self.keyword_clf   = KeywordClassifier()
        self.llm_clf       = LLMClassifier(llm) if llm else None
        self.threshold     = fallback_threshold

    def classify(self, text: str) -> Tuple[str, float]:
        doc_type, confidence = self.keyword_clf.classify(text)

        if confidence >= self.threshold or self.llm_clf is None:
            return doc_type, confidence

        # Fall back to LLM
        logger.info(f"Keyword confidence {confidence:.2f} < {self.threshold}, using LLM classifier")
        return self.llm_clf.classify(text)
