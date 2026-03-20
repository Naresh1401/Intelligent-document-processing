"""
src/evaluation/accuracy.py
============================
Evaluation framework for IDP accuracy measurement.

Metrics:
  - Field-level accuracy:    % of fields exactly or approximately correct
  - Document-level accuracy: % of documents with all fields correct
  - Confidence calibration:  do high-confidence predictions actually score better?
  - F1 by field:             per-field precision and recall

Real-world benchmark targets:
  Best-in-class IDP systems: 95-99% field accuracy on clean documents
  Acceptable production threshold: 90%+ on structured docs, 80%+ on handwritten
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from loguru import logger


@dataclass
class FieldResult:
    field_name:     str
    predicted:      Any
    ground_truth:   Any
    confidence:     float
    is_correct:     bool
    is_close:       bool    # Approximate match (e.g. "01/15/2024" vs "January 15, 2024")


@dataclass
class EvalReport:
    n_documents:       int
    n_fields:          int
    exact_accuracy:    float    # % exactly correct
    approx_accuracy:   float    # % approximately correct
    doc_accuracy:      float    # % documents with all fields correct
    avg_confidence:    float
    confidence_calibration: float  # Correlation between confidence and correctness
    per_field_accuracy: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 52,
            "  IDP ACCURACY EVALUATION REPORT",
            "=" * 52,
            f"  Documents:          {self.n_documents}",
            f"  Fields evaluated:   {self.n_fields}",
            f"  Exact accuracy:     {self.exact_accuracy:.1%}",
            f"  Approx accuracy:    {self.approx_accuracy:.1%}",
            f"  Doc-level accuracy: {self.doc_accuracy:.1%}",
            f"  Avg confidence:     {self.avg_confidence:.3f}",
            f"  Conf calibration:   {self.confidence_calibration:.3f}",
        ]
        if self.per_field_accuracy:
            lines.append("\n  Per-field accuracy:")
            for fname, acc in sorted(self.per_field_accuracy.items(), key=lambda x: x[1]):
                bar = "█" * int(acc * 20)
                lines.append(f"    {fname:25s} {acc:.1%}  {bar}")
        lines.append("=" * 52)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "n_documents":       self.n_documents,
            "n_fields":          self.n_fields,
            "exact_accuracy":    round(self.exact_accuracy, 4),
            "approx_accuracy":   round(self.approx_accuracy, 4),
            "doc_accuracy":      round(self.doc_accuracy, 4),
            "avg_confidence":    round(self.avg_confidence, 4),
            "confidence_calibration": round(self.confidence_calibration, 4),
            "per_field_accuracy": {k: round(v, 4) for k, v in self.per_field_accuracy.items()},
        }


class IDPEvaluator:
    """
    Evaluates IDP extraction accuracy against a ground truth dataset.

    Ground truth format:
        [
            {
                "filename": "invoice_001.pdf",
                "fields": {
                    "invoice_number": "INV-2024-001",
                    "total_amount": 1250.00,
                    ...
                }
            }
        ]
    """

    def _normalize(self, value: Any) -> str:
        """Normalize a value for comparison."""
        if value is None:
            return ""
        s = str(value).lower().strip()
        # Remove common formatting
        s = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
        s = s.replace("/", "-").replace(".", "-")
        return s

    def _is_exact(self, pred: Any, truth: Any) -> bool:
        return self._normalize(pred) == self._normalize(truth)

    def _is_approx(self, pred: Any, truth: Any) -> bool:
        """Approximate match — handles date format variations, currency formats, etc."""
        if self._is_exact(pred, truth):
            return True
        p = self._normalize(pred)
        t = self._normalize(truth)
        if not p or not t:
            return False
        # Numeric proximity (within 1%)
        try:
            pf, tf = float(p.replace("-", ".")), float(t.replace("-", "."))
            return abs(pf - tf) / max(abs(tf), 1e-9) < 0.01
        except ValueError:
            pass
        # String containment
        return p in t or t in p

    def evaluate(
        self,
        predictions: List[Dict],  # List of ProcessingResult.to_dict()
        ground_truths: List[Dict],
    ) -> EvalReport:
        """
        Compare predictions against ground truths.

        Args:
            predictions:   List of extraction results (from processor.process().to_dict())
            ground_truths: List of ground truth dicts with 'filename' and 'fields'
        """
        # Build lookup by filename
        gt_map = {g["filename"]: g["fields"] for g in ground_truths}

        all_results: List[FieldResult] = []
        doc_all_correct: List[bool]    = []

        for pred in predictions:
            fname     = pred["filename"]
            gt_fields = gt_map.get(fname, {})
            if not gt_fields:
                logger.warning(f"No ground truth for {fname}")
                continue

            pred_fields    = pred.get("extracted_fields", {})
            doc_correct    = True
            doc_results    = []

            for field_name, gt_value in gt_fields.items():
                pred_field = pred_fields.get(field_name, {})
                pred_value = pred_field.get("value") if isinstance(pred_field, dict) else None
                confidence = pred_field.get("confidence", 0.0) if isinstance(pred_field, dict) else 0.0

                exact  = self._is_exact(pred_value, gt_value)
                approx = self._is_approx(pred_value, gt_value)

                if not approx:
                    doc_correct = False

                result = FieldResult(
                    field_name   = field_name,
                    predicted    = pred_value,
                    ground_truth = gt_value,
                    confidence   = confidence,
                    is_correct   = exact,
                    is_close     = approx,
                )
                doc_results.append(result)
                all_results.append(result)

            doc_all_correct.append(doc_correct)

        if not all_results:
            logger.warning("No results to evaluate.")
            return EvalReport(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Aggregate metrics
        exact_acc  = sum(r.is_correct for r in all_results) / len(all_results)
        approx_acc = sum(r.is_close   for r in all_results) / len(all_results)
        doc_acc    = sum(doc_all_correct) / max(len(doc_all_correct), 1)
        avg_conf   = sum(r.confidence for r in all_results) / len(all_results)

        # Confidence calibration: correlation between confidence and is_correct
        confs    = np.array([r.confidence for r in all_results])
        correct  = np.array([float(r.is_close) for r in all_results])
        if confs.std() > 0 and correct.std() > 0:
            calibration = float(np.corrcoef(confs, correct)[0, 1])
        else:
            calibration = 0.0

        # Per-field accuracy
        per_field: Dict[str, List[bool]] = {}
        for r in all_results:
            per_field.setdefault(r.field_name, []).append(r.is_close)
        per_field_acc = {k: sum(v) / len(v) for k, v in per_field.items()}

        report = EvalReport(
            n_documents            = len(doc_all_correct),
            n_fields               = len(all_results),
            exact_accuracy         = exact_acc,
            approx_accuracy        = approx_acc,
            doc_accuracy           = doc_acc,
            avg_confidence         = avg_conf,
            confidence_calibration = calibration,
            per_field_accuracy     = per_field_acc,
        )

        logger.info(str(report))
        return report
