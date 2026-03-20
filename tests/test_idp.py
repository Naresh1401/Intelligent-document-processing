"""
tests/test_idp.py — Unit tests for IDP pipeline
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.processing.classifier import KeywordClassifier
from src.processing.schemas import InvoiceSchema, ResumeSchema, get_schema, SCHEMA_REGISTRY
from src.evaluation.accuracy import IDPEvaluator


# ── Classifier tests ──────────────────────────────────────────────────────

def test_keyword_classifier_invoice():
    clf  = KeywordClassifier()
    text = "INVOICE #INV-2024-001\nBill To: Acme Corp\nSubtotal: $1,000\nTax: $100\nTotal Amount Due: $1,100\nPayment Terms: Net 30"
    doc_type, conf = clf.classify(text)
    assert doc_type == "invoice"
    assert conf > 0.2

def test_keyword_classifier_resume():
    clf  = KeywordClassifier()
    text = "John Doe\nWork Experience\nSoftware Engineer at Google\nEducation\nB.S. Computer Science\nSkills: Python, Java, SQL\nCertifications: AWS Solutions Architect"
    doc_type, conf = clf.classify(text)
    assert doc_type == "resume"
    assert conf > 0.2

def test_keyword_classifier_contract():
    clf  = KeywordClassifier()
    text = "SERVICE AGREEMENT\nThis Agreement is entered into between Party A and Party B.\nWhereas the parties wish to set forth terms and conditions.\nGoverning Law: California\nTermination: 30 days notice"
    doc_type, conf = clf.classify(text)
    assert doc_type == "contract"

def test_keyword_classifier_unknown():
    clf  = KeywordClassifier()
    text = "The quick brown fox jumps over the lazy dog."
    doc_type, conf = clf.classify(text)
    # Low confidence for unrecognised text
    assert conf < 0.3

def test_keyword_classifier_bank_statement():
    clf  = KeywordClassifier()
    text = "Account Number: 1234567890\nOpening Balance: $5,000.00\nClosing Balance: $4,250.00\nTotal Debits: $850.00\nTotal Credits: $100.00\nStatement Period: January 2024"
    doc_type, conf = clf.classify(text)
    assert doc_type == "bank_statement"


# ── Schema tests ──────────────────────────────────────────────────────────

def test_get_schema_valid():
    schema = get_schema("invoice")
    assert schema == InvoiceSchema

def test_get_schema_invalid():
    with pytest.raises(ValueError, match="Unknown document type"):
        get_schema("unknown_type")

def test_schema_registry_complete():
    expected = {"invoice", "contract", "resume", "medical_form", "bank_statement"}
    assert expected == set(SCHEMA_REGISTRY.keys())

def test_invoice_schema_fields():
    schema = InvoiceSchema()
    assert hasattr(schema, "invoice_number")
    assert hasattr(schema, "total_amount")
    assert hasattr(schema, "line_items")
    assert hasattr(schema, "vendor_name")

def test_resume_schema_fields():
    schema = ResumeSchema()
    assert hasattr(schema, "full_name")
    assert hasattr(schema, "email")
    assert hasattr(schema, "work_experience")
    assert hasattr(schema, "skills")


# ── Evaluator tests ───────────────────────────────────────────────────────

def make_prediction(filename, fields_with_conf):
    """Helper to create a mock prediction dict."""
    extracted = {k: {"value": v, "confidence": c} for k, v, c in fields_with_conf}
    return {
        "filename":              filename,
        "doc_type":              "invoice",
        "extracted_fields":      extracted,
        "low_confidence_fields": [],
        "needs_review":          False,
    }

def test_evaluator_perfect_accuracy():
    ev   = IDPEvaluator()
    pred = [make_prediction("inv.pdf", [
        ("invoice_number", "INV-001", 0.95),
        ("total_amount",   "1250.00", 0.90),
    ])]
    gt   = [{"filename": "inv.pdf", "fields": {"invoice_number": "INV-001", "total_amount": "1250.00"}}]
    report = ev.evaluate(pred, gt)
    assert report.exact_accuracy == 1.0

def test_evaluator_zero_accuracy():
    ev   = IDPEvaluator()
    pred = [make_prediction("inv.pdf", [
        ("invoice_number", "WRONG-999", 0.3),
        ("total_amount",   "9999.00", 0.4),
    ])]
    gt   = [{"filename": "inv.pdf", "fields": {"invoice_number": "INV-001", "total_amount": "1250.00"}}]
    report = ev.evaluate(pred, gt)
    assert report.exact_accuracy == 0.0

def test_evaluator_approx_match():
    """Date format differences should count as approximate match."""
    ev   = IDPEvaluator()
    pred = [make_prediction("inv.pdf", [("invoice_date", "01/15/2024", 0.8)])]
    gt   = [{"filename": "inv.pdf", "fields": {"invoice_date": "01-15-2024"}}]
    report = ev.evaluate(pred, gt)
    assert report.approx_accuracy >= 0.5  # Approx match

def test_evaluator_per_field():
    ev   = IDPEvaluator()
    pred = [make_prediction("inv.pdf", [
        ("invoice_number", "INV-001", 0.95),
        ("total_amount",   "WRONG",   0.3),
    ])]
    gt   = [{"filename": "inv.pdf", "fields": {
        "invoice_number": "INV-001",
        "total_amount":   "1250.00",
    }}]
    report = ev.evaluate(pred, gt)
    assert report.per_field_accuracy["invoice_number"] == 1.0
    assert report.per_field_accuracy["total_amount"]   == 0.0

def test_evaluator_no_ground_truth():
    """Should handle missing ground truth gracefully."""
    ev   = IDPEvaluator()
    pred = [make_prediction("no_gt.pdf", [("field", "value", 0.9)])]
    gt   = []
    report = ev.evaluate(pred, gt)
    assert report.n_documents == 0


if __name__ == "__main__":
    # Run manually
    tests = [
        test_keyword_classifier_invoice,
        test_keyword_classifier_resume,
        test_keyword_classifier_contract,
        test_keyword_classifier_unknown,
        test_keyword_classifier_bank_statement,
        test_get_schema_valid,
        test_get_schema_invalid,
        test_schema_registry_complete,
        test_invoice_schema_fields,
        test_resume_schema_fields,
        test_evaluator_perfect_accuracy,
        test_evaluator_zero_accuracy,
        test_evaluator_approx_match,
        test_evaluator_per_field,
        test_evaluator_no_ground_truth,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
