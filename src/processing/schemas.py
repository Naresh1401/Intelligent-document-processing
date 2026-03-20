"""
src/processing/schemas.py
==========================
Pydantic schemas defining the structured output for each document type.

Real-world insight: IDP value comes from having a DEFINED schema per document type.
The LLM extraction quality is only as good as how precisely you define
what you want to extract. Vague schemas → vague extractions.

Supported document types:
  - Invoice
  - Contract
  - Medical Form / Patient Record
  - Resume / CV
  - Bank Statement
  - Purchase Order
  - Custom (user-defined JSON schema)
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import date
from enum import Enum


class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # >= 0.85
    MEDIUM = "medium"  # 0.50 - 0.85
    LOW    = "low"     # < 0.50


class ExtractedField(BaseModel):
    """A single extracted field with its confidence score."""
    value:      Optional[Any]   = None
    confidence: float           = Field(0.0, ge=0.0, le=1.0)
    raw_text:   Optional[str]   = None   # The raw text the value was extracted from

    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.50:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW


# ── Invoice ──────────────────────────────────────────────────────────────

class LineItem(BaseModel):
    description: Optional[str]   = None
    quantity:    Optional[float] = None
    unit_price:  Optional[float] = None
    total:       Optional[float] = None
    tax_rate:    Optional[float] = None


class InvoiceSchema(BaseModel):
    """Structured extraction schema for invoices."""
    invoice_number:   ExtractedField = Field(default_factory=ExtractedField)
    invoice_date:     ExtractedField = Field(default_factory=ExtractedField)
    due_date:         ExtractedField = Field(default_factory=ExtractedField)
    vendor_name:      ExtractedField = Field(default_factory=ExtractedField)
    vendor_address:   ExtractedField = Field(default_factory=ExtractedField)
    vendor_tax_id:    ExtractedField = Field(default_factory=ExtractedField)
    customer_name:    ExtractedField = Field(default_factory=ExtractedField)
    customer_address: ExtractedField = Field(default_factory=ExtractedField)
    line_items:       List[LineItem] = Field(default_factory=list)
    subtotal:         ExtractedField = Field(default_factory=ExtractedField)
    tax_amount:       ExtractedField = Field(default_factory=ExtractedField)
    total_amount:     ExtractedField = Field(default_factory=ExtractedField)
    currency:         ExtractedField = Field(default_factory=ExtractedField)
    payment_terms:    ExtractedField = Field(default_factory=ExtractedField)
    bank_details:     ExtractedField = Field(default_factory=ExtractedField)


# ── Contract ─────────────────────────────────────────────────────────────

class ContractParty(BaseModel):
    name:    Optional[str] = None
    role:    Optional[str] = None
    address: Optional[str] = None


class ContractSchema(BaseModel):
    """Structured extraction schema for contracts and agreements."""
    contract_title:      ExtractedField       = Field(default_factory=ExtractedField)
    contract_type:       ExtractedField       = Field(default_factory=ExtractedField)
    effective_date:      ExtractedField       = Field(default_factory=ExtractedField)
    expiry_date:         ExtractedField       = Field(default_factory=ExtractedField)
    parties:             List[ContractParty]  = Field(default_factory=list)
    governing_law:       ExtractedField       = Field(default_factory=ExtractedField)
    contract_value:      ExtractedField       = Field(default_factory=ExtractedField)
    payment_terms:       ExtractedField       = Field(default_factory=ExtractedField)
    notice_period:       ExtractedField       = Field(default_factory=ExtractedField)
    termination_clauses: ExtractedField       = Field(default_factory=ExtractedField)
    key_obligations:     List[str]            = Field(default_factory=list)
    auto_renewal:        ExtractedField       = Field(default_factory=ExtractedField)
    confidentiality:     ExtractedField       = Field(default_factory=ExtractedField)


# ── Resume / CV ───────────────────────────────────────────────────────────

class WorkExperience(BaseModel):
    company:    Optional[str] = None
    title:      Optional[str] = None
    start_date: Optional[str] = None
    end_date:   Optional[str] = None
    description: Optional[str] = None


class Education(BaseModel):
    institution: Optional[str] = None
    degree:      Optional[str] = None
    field:       Optional[str] = None
    year:        Optional[str] = None
    gpa:         Optional[str] = None


class ResumeSchema(BaseModel):
    """Structured extraction schema for resumes and CVs."""
    full_name:       ExtractedField       = Field(default_factory=ExtractedField)
    email:           ExtractedField       = Field(default_factory=ExtractedField)
    phone:           ExtractedField       = Field(default_factory=ExtractedField)
    location:        ExtractedField       = Field(default_factory=ExtractedField)
    linkedin:        ExtractedField       = Field(default_factory=ExtractedField)
    github:          ExtractedField       = Field(default_factory=ExtractedField)
    summary:         ExtractedField       = Field(default_factory=ExtractedField)
    skills:          List[str]            = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education:       List[Education]      = Field(default_factory=list)
    certifications:  List[str]            = Field(default_factory=list)
    languages:       List[str]            = Field(default_factory=list)
    total_experience_years: ExtractedField = Field(default_factory=ExtractedField)


# ── Medical Form ─────────────────────────────────────────────────────────

class MedicalFormSchema(BaseModel):
    """Structured extraction for medical intake forms and patient records."""
    patient_name:     ExtractedField = Field(default_factory=ExtractedField)
    date_of_birth:    ExtractedField = Field(default_factory=ExtractedField)
    patient_id:       ExtractedField = Field(default_factory=ExtractedField)
    gender:           ExtractedField = Field(default_factory=ExtractedField)
    visit_date:       ExtractedField = Field(default_factory=ExtractedField)
    chief_complaint:  ExtractedField = Field(default_factory=ExtractedField)
    diagnosis:        ExtractedField = Field(default_factory=ExtractedField)
    medications:      List[str]      = Field(default_factory=list)
    allergies:        List[str]      = Field(default_factory=list)
    blood_pressure:   ExtractedField = Field(default_factory=ExtractedField)
    heart_rate:       ExtractedField = Field(default_factory=ExtractedField)
    temperature:      ExtractedField = Field(default_factory=ExtractedField)
    physician_name:   ExtractedField = Field(default_factory=ExtractedField)
    insurance_id:     ExtractedField = Field(default_factory=ExtractedField)


# ── Bank Statement ────────────────────────────────────────────────────────

class Transaction(BaseModel):
    date:        Optional[str]   = None
    description: Optional[str]  = None
    debit:       Optional[float] = None
    credit:      Optional[float] = None
    balance:     Optional[float] = None


class BankStatementSchema(BaseModel):
    """Structured extraction for bank statements."""
    account_holder:    ExtractedField    = Field(default_factory=ExtractedField)
    account_number:    ExtractedField    = Field(default_factory=ExtractedField)
    bank_name:         ExtractedField    = Field(default_factory=ExtractedField)
    statement_period:  ExtractedField    = Field(default_factory=ExtractedField)
    opening_balance:   ExtractedField    = Field(default_factory=ExtractedField)
    closing_balance:   ExtractedField    = Field(default_factory=ExtractedField)
    total_credits:     ExtractedField    = Field(default_factory=ExtractedField)
    total_debits:      ExtractedField    = Field(default_factory=ExtractedField)
    transactions:      List[Transaction] = Field(default_factory=list)
    currency:          ExtractedField    = Field(default_factory=ExtractedField)


# ── Registry ──────────────────────────────────────────────────────────────

SCHEMA_REGISTRY: Dict[str, type] = {
    "invoice":       InvoiceSchema,
    "contract":      ContractSchema,
    "resume":        ResumeSchema,
    "medical_form":  MedicalFormSchema,
    "bank_statement": BankStatementSchema,
}

def get_schema(doc_type: str) -> type:
    if doc_type not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown document type: '{doc_type}'. Available: {list(SCHEMA_REGISTRY.keys())}")
    return SCHEMA_REGISTRY[doc_type]
