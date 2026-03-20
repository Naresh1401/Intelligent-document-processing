"""
src/extraction/llm_extractor.py
================================
Core LLM-based field extraction engine.

Takes a document text + schema → returns structured JSON with
per-field confidence scores.

Key design decisions:
  1. Schema-driven prompting: inject field descriptions into the prompt
     so the model knows exactly what to look for
  2. Confidence per field: model rates its own certainty per field
  3. Raw text capture: store the exact text the value was extracted from
     for auditability and human review
  4. JSON repair: LLM sometimes returns malformed JSON — we fix it
"""

import json
import re
from typing import Dict, Any, Optional, Type
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from pydantic import BaseModel

from src.processing.schemas import ExtractedField, get_schema


EXTRACTION_SYSTEM_PROMPT = """You are a precise document data extraction engine.

Your task is to extract structured information from documents.

Rules:
1. Extract ONLY information explicitly present in the document
2. For each field, provide a confidence score (0.0-1.0):
   - 1.0 = explicitly stated, no ambiguity
   - 0.7 = clearly implied or derived
   - 0.4 = uncertain, may be partially correct
   - 0.0 = not found in document
3. For missing fields, set value to null and confidence to 0.0
4. Never invent or hallucinate values
5. Return ONLY valid JSON, no explanations"""


def build_extraction_prompt(text: str, schema_class: Type[BaseModel], doc_type: str) -> str:
    """Build a structured extraction prompt from the document text and schema."""

    # Get field descriptions from schema
    field_descriptions = []
    for field_name, field_info in schema_class.model_fields.items():
        desc = field_info.description or field_name.replace("_", " ").title()
        field_descriptions.append(f"  - {field_name}: {desc}")

    fields_str = "\n".join(field_descriptions)

    return f"""Extract information from this {doc_type.replace('_', ' ')}.

Fields to extract:
{fields_str}

For each field, return:
{{
  "field_name": {{
    "value": <extracted value or null>,
    "confidence": <0.0-1.0>,
    "raw_text": "<exact text from document this was extracted from>"
  }}
}}

For list fields (line_items, transactions, etc.), return an array of objects.

Document text:
---
{text[:4000]}
---

Return ONLY valid JSON:"""


class LLMExtractor:
    """
    Extracts structured fields from documents using LLM.

    Supports OpenAI and Anthropic models.
    Includes JSON repair for malformed LLM responses.
    """

    def __init__(self, llm):
        self.llm = llm

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def extract(
        self,
        text: str,
        doc_type: str,
        schema_class: Optional[Type[BaseModel]] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured data from document text.

        Args:
            text:         Raw document text
            doc_type:     Document type key (invoice, contract, resume, etc.)
            schema_class: Optional override for the schema class

        Returns:
            Dict with extracted fields and confidence scores
        """
        if schema_class is None:
            schema_class = get_schema(doc_type)

        prompt = build_extraction_prompt(text, schema_class, doc_type)

        raw_response = self.llm.generate(EXTRACTION_SYSTEM_PROMPT, prompt)
        logger.debug(f"LLM raw response length: {len(raw_response)} chars")

        # Parse and repair JSON
        extracted = self._parse_response(raw_response)

        # Validate and hydrate into schema
        result = self._hydrate_schema(extracted, schema_class)

        return result

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response, attempting JSON repair if needed."""
        # Strip markdown code fences
        clean = re.sub(r"```(?:json)?|```", "", response).strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Attempt to extract JSON object
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Attempt to fix common issues: trailing commas, single quotes
        repaired = clean
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)
        repaired = repaired.replace("'", '"')

        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON repair failed: {e}. Returning empty dict.")
            return {}

    def _hydrate_schema(
        self,
        raw: Dict[str, Any],
        schema_class: Type[BaseModel],
    ) -> Dict[str, Any]:
        """
        Convert raw LLM output to schema-validated structure.

        Handles both flat field dicts and nested {value, confidence} dicts.
        """
        result = {}
        schema_fields = schema_class.model_fields

        for field_name in schema_fields:
            field_data = raw.get(field_name)

            if field_data is None:
                result[field_name] = ExtractedField(value=None, confidence=0.0)
                continue

            # Handle list fields (line_items, transactions, skills, etc.)
            if isinstance(field_data, list):
                result[field_name] = field_data
                continue

            # Handle {value, confidence, raw_text} dict
            if isinstance(field_data, dict) and "value" in field_data:
                result[field_name] = ExtractedField(
                    value      = field_data.get("value"),
                    confidence = float(field_data.get("confidence", 0.5)),
                    raw_text   = field_data.get("raw_text"),
                )
            else:
                # Bare value (no confidence wrapper)
                result[field_name] = ExtractedField(
                    value      = field_data,
                    confidence = 0.7,  # Default confidence for bare values
                )

        return result


class OpenAILLM:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2048):
        import os
        from openai import OpenAI
        self.client     = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model      = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model       = self.model,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            messages    = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content.strip()


class AnthropicLLM:
    def __init__(self, model: str = "claude-3-5-sonnet-20241022",
                 api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 2048):
        import os, anthropic
        self.client     = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model      = model
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def generate(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model      = self.model,
            max_tokens = self.max_tokens,
            system     = system,
            messages   = [{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()
