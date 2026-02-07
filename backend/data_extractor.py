"""
Data Extractor Module
Extracts structured financial data from source documents (tax returns, bank statements)
with dual-model verification for bankruptcy form filling.

Model Strategy:
- Pass 1 (Sonnet 4.5): Mechanical OCR/extraction
- Pass 2 (Opus 4.6): Independent verification with max reasoning
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from config import get_anthropic_api_key, get_claude_sonnet_model, get_claude_opus_model
from data_profile import DataField, DataProfile

logger = logging.getLogger(__name__)


async def _call_claude_vision(
    image_or_pdf_path: str,
    prompt: str,
    model: str = None,
    max_tokens: int = 4000,
) -> Dict[str, Any]:
    """
    Send a document (image or PDF) to Claude with a vision prompt.
    For PDFs, sends each page as an image (via base64 document).
    Returns: {"text": str, "usage": dict}
    """
    api_key = get_anthropic_api_key()
    if not api_key or api_key.startswith("your_"):
        return {"text": "", "usage": {}, "error": "Claude API key not configured"}

    if model is None:
        model = get_claude_sonnet_model()

    path = Path(image_or_pdf_path)
    suffix = path.suffix.lower()

    # Build content blocks
    content_blocks = []

    if suffix == ".pdf":
        # Send PDF as base64 document
        pdf_data = path.read_bytes()
        b64_data = base64.b64encode(pdf_data).decode("utf-8")
        content_blocks.append({
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64_data,
            }
        })
    elif suffix in (".png", ".jpg", ".jpeg"):
        image_data = path.read_bytes()
        b64_data = base64.b64encode(image_data).decode("utf-8")
        media_type = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.lstrip("."), "image/png")
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            }
        })
    else:
        # Plain text file - read and include as text
        text_content = path.read_text(encoding="utf-8", errors="ignore")
        content_blocks.append({"type": "text", "text": f"Document content:\n{text_content}"})

    content_blocks.append({"type": "text", "text": prompt})

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": content_blocks}],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = ""
            if data.get("content"):
                text = data["content"][0].get("text", "")
            usage = data.get("usage", {})
            return {
                "text": text,
                "usage": {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
            }
    except Exception as e:
        logger.error(f"Claude vision call failed: {e}")
        return {"text": "", "usage": {}, "error": str(e)}


# ======================================================================
# Extraction Prompts
# ======================================================================

TAX_RETURN_EXTRACTION_PROMPT = """You are a precise data extraction system. Extract ALL of the following fields from this tax return document.
Return the data as a JSON object. For each field, provide:
- "value": the exact value as shown on the form
- "page": the page number where you found it
- "line": the form line number or field label

If a field is not present, set value to null.

Required fields:
{
  "filing_status": "Single/MFJ/MFS/HOH/QW",
  "tax_year": "YYYY",
  "taxpayer_name": "full name as shown",
  "taxpayer_ssn_last4": "last 4 digits only (for verification, NOT full SSN)",
  "address": "full address as shown",
  "city_state_zip": "city, state, ZIP",
  "wages_salaries_tips": "Line 1 amount",
  "interest_income": "Line 2b amount",
  "dividend_income": "Line 3b amount",
  "business_income_loss": "Schedule C net profit/loss",
  "capital_gains_losses": "Line 7 amount",
  "other_income": "Line 8 amount",
  "total_income": "Line 9 amount",
  "adjusted_gross_income": "Line 11 amount",
  "standard_deduction": "Line 12 amount",
  "taxable_income": "Line 15 amount",
  "total_tax": "Line 24 amount",
  "total_payments": "Line 33 amount",
  "refund_amount": "Line 35a amount",
  "amount_owed": "Line 37 amount",
  "self_employment_tax": "Schedule SE amount if present",
  "self_employment_income": "Schedule SE net earnings if present",
  "schedule_c_gross_receipts": "Schedule C Line 1",
  "schedule_c_net_profit": "Schedule C Line 31",
  "schedule_c_business_name": "Schedule C business name",
  "schedule_c_business_code": "Schedule C business activity code"
}

CRITICAL RULES:
1. Extract EXACT values - do not round or estimate
2. Include cents (e.g., "$45,231.00" not "$45,231")
3. If a value is negative, preserve the negative sign or parentheses
4. For SSN, extract ONLY the last 4 digits for verification
5. If you cannot read a value clearly, set value to null and add a "note" field explaining why
6. Return ONLY valid JSON, no commentary before or after"""

BANK_STATEMENT_EXTRACTION_PROMPT = """You are a precise data extraction system. Extract the following data from this bank statement.
Return the data as a JSON object.

Required fields:
{
  "bank_name": "name of the bank",
  "account_type": "checking/savings/business checking/etc",
  "account_holder": "name on the account",
  "statement_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "beginning_balance": "dollar amount",
  "ending_balance": "dollar amount",
  "total_deposits": "total deposits for period",
  "total_withdrawals": "total withdrawals for period",
  "deposit_count": "number of deposits",
  "withdrawal_count": "number of withdrawals",
  "largest_deposit": {"amount": "$X", "date": "YYYY-MM-DD", "description": "..."},
  "largest_withdrawal": {"amount": "$X", "date": "YYYY-MM-DD", "description": "..."},
  "recurring_deposits": [{"description": "...", "amount": "$X", "frequency": "monthly/biweekly/etc"}],
  "recurring_withdrawals": [{"description": "...", "amount": "$X", "frequency": "monthly/biweekly/etc"}]
}

CRITICAL RULES:
1. Extract EXACT dollar amounts including cents
2. Dates must be in YYYY-MM-DD format
3. If a value is not clearly visible, set to null
4. Return ONLY valid JSON"""


class TaxReturnExtractor:
    """Extracts structured data from tax return PDFs using Claude vision."""

    async def extract(self, pdf_path: str, model: str = None) -> Dict[str, Any]:
        """
        Extract tax return data from a PDF.
        Returns dict with extracted fields and metadata.
        """
        if model is None:
            model = get_claude_sonnet_model()

        logger.info(f"Extracting tax return data from {pdf_path} using {model}")
        result = await _call_claude_vision(pdf_path, TAX_RETURN_EXTRACTION_PROMPT, model=model)

        if result.get("error"):
            return {"success": False, "error": result["error"], "fields": {}}

        # Parse the JSON response
        try:
            text = result["text"].strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            fields = json.loads(text)
            return {
                "success": True,
                "fields": fields,
                "usage": result.get("usage", {}),
                "model": model,
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tax return extraction: {e}")
            return {
                "success": False,
                "error": f"JSON parse error: {e}",
                "raw_text": result["text"][:2000],
                "fields": {},
            }


class BankStatementExtractor:
    """Extracts structured data from bank statements."""

    async def extract(self, file_path: str, model: str = None) -> Dict[str, Any]:
        """Extract bank statement data. Tries pypdf first, falls back to Claude vision."""
        path = Path(file_path)

        # Try pypdf text extraction first (faster, free)
        if path.suffix.lower() == ".pdf":
            text = self._try_pypdf(path)
            if text and len(text) > 200:
                logger.info(f"pypdf extracted {len(text)} chars from {path.name}")
                # Send the text to Claude for structured extraction (no vision needed)
                return await self._extract_from_text(text, path.name, model)

        # Fallback: Claude vision (handles scanned PDFs, images)
        logger.info(f"Using Claude vision for bank statement: {path.name}")
        if model is None:
            model = get_claude_sonnet_model()

        result = await _call_claude_vision(file_path, BANK_STATEMENT_EXTRACTION_PROMPT, model=model)

        if result.get("error"):
            return {"success": False, "error": result["error"], "fields": {}}

        try:
            text = result["text"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            fields = json.loads(text)
            return {"success": True, "fields": fields, "usage": result.get("usage", {}), "model": model}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON parse error: {e}", "fields": {}}

    def _try_pypdf(self, path: Path) -> str:
        """Try extracting text with pypdf."""
        try:
            import pypdf
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                parts = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
            return "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"pypdf extraction failed for {path}: {e}")
            return ""

    async def _extract_from_text(self, text: str, filename: str, model: str = None) -> Dict[str, Any]:
        """Extract structured data from already-extracted text."""
        if model is None:
            model = get_claude_sonnet_model()

        prompt = f"Document: {filename}\n\nText content:\n{text[:8000]}\n\n{BANK_STATEMENT_EXTRACTION_PROMPT}"

        api_key = get_anthropic_api_key()
        if not api_key or api_key.startswith("your_"):
            return {"success": False, "error": "Claude API key not configured", "fields": {}}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": model,
                        "max_tokens": 4000,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text_resp = ""
                if data.get("content"):
                    text_resp = data["content"][0].get("text", "")

                # Parse JSON
                text_resp = text_resp.strip()
                if text_resp.startswith("```"):
                    text_resp = text_resp.split("\n", 1)[1] if "\n" in text_resp else text_resp[3:]
                    if text_resp.endswith("```"):
                        text_resp = text_resp[:-3]
                    text_resp = text_resp.strip()

                fields = json.loads(text_resp)
                return {"success": True, "fields": fields, "usage": data.get("usage", {}), "model": model}
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {"success": False, "error": str(e), "fields": {}}


class DualExtractionVerifier:
    """
    Two-model verification: extracts data twice with different models
    and compares results. Discrepancies are flagged for human review.

    Pass 1: Claude Sonnet (mechanical extraction)
    Pass 2: Claude Opus (independent verification with max reasoning)
    """

    def __init__(self):
        self.tax_extractor = TaxReturnExtractor()
        self.bank_extractor = BankStatementExtractor()

    async def verify_tax_return(self, pdf_path: str) -> Dict[str, Any]:
        """Dual-extract tax return and compare results."""
        sonnet = get_claude_sonnet_model()
        opus = get_claude_opus_model()

        logger.info(f"Dual extraction: tax return {pdf_path}")

        # Pass 1: Sonnet
        result1 = await self.tax_extractor.extract(pdf_path, model=sonnet)
        if not result1.get("success"):
            return {"success": False, "error": f"Pass 1 (Sonnet) failed: {result1.get('error')}", "fields": {}}

        # Pass 2: Opus
        result2 = await self.tax_extractor.extract(pdf_path, model=opus)
        if not result2.get("success"):
            # Opus failed but Sonnet succeeded - use Sonnet with lower confidence
            logger.warning("Opus verification failed, using Sonnet-only results with lower confidence")
            return {
                "success": True,
                "fields": result1["fields"],
                "single_pass": True,
                "confidence_penalty": 0.2,
                "usage": {
                    "pass1": result1.get("usage", {}),
                    "pass2_error": result2.get("error"),
                },
            }

        # Compare results
        comparison = self._compare_extractions(result1["fields"], result2["fields"])

        return {
            "success": True,
            "fields": comparison["merged_fields"],
            "matches": comparison["matches"],
            "discrepancies": comparison["discrepancies"],
            "single_pass": False,
            "usage": {
                "pass1": result1.get("usage", {}),
                "pass2": result2.get("usage", {}),
            },
        }

    async def verify_bank_statement(self, file_path: str) -> Dict[str, Any]:
        """Dual-extract bank statement and compare results."""
        sonnet = get_claude_sonnet_model()
        opus = get_claude_opus_model()

        logger.info(f"Dual extraction: bank statement {file_path}")

        result1 = await self.bank_extractor.extract(file_path, model=sonnet)
        if not result1.get("success"):
            return {"success": False, "error": f"Pass 1 failed: {result1.get('error')}", "fields": {}}

        result2 = await self.bank_extractor.extract(file_path, model=opus)
        if not result2.get("success"):
            logger.warning("Opus verification failed, using Sonnet-only results")
            return {
                "success": True,
                "fields": result1["fields"],
                "single_pass": True,
                "confidence_penalty": 0.2,
                "usage": {"pass1": result1.get("usage", {})},
            }

        comparison = self._compare_extractions(result1["fields"], result2["fields"])
        return {
            "success": True,
            "fields": comparison["merged_fields"],
            "matches": comparison["matches"],
            "discrepancies": comparison["discrepancies"],
            "single_pass": False,
            "usage": {
                "pass1": result1.get("usage", {}),
                "pass2": result2.get("usage", {}),
            },
        }

    def _compare_extractions(
        self, fields1: Dict[str, Any], fields2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two extraction results field by field.
        Returns merged fields with confidence scores.
        """
        all_keys = set(list(fields1.keys()) + list(fields2.keys()))
        matches = []
        discrepancies = []
        merged = {}

        for key in all_keys:
            v1 = fields1.get(key)
            v2 = fields2.get(key)

            if v1 is None and v2 is None:
                continue

            # Normalize for comparison (strip whitespace, standardize format)
            v1_norm = self._normalize_value(v1)
            v2_norm = self._normalize_value(v2)

            if v1_norm == v2_norm:
                # Both models agree
                matches.append(key)
                merged[key] = {"value": v1, "confidence": 0.95}
            elif v1 is None:
                # Only Opus found it
                merged[key] = {"value": v2, "confidence": 0.7}
                discrepancies.append({"field": key, "sonnet": None, "opus": v2, "reason": "Only found by Opus"})
            elif v2 is None:
                # Only Sonnet found it
                merged[key] = {"value": v1, "confidence": 0.7}
                discrepancies.append({"field": key, "sonnet": v1, "opus": None, "reason": "Only found by Sonnet"})
            else:
                # Both found different values - flag for review
                merged[key] = {"value": v1, "confidence": 0.5, "alternate": v2}
                discrepancies.append({"field": key, "sonnet": v1, "opus": v2, "reason": "Models disagree"})

        return {
            "merged_fields": merged,
            "matches": matches,
            "discrepancies": discrepancies,
        }

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        """Normalize a value for comparison."""
        if value is None:
            return None
        if isinstance(value, str):
            # Strip whitespace, remove $ and commas for numeric comparison
            v = value.strip()
            # Try to normalize dollar amounts
            cleaned = v.replace("$", "").replace(",", "").strip()
            try:
                return float(cleaned)
            except (ValueError, TypeError):
                return v.lower()
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True)
        return value


async def build_data_profile(
    project_name: str,
    document_paths: List[str],
    use_dual_verification: bool = True,
) -> Dict[str, Any]:
    """
    Build or update a DataProfile by extracting data from provided documents.

    Args:
        project_name: Project to store profile in
        document_paths: List of file paths to extract from
        use_dual_verification: If True, use both Sonnet and Opus for verification

    Returns:
        Dict with profile summary and any discrepancies found.
    """
    profile = DataProfile(project_name)
    profile.load()  # Load existing if any

    verifier = DualExtractionVerifier()
    all_discrepancies = []
    total_fields = 0
    documents_processed = []

    for doc_path in document_paths:
        path = Path(doc_path)
        if not path.exists():
            logger.warning(f"Document not found: {doc_path}")
            continue

        filename = path.name.lower()

        # Determine document type and extract
        if "tax" in filename or "federal_return" in filename or "federal tax" in filename:
            logger.info(f"Processing tax return: {path.name}")
            if use_dual_verification:
                result = await verifier.verify_tax_return(doc_path)
            else:
                result = await TaxReturnExtractor().extract(doc_path)

            if result.get("success"):
                fields = result.get("fields", {})
                confidence_penalty = result.get("confidence_penalty", 0.0)

                # Map tax fields to profile
                _map_tax_to_profile(profile, fields, path.name, confidence_penalty)
                total_fields += len([v for v in fields.values() if v is not None])

                if result.get("discrepancies"):
                    all_discrepancies.extend(result["discrepancies"])

                profile.record_extraction_run(path.name, "TaxReturnExtractor", len(fields))
                documents_processed.append(path.name)

        elif "bank" in filename or "statement" in filename:
            logger.info(f"Processing bank statement: {path.name}")
            if use_dual_verification:
                result = await verifier.verify_bank_statement(doc_path)
            else:
                result = await BankStatementExtractor().extract(doc_path)

            if result.get("success"):
                fields = result.get("fields", {})
                confidence_penalty = result.get("confidence_penalty", 0.0)

                _map_bank_to_profile(profile, fields, path.name, confidence_penalty)
                total_fields += len([v for v in fields.values() if v is not None])

                if result.get("discrepancies"):
                    all_discrepancies.extend(result["discrepancies"])

                profile.record_extraction_run(path.name, "BankStatementExtractor", len(fields))
                documents_processed.append(path.name)
        else:
            logger.info(f"Skipping unrecognized document type: {path.name}")

    # Save the updated profile
    profile.save()

    return {
        "success": True,
        "documents_processed": documents_processed,
        "total_fields_extracted": total_fields,
        "discrepancies": all_discrepancies,
        "profile_summary": profile.get_summary(),
    }


def _map_tax_to_profile(profile: DataProfile, fields: dict, source: str, confidence_penalty: float = 0.0):
    """Map extracted tax return fields to DataProfile sections."""
    def _conf(field_data):
        """Get confidence from dual-extraction merged fields or default."""
        if isinstance(field_data, dict) and "confidence" in field_data:
            return max(0.0, field_data["confidence"] - confidence_penalty)
        return max(0.0, 0.85 - confidence_penalty)

    def _val(field_data):
        """Get value from dual-extraction merged fields or raw value."""
        if isinstance(field_data, dict) and "value" in field_data:
            return field_data["value"]
        return field_data

    # Personal info
    for field_name, profile_field, section in [
        ("taxpayer_name", "full_name", "personal_info"),
        ("address", "address", "personal_info"),
        ("city_state_zip", "city_state_zip", "personal_info"),
        ("filing_status", "filing_status", "personal_info"),
        ("taxpayer_ssn_last4", "ssn_last4", "personal_info"),
    ]:
        val = fields.get(field_name)
        if val is not None and _val(val) is not None:
            profile.set_field(section, profile_field, DataField(
                value=_val(val),
                source=source,
                line=field_name,
                confidence=_conf(val),
            ))

    # Income fields -> tax_data section
    income_fields = [
        ("wages_salaries_tips", "wages"),
        ("interest_income", "interest_income"),
        ("dividend_income", "dividend_income"),
        ("business_income_loss", "business_income"),
        ("capital_gains_losses", "capital_gains"),
        ("other_income", "other_income"),
        ("total_income", "total_income"),
        ("adjusted_gross_income", "agi"),
        ("taxable_income", "taxable_income"),
        ("total_tax", "total_tax"),
        ("refund_amount", "refund"),
        ("amount_owed", "amount_owed"),
        ("self_employment_tax", "se_tax"),
        ("self_employment_income", "se_income"),
        ("schedule_c_gross_receipts", "schedule_c_gross"),
        ("schedule_c_net_profit", "schedule_c_net"),
        ("schedule_c_business_name", "business_name"),
        ("schedule_c_business_code", "business_code"),
        ("tax_year", "tax_year"),
    ]

    for ext_name, prof_name in income_fields:
        val = fields.get(ext_name)
        if val is not None and _val(val) is not None:
            profile.set_field("tax_data", prof_name, DataField(
                value=_val(val),
                source=source,
                line=ext_name,
                confidence=_conf(val),
            ))

    # Also set key income fields in the income section for form filling
    agi = fields.get("adjusted_gross_income")
    if agi and _val(agi):
        profile.set_field("income", "annual_gross_income", DataField(
            value=_val(agi),
            source=source,
            line="adjusted_gross_income",
            confidence=_conf(agi),
        ))

    wages = fields.get("wages_salaries_tips")
    if wages and _val(wages):
        profile.set_field("income", "wages", DataField(
            value=_val(wages),
            source=source,
            line="wages_salaries_tips",
            confidence=_conf(wages),
        ))

    biz = fields.get("schedule_c_net_profit")
    if biz and _val(biz):
        profile.set_field("income", "self_employment_net", DataField(
            value=_val(biz),
            source=source,
            line="schedule_c_net_profit",
            confidence=_conf(biz),
        ))


def _map_bank_to_profile(profile: DataProfile, fields: dict, source: str, confidence_penalty: float = 0.0):
    """Map extracted bank statement fields to DataProfile sections."""
    def _conf(field_data):
        if isinstance(field_data, dict) and "confidence" in field_data:
            return max(0.0, field_data["confidence"] - confidence_penalty)
        return max(0.0, 0.85 - confidence_penalty)

    def _val(field_data):
        if isinstance(field_data, dict) and "value" in field_data:
            return field_data["value"]
        return field_data

    # Bank data section
    bank_fields = [
        ("bank_name", "bank_name"),
        ("account_type", "account_type"),
        ("account_holder", "account_holder"),
        ("statement_period", "statement_period"),
        ("beginning_balance", "beginning_balance"),
        ("ending_balance", "ending_balance"),
        ("total_deposits", "total_deposits"),
        ("total_withdrawals", "total_withdrawals"),
        ("deposit_count", "deposit_count"),
        ("withdrawal_count", "withdrawal_count"),
        ("recurring_deposits", "recurring_deposits"),
        ("recurring_withdrawals", "recurring_withdrawals"),
    ]

    for ext_name, prof_name in bank_fields:
        val = fields.get(ext_name)
        if val is not None and _val(val) is not None:
            # Use the source filename to create unique keys per statement
            key = f"{prof_name}_{source.replace(' ', '_').replace('.', '_')}"
            profile.set_field("bank_data", key, DataField(
                value=_val(val),
                source=source,
                line=ext_name,
                confidence=_conf(val),
            ))

    # Also aggregate deposits into income section for cross-reference
    deposits = fields.get("total_deposits")
    if deposits and _val(deposits):
        key = f"deposits_{source.replace(' ', '_').replace('.', '_')}"
        profile.set_field("income", key, DataField(
            value=_val(deposits),
            source=source,
            line="total_deposits",
            confidence=_conf(deposits),
            notes="Bank statement deposits (for cross-reference with reported income)",
        ))
