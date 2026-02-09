"""
Form Filler Engine
Pipeline for safe, verified bankruptcy form filling.

Pipeline (NEVER skips steps):
1. Load form mapping + data profile
2. For each field: propose value with confidence, flag low-confidence or missing
3. Present fill plan to user (table format) - REQUIRED before any filling
4. User reviews/approves/corrects
5. Fill PDF via PDFFormFiller
6. Read back filled PDF via PDFFormReader
7. Compare filled vs expected (PDFVerifier)
8. Generate audit log JSON
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_project_kb_path, get_anthropic_api_key, get_claude_opus_model
from data_profile import DataProfile, DataField, SENSITIVE_FIELDS
from pdf_tools import PDFFormReader, PDFFormFiller, PDFVerifier

logger = logging.getLogger(__name__)


class FormMapping:
    """Loads and manages form field mappings from JSON files."""

    MAPPINGS_DIR = Path(__file__).parent / "form_mappings"

    @classmethod
    def load(cls, form_id: str) -> Optional[Dict[str, Any]]:
        """Load a form mapping by form ID."""
        # Try exact match first, then variations
        # Strip common "B" prefix (e.g. "B122A-1" -> "122A-1")
        stripped = form_id.lstrip("Bb")
        normalized = stripped.lower().replace("-", "_")
        candidates = [
            cls.MAPPINGS_DIR / f"form_{form_id}.json",
            cls.MAPPINGS_DIR / f"form_{form_id.lower()}.json",
            cls.MAPPINGS_DIR / f"form_{form_id.replace('-', '_')}.json",
            cls.MAPPINGS_DIR / f"form_{normalized}.json",
            cls.MAPPINGS_DIR / f"form_{stripped}.json",
            cls.MAPPINGS_DIR / f"form_{stripped.lower()}.json",
        ]

        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load form mapping {path}: {e}")
                    return None

        logger.warning(f"No form mapping found for form {form_id}")
        return None

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available form mappings."""
        if not cls.MAPPINGS_DIR.exists():
            return []
        return [
            p.stem.replace("form_", "")
            for p in cls.MAPPINGS_DIR.glob("form_*.json")
        ]


class FillPlan:
    """Represents a proposed form fill plan for user review."""

    def __init__(self, form_id: str, form_name: str):
        self.form_id = form_id
        self.form_name = form_name
        self.fields: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()

    def add_field(
        self,
        pdf_field: str,
        human_label: str,
        proposed_value: Any,
        source: str,
        confidence: float,
        status: str,  # "auto", "verify", "needs_input", "sensitive"
        notes: str = "",
    ):
        self.fields.append({
            "pdf_field": pdf_field,
            "human_label": human_label,
            "proposed_value": proposed_value,
            "source": source,
            "confidence": confidence,
            "status": status,
            "notes": notes,
            "user_approved": None,
            "user_value": None,
        })

    def to_markdown_table(self) -> str:
        """Generate a markdown table for chat display."""
        lines = [
            f"## Fill Plan: Form {self.form_id} - {self.form_name}",
            "",
            "| # | Field | Proposed Value | Source | Confidence | Status |",
            "|---|-------|---------------|--------|------------|--------|",
        ]

        for i, f in enumerate(self.fields, 1):
            value = f["proposed_value"] if f["proposed_value"] is not None else "---"
            if f["status"] == "sensitive":
                value = "[USER INPUT REQUIRED]"
            elif f["status"] == "verify":
                value = f"VERIFY: {value}"
            elif f["status"] == "needs_input":
                value = "NEEDS INPUT"

            conf = f"{f['confidence']:.0%}" if f["confidence"] > 0 else "---"
            source = f["source"][:30] + "..." if len(f["source"]) > 30 else f["source"]

            lines.append(f"| {i} | {f['human_label']} | {value} | {source} | {conf} | {f['status'].upper()} |")

        # Summary
        auto = sum(1 for f in self.fields if f["status"] == "auto")
        verify = sum(1 for f in self.fields if f["status"] == "verify")
        needs = sum(1 for f in self.fields if f["status"] == "needs_input")
        sensitive = sum(1 for f in self.fields if f["status"] == "sensitive")

        lines.extend([
            "",
            f"**Summary:** {auto} auto-fill, {verify} need verification, {needs} need input, {sensitive} sensitive (manual only)",
            "",
            "To approve: reply with 'approve' or provide corrections for specific fields.",
        ])

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "form_id": self.form_id,
            "form_name": self.form_name,
            "fields": self.fields,
            "created_at": self.created_at,
        }


class FormFillerEngine:
    """
    Orchestrates the complete form-filling pipeline with verification.
    """

    def __init__(self, project_name: str = "Chapter_7_Assistant"):
        self.project_name = project_name
        self.profile = DataProfile(project_name)
        self.profile.load()

    async def generate_fill_plan(self, form_id: str) -> Dict[str, Any]:
        """
        Step 1: Generate a fill plan for a form.
        Maps profile data to form fields and scores confidence.
        Returns the plan for user review - MUST be approved before filling.
        """
        mapping = FormMapping.load(form_id)
        if not mapping:
            available = FormMapping.list_available()
            return {
                "success": False,
                "error": f"No mapping found for form {form_id}. Available: {available}",
            }

        plan = FillPlan(form_id, mapping.get("form_name", f"Form {form_id}"))

        for field_def in mapping.get("fields", []):
            pdf_field = field_def["pdf_field"]
            human_label = field_def["human_label"]
            profile_path = field_def.get("profile_path")
            default_value = field_def.get("default_value")
            is_sensitive = field_def.get("sensitive", False)
            is_required = field_def.get("required", False)

            # Check if this is a sensitive field
            if is_sensitive or (profile_path and profile_path in SENSITIVE_FIELDS):
                plan.add_field(
                    pdf_field=pdf_field,
                    human_label=human_label,
                    proposed_value=None,
                    source="",
                    confidence=0.0,
                    status="sensitive",
                    notes="Sensitive field - user must provide value at fill time",
                )
                continue

            # Try to get value from profile
            proposed_value = None
            source = ""
            confidence = 0.0

            if profile_path:
                data_field = self.profile.get_field_by_path(profile_path)
                if data_field and data_field.value is not None:
                    proposed_value = data_field.value
                    source = data_field.source
                    confidence = data_field.confidence

            # Fall back to default value
            if proposed_value is None and default_value is not None:
                proposed_value = default_value
                source = "default"
                confidence = 1.0

            # Determine status
            if proposed_value is None:
                status = "needs_input"
            elif confidence >= 0.8:
                status = "auto"
            else:
                status = "verify"

            plan.add_field(
                pdf_field=pdf_field,
                human_label=human_label,
                proposed_value=proposed_value,
                source=source,
                confidence=confidence,
                status=status,
            )

        return {
            "success": True,
            "plan": plan.to_dict(),
            "markdown": plan.to_markdown_table(),
        }

    async def opus_audit_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send the complete fill plan to Opus for independent audit review.
        Opus reviews every proposed value against sources and flags concerns.
        """
        import httpx

        api_key = get_anthropic_api_key()
        if not api_key or api_key.startswith("your_"):
            return {"success": False, "error": "Claude API key not configured"}

        opus_model = get_claude_opus_model()
        profile_data = self.profile.to_dict()

        audit_prompt = f"""You are an independent auditor reviewing a bankruptcy form fill plan.

FORM: {plan.get('form_id')} - {plan.get('form_name')}

DATA PROFILE (all extracted values with sources):
{json.dumps(profile_data, indent=2)[:6000]}

PROPOSED FILL PLAN:
{json.dumps(plan.get('fields', []), indent=2)}

YOUR TASK:
1. Verify each proposed value against its cited source in the profile
2. Check arithmetic (any totals, sums, or calculated values)
3. Flag values that seem unreasonable
4. Identify required fields marked as "needs_input" that SHOULD have been extractable
5. Check for internal consistency

Return a JSON object:
{{
  "approved": true/false,
  "concerns": [
    {{"field": "...", "issue": "...", "severity": "error|warning|info"}}
  ],
  "arithmetic_checks": [
    {{"calculation": "...", "expected": "...", "actual": "...", "pass": true/false}}
  ],
  "missing_but_extractable": ["field names that should have been found"],
  "summary": "one paragraph overall assessment"
}}

CRITICAL: If ANY field has severity "error", set approved to false."""

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
                        "model": opus_model,
                        "max_tokens": 4000,
                        "messages": [{"role": "user", "content": audit_prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                text = ""
                if data.get("content"):
                    text = data["content"][0].get("text", "")

                # Parse JSON response
                text = text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                audit_result = json.loads(text)
                audit_result["model"] = opus_model
                audit_result["usage"] = data.get("usage", {})
                return {"success": True, "audit": audit_result}

        except json.JSONDecodeError:
            return {"success": True, "audit": {"approved": False, "summary": text[:500], "concerns": []}}
        except Exception as e:
            logger.error(f"Opus audit failed: {e}")
            return {"success": False, "error": str(e)}

    async def execute_fill(
        self,
        form_id: str,
        approved_plan: Dict[str, Any],
        input_pdf_path: str,
        output_pdf_path: str,
    ) -> Dict[str, Any]:
        """
        Step 3: Execute the fill after plan approval.
        Fills the PDF and runs post-fill verification.
        """
        fields_to_fill = {}
        audit_entries = []

        for field in approved_plan.get("fields", []):
            # Use user-corrected value if provided, otherwise proposed value
            value = field.get("user_value") or field.get("proposed_value")

            if value is None:
                continue

            if field.get("status") == "sensitive" and not field.get("user_value"):
                # Sensitive field without user input - skip
                continue

            pdf_field = field["pdf_field"]
            fields_to_fill[pdf_field] = str(value)

            audit_entries.append({
                "pdf_field": pdf_field,
                "human_label": field["human_label"],
                "value_filled": str(value),
                "source": field.get("source", ""),
                "confidence": field.get("confidence", 0),
                "user_approved": field.get("user_approved", False),
                "was_user_corrected": field.get("user_value") is not None,
            })

        if not fields_to_fill:
            return {"success": False, "error": "No fields to fill"}

        # Fill the PDF
        fill_result = PDFFormFiller.fill_form(
            input_path=input_pdf_path,
            output_path=output_pdf_path,
            field_values=fields_to_fill,
        )

        if not fill_result.get("success"):
            return fill_result

        # Post-fill verification: read back and compare
        verify_result = PDFVerifier.verify_filled_form(
            original_path=input_pdf_path,
            filled_path=output_pdf_path,
            expected_values=fields_to_fill,
        )

        # Generate audit log
        audit_log = {
            "form_id": form_id,
            "timestamp": datetime.now().isoformat(),
            "input_pdf": input_pdf_path,
            "output_pdf": output_pdf_path,
            "fields_filled": len(fields_to_fill),
            "fill_result": fill_result,
            "verification": verify_result,
            "field_audit": audit_entries,
        }

        # Save audit log
        audit_dir = Path(get_project_kb_path()) / self.project_name / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"form_{form_id}_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(audit_path, "w") as f:
            json.dump(audit_log, f, indent=2)

        return {
            "success": True,
            "output_pdf": output_pdf_path,
            "fields_filled": fill_result.get("fields_filled", 0),
            "verification": verify_result,
            "audit_log": str(audit_path),
            "message": f"Form {form_id} filled with {len(fields_to_fill)} fields. Audit log saved to {audit_path}",
        }


async def handle_fill_form_tool(
    action: str,
    form_id: str,
    input_pdf: str = "",
    output_pdf: str = "",
    field_corrections: Optional[Dict[str, str]] = None,
    project_name: str = "Chapter_7_Assistant",
) -> Dict[str, Any]:
    """
    Groq tool handler for fill_bankruptcy_form.

    Actions:
    - "plan": Generate fill plan for review (REQUIRED first step)
    - "audit": Send plan to Opus for independent review
    - "fill": Execute the fill (requires approved plan)
    """
    engine = FormFillerEngine(project_name)

    if action == "plan":
        return await engine.generate_fill_plan(form_id)

    elif action == "audit":
        # Generate plan first, then audit it
        plan_result = await engine.generate_fill_plan(form_id)
        if not plan_result.get("success"):
            return plan_result
        return await engine.opus_audit_plan(plan_result["plan"])

    elif action == "fill":
        # Auto-resolve paths if not provided
        if not input_pdf or not output_pdf:
            from pdf_tools import PDFDownloader, BANKRUPTCY_FORM_URLS

            # Map our internal form IDs to download keys
            FORM_ID_TO_DOWNLOAD_KEY = {
                "101": "101", "122a_1": "122A-1", "122a_2": "122A-2",
                "106ab": "106AB", "106c": "106C", "106d": "106D",
                "106ef": "106EF", "106g": "106G", "106h": "106H",
                "106i": "106I", "106j": "106J", "106sum": "106SUM",
                "106dec": "106DEC", "107": "107", "108": "108", "121": "121",
            }

            # Normalize form_id for lookup
            normalized = form_id.lstrip("Bb").lower().replace("-", "_")
            download_key = FORM_ID_TO_DOWNLOAD_KEY.get(normalized, form_id.upper())

            # Map form IDs to known local filenames in KB documents
            FORM_ID_TO_LOCAL_FILENAME = {
                "101": "Bankruptcy initial form_b_101_0624_fillable_clean.pdf",
                "106ab": "form_b106ab Assets.pdf",
                "106c": "form_b_106c.pdf",
                "106d": "form_b106d.pdf",
                "106ef": "form_b106ef.pdf",
                "106g": "form_b106g.pdf",
                "106h": "B 106H Schedule H.pdf",
                "106i": "form_b106i_Schedule_I.pdf",
                "106j": "form_b106j.pdf",
                "106dec": "form_b106dec.pdf",
                "106sum": "form_b106sum.pdf",
                "107": "form_b107.pdf",
                "108": "form_b108.pdf",
                "119": "form_b119.pdf",
                "121": "form_b121.pdf",
                "122a_1": "form_b122a-1.pdf",
                "122a_2": "form_b122a-2.pdf",
            }

            # Directories
            from config import get_synced_kb_path
            kb_path = Path(get_project_kb_path()) / project_name
            docs_dir = Path(get_synced_kb_path()) / project_name / "documents"
            blank_dir = kb_path / "blank_forms"
            filled_dir = kb_path / "filled_forms"
            blank_dir.mkdir(parents=True, exist_ok=True)
            filled_dir.mkdir(parents=True, exist_ok=True)

            if not input_pdf:
                # 1. Try local KB documents folder first
                local_filename = FORM_ID_TO_LOCAL_FILENAME.get(normalized)
                if local_filename and (docs_dir / local_filename).exists():
                    input_pdf = str(docs_dir / local_filename)
                    logger.info(f"Using local blank form: {input_pdf}")
                else:
                    # 2. Try blank_forms cache
                    blank_path = blank_dir / f"form_{download_key}.pdf"
                    if blank_path.exists():
                        input_pdf = str(blank_path)
                    else:
                        # 3. Download from uscourts.gov as last resort
                        dl_result = await PDFDownloader.download_form(download_key, str(blank_dir))
                        if not dl_result.get("success"):
                            return {"success": False, "error": f"Could not get blank form: {dl_result.get('error', 'download failed')}"}
                        input_pdf = dl_result.get("path", str(blank_path))

            if not output_pdf:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_pdf = str(filled_dir / f"form_{normalized}_filled_{timestamp}.pdf")

        # Generate plan (with any corrections applied)
        plan_result = await engine.generate_fill_plan(form_id)
        if not plan_result.get("success"):
            return plan_result

        plan = plan_result["plan"]

        # Apply field corrections if provided
        if field_corrections:
            for field in plan.get("fields", []):
                if field["pdf_field"] in field_corrections:
                    field["user_value"] = field_corrections[field["pdf_field"]]
                    field["user_approved"] = True

        return await engine.execute_fill(form_id, plan, input_pdf, output_pdf)

    else:
        return {"success": False, "error": f"Unknown action: {action}. Use 'plan', 'audit', or 'fill'."}
