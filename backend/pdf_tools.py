"""
PDF Form-Filling Tools
Provides programmatic PDF form manipulation using pdfrw and reportlab.
"""

import os
import io
import json
import logging
import httpx
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Bankruptcy form download URLs (uscourts.gov)
BANKRUPTCY_FORM_URLS = {
    # Official Bankruptcy Forms (uscourts.gov verified URLs)
    "101": "https://www.uscourts.gov/sites/default/files/form_b101.pdf",
    "106A": "https://www.uscourts.gov/sites/default/files/form_b106ab.pdf",
    "106AB": "https://www.uscourts.gov/sites/default/files/form_b106ab.pdf",
    "106C": "https://www.uscourts.gov/sites/default/files/form_b106c.pdf",
    "106D": "https://www.uscourts.gov/sites/default/files/form_b106d.pdf",
    "106E": "https://www.uscourts.gov/sites/default/files/form_b106ef.pdf",
    "106EF": "https://www.uscourts.gov/sites/default/files/form_b106ef.pdf",
    "106G": "https://www.uscourts.gov/sites/default/files/form_b106g.pdf",
    "106H": "https://www.uscourts.gov/sites/default/files/form_b106h.pdf",
    "106I": "https://www.uscourts.gov/sites/default/files/form_b106i.pdf",
    "106J": "https://www.uscourts.gov/sites/default/files/form_b106j.pdf",
    "106SUM": "https://www.uscourts.gov/sites/default/files/form_b106sum.pdf",
    "106DEC": "https://www.uscourts.gov/sites/default/files/form_b106dec.pdf",
    "107": "https://www.uscourts.gov/sites/default/files/form_b107.pdf",
    "108": "https://www.uscourts.gov/sites/default/files/form_b108.pdf",
    "121": "https://www.uscourts.gov/sites/default/files/form_b121.pdf",
    "122A-1": "https://www.uscourts.gov/sites/default/files/form_b122a-1.pdf",
    "122A-2": "https://www.uscourts.gov/sites/default/files/form_b122a-2.pdf",
}

# Form instructions PDF
INSTRUCTIONS_URL = "https://www.uscourts.gov/sites/default/files/instructions_for_bankruptcy_forms_for_individuals.pdf"


class PDFFormReader:
    """Read form fields from PDF files."""

    @staticmethod
    def read_form_fields(pdf_path: str) -> Dict[str, Any]:
        """
        Read all form fields from a PDF.

        Returns:
            Dict with field names, types, and current values.
        """
        try:
            from pdfrw import PdfReader

            reader = PdfReader(pdf_path)
            fields = {}

            if reader.Root.AcroForm:
                form_fields = reader.Root.AcroForm.Fields or []
                for field in form_fields:
                    field_name = field.T if field.T else "unnamed"
                    # Clean the field name (remove parentheses wrapper)
                    if isinstance(field_name, str):
                        if field_name.startswith("(") and field_name.endswith(")"):
                            field_name = field_name[1:-1]
                    else:
                        field_name = str(field_name)

                    field_type = str(field.FT) if field.FT else "unknown"
                    current_value = str(field.V) if field.V else ""
                    if current_value.startswith("(") and current_value.endswith(")"):
                        current_value = current_value[1:-1]

                    fields[field_name] = {
                        "type": field_type,
                        "value": current_value,
                    }

            return {
                "success": True,
                "path": pdf_path,
                "fields": fields,
                "field_count": len(fields),
                "has_form": bool(reader.Root.AcroForm),
            }
        except ImportError:
            return {"success": False, "error": "pdfrw not installed. Run: pip install pdfrw"}
        except Exception as e:
            logger.error(f"Error reading PDF form fields: {e}")
            return {"success": False, "error": str(e)}


class PDFFormFiller:
    """Fill PDF forms programmatically."""

    @staticmethod
    def fill_form(
        input_path: str,
        output_path: str,
        field_values: Dict[str, str],
        flatten: bool = False,
    ) -> Dict[str, Any]:
        """
        Fill a PDF form with provided values.

        Args:
            input_path: Path to input PDF with form fields
            output_path: Path to save filled PDF
            field_values: Dict mapping field names to values
            flatten: If True, flatten form (make fields non-editable)

        Returns:
            Dict with success status and details.
        """
        try:
            from pdfrw import PdfReader, PdfWriter

            reader = PdfReader(input_path)

            if not reader.Root.AcroForm:
                return {"success": False, "error": "PDF has no form fields"}

            # Set NeedAppearances flag so PDF viewers regenerate field visuals
            from pdfrw import PdfName, PdfObject
            reader.Root.AcroForm.NeedAppearances = PdfObject('true')

            filled_count = 0
            for field in reader.Root.AcroForm.Fields or []:
                field_name = field.T if field.T else ""
                if isinstance(field_name, str):
                    if field_name.startswith("(") and field_name.endswith(")"):
                        field_name = field_name[1:-1]
                else:
                    field_name = str(field_name)

                if field_name in field_values:
                    # Set the value
                    field.V = f"({field_values[field_name]})"
                    # Clear appearance to force regeneration
                    if hasattr(field, 'AP'):
                        field.AP = None
                    filled_count += 1

            # Write output
            writer = PdfWriter(output_path)
            writer.trailer = reader
            writer.write()

            return {
                "success": True,
                "input_path": input_path,
                "output_path": output_path,
                "fields_filled": filled_count,
                "message": f"Filled {filled_count} fields, saved to {output_path}",
            }
        except ImportError:
            return {"success": False, "error": "pdfrw not installed. Run: pip install pdfrw"}
        except Exception as e:
            logger.error(f"Error filling PDF form: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def create_text_overlay(
        input_path: str,
        output_path: str,
        text_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Add text overlays to a PDF (for forms without fillable fields).
        Uses reportlab for text generation.

        Args:
            input_path: Path to input PDF
            output_path: Path to save output PDF
            text_positions: List of dicts with 'text', 'x', 'y', 'page', 'font_size'

        Returns:
            Dict with success status and details.
        """
        try:
            from pdfrw import PdfReader, PdfWriter, PageMerge
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter

            reader = PdfReader(input_path)

            # Group text by page
            text_by_page = {}
            for item in text_positions:
                page_num = item.get("page", 1) - 1  # 0-indexed
                if page_num not in text_by_page:
                    text_by_page[page_num] = []
                text_by_page[page_num].append(item)

            # Process each page
            for page_num, page in enumerate(reader.pages):
                if page_num in text_by_page:
                    # Create overlay with reportlab
                    packet = io.BytesIO()
                    c = canvas.Canvas(packet, pagesize=letter)

                    for item in text_by_page[page_num]:
                        font_size = item.get("font_size", 12)
                        c.setFont("Helvetica", font_size)
                        c.drawString(item["x"], item["y"], str(item["text"]))

                    c.save()
                    packet.seek(0)

                    # Merge overlay with page
                    overlay = PdfReader(packet)
                    if overlay.pages:
                        merger = PageMerge(page)
                        merger.add(overlay.pages[0]).render()

            # Write output
            writer = PdfWriter(output_path)
            writer.addpages(reader.pages)
            writer.write()

            return {
                "success": True,
                "input_path": input_path,
                "output_path": output_path,
                "pages_modified": len(text_by_page),
                "text_items_added": sum(len(v) for v in text_by_page.values()),
            }
        except ImportError as e:
            missing = "pdfrw" if "pdfrw" in str(e) else "reportlab"
            return {"success": False, "error": f"{missing} not installed. Run: pip install {missing}"}
        except Exception as e:
            logger.error(f"Error creating PDF overlay: {e}")
            return {"success": False, "error": str(e)}


class PDFDownloader:
    """Download official bankruptcy forms from uscourts.gov."""

    @staticmethod
    async def download_form(
        form_id: str,
        output_dir: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Download a bankruptcy form by ID.

        Args:
            form_id: Form identifier (e.g., "101", "106A", "122A-1")
            output_dir: Directory to save the downloaded form
            overwrite: If True, overwrite existing file

        Returns:
            Dict with success status and file path.
        """
        form_id_upper = form_id.upper().replace(" ", "").replace("_", "")

        url = BANKRUPTCY_FORM_URLS.get(form_id_upper)
        if not url:
            # Try variations
            for key in BANKRUPTCY_FORM_URLS:
                if key.replace("-", "").replace("_", "") == form_id_upper:
                    url = BANKRUPTCY_FORM_URLS[key]
                    form_id_upper = key
                    break

        if not url:
            return {
                "success": False,
                "error": f"Unknown form ID: {form_id}. Available: {list(BANKRUPTCY_FORM_URLS.keys())}",
            }

        output_path = Path(output_dir) / f"form_{form_id_upper}.pdf"

        if output_path.exists() and not overwrite:
            return {
                "success": True,
                "path": str(output_path),
                "message": f"Form already exists at {output_path}",
                "downloaded": False,
            }

        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Ensure directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Write file
                output_path.write_bytes(response.content)

                return {
                    "success": True,
                    "path": str(output_path),
                    "form_id": form_id_upper,
                    "url": url,
                    "size": len(response.content),
                    "message": f"Downloaded {form_id_upper} to {output_path}",
                    "downloaded": True,
                }
        except httpx.HTTPError as e:
            return {"success": False, "error": f"HTTP error downloading form: {e}"}
        except Exception as e:
            logger.error(f"Error downloading form {form_id}: {e}")
            return {"success": False, "error": str(e)}

    @staticmethod
    async def download_instructions(output_dir: str) -> Dict[str, Any]:
        """Download the official bankruptcy form instructions PDF."""
        output_path = Path(output_dir) / "instructions_for_bankruptcy_forms.pdf"

        if output_path.exists():
            return {
                "success": True,
                "path": str(output_path),
                "message": "Instructions already downloaded",
                "downloaded": False,
            }

        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.get(INSTRUCTIONS_URL)
                response.raise_for_status()

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(response.content)

                return {
                    "success": True,
                    "path": str(output_path),
                    "url": INSTRUCTIONS_URL,
                    "size": len(response.content),
                    "message": f"Downloaded instructions to {output_path}",
                    "downloaded": True,
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    async def ensure_forms_available(
        form_ids: List[str],
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Ensure specified forms are downloaded and available.

        Args:
            form_ids: List of form IDs to check/download
            output_dir: Directory for forms

        Returns:
            Dict with results for each form.
        """
        results = {}
        downloaded = []
        already_present = []
        failed = []

        for form_id in form_ids:
            result = await PDFDownloader.download_form(form_id, output_dir)
            results[form_id] = result

            if result["success"]:
                if result.get("downloaded"):
                    downloaded.append(form_id)
                else:
                    already_present.append(form_id)
            else:
                failed.append(form_id)

        return {
            "success": len(failed) == 0,
            "results": results,
            "downloaded": downloaded,
            "already_present": already_present,
            "failed": failed,
            "summary": f"Downloaded: {len(downloaded)}, Already present: {len(already_present)}, Failed: {len(failed)}",
        }


class PDFVerifier:
    """Verify PDF file integrity and content."""

    @staticmethod
    def verify_pdf(pdf_path: str) -> Dict[str, Any]:
        """
        Verify a PDF file is valid and readable.

        Returns:
            Dict with verification results.
        """
        try:
            from pdfrw import PdfReader

            path = Path(pdf_path)
            if not path.exists():
                return {"success": False, "valid": False, "error": "File not found"}

            if not path.suffix.lower() == ".pdf":
                return {"success": False, "valid": False, "error": "Not a PDF file"}

            reader = PdfReader(pdf_path)

            return {
                "success": True,
                "valid": True,
                "path": pdf_path,
                "page_count": len(reader.pages),
                "has_form": bool(reader.Root.AcroForm),
                "file_size": path.stat().st_size,
            }
        except Exception as e:
            return {"success": False, "valid": False, "error": str(e)}

    @staticmethod
    def verify_filled_form(
        original_path: str,
        filled_path: str,
        expected_values: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Verify that a filled form contains the expected values.

        Args:
            original_path: Path to original unfilled PDF
            filled_path: Path to filled PDF
            expected_values: Dict of field names to expected values

        Returns:
            Dict with verification results.
        """
        try:
            # Read the filled form
            filled_result = PDFFormReader.read_form_fields(filled_path)
            if not filled_result["success"]:
                return filled_result

            filled_fields = filled_result["fields"]

            # Check each expected value
            matches = []
            mismatches = []
            missing = []

            for field_name, expected_value in expected_values.items():
                if field_name not in filled_fields:
                    missing.append(field_name)
                else:
                    actual_value = filled_fields[field_name].get("value", "")
                    # Strip pdfrw parentheses wrapper for comparison
                    clean_actual = actual_value
                    if clean_actual.startswith("\\(") and clean_actual.endswith("\\)"):
                        clean_actual = clean_actual[2:-2]
                    elif clean_actual.startswith("(") and clean_actual.endswith(")"):
                        clean_actual = clean_actual[1:-1]
                    if clean_actual == expected_value or actual_value == expected_value:
                        matches.append(field_name)
                    else:
                        mismatches.append({
                            "field": field_name,
                            "expected": expected_value,
                            "actual": actual_value,
                        })

            all_match = len(mismatches) == 0 and len(missing) == 0

            return {
                "success": True,
                "verified": all_match,
                "matches": len(matches),
                "mismatches": mismatches,
                "missing_fields": missing,
                "total_checked": len(expected_values),
                "message": "All values match" if all_match else f"{len(mismatches)} mismatches, {len(missing)} missing",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# Tool definitions for LLM prompting
PDF_TOOL_DEFINITIONS = """
PDF Form Tools (for programmatic PDF manipulation):

1. read_pdf_fields - Read all fillable field names from a PDF form
   Usage: <tool>read_pdf_fields</tool><path>PDF_PATH</path>

2. fill_pdf_form - Fill a PDF form with values and save to new file
   Usage: <tool>fill_pdf_form</tool><input>INPUT_PDF</input><output>OUTPUT_PDF</output><fields>{"field_name": "value", ...}</fields>

3. add_pdf_text - Add text overlays at specific coordinates (for non-fillable PDFs)
   Usage: <tool>add_pdf_text</tool><input>INPUT_PDF</input><output>OUTPUT_PDF</output><positions>[{"text": "...", "x": 100, "y": 700, "page": 1}, ...]</positions>

4. download_form - Download official bankruptcy form by ID (e.g., "101", "106A", "122A-1")
   Usage: <tool>download_form</tool><form_id>FORM_ID</form_id><output_dir>DIRECTORY</output_dir>

5. verify_pdf - Check if a PDF is valid and readable
   Usage: <tool>verify_pdf</tool><path>PDF_PATH</path>

Use these tools when the user wants to programmatically work with PDF forms.
"""


# Convenience functions for tool execution
def execute_pdf_tool(tool_call: Dict[str, Any], allowed_paths: List[str]) -> Dict[str, Any]:
    """Execute a PDF-related tool call."""
    import asyncio

    tool_name = tool_call.get("tool")

    if tool_name == "read_pdf_fields":
        path = tool_call.get("path")
        return PDFFormReader.read_form_fields(path)

    elif tool_name == "fill_pdf_form":
        input_path = tool_call.get("input")
        output_path = tool_call.get("output")
        fields = tool_call.get("fields", {})
        if isinstance(fields, str):
            fields = json.loads(fields)
        return PDFFormFiller.fill_form(input_path, output_path, fields)

    elif tool_name == "add_pdf_text":
        input_path = tool_call.get("input")
        output_path = tool_call.get("output")
        positions = tool_call.get("positions", [])
        if isinstance(positions, str):
            positions = json.loads(positions)
        return PDFFormFiller.create_text_overlay(input_path, output_path, positions)

    elif tool_name == "download_form":
        form_id = tool_call.get("form_id")
        output_dir = tool_call.get("output_dir")
        # Run async function in sync context
        return asyncio.get_event_loop().run_until_complete(
            PDFDownloader.download_form(form_id, output_dir)
        )

    elif tool_name == "verify_pdf":
        path = tool_call.get("path")
        return PDFVerifier.verify_pdf(path)

    else:
        return {"success": False, "error": f"Unknown PDF tool: {tool_name}"}
