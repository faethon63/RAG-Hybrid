"""
Filing Tracker Export
Generates a CSV/JSON view of all data items needed for Chapter 7 filing.
Reads all 16 form mappings + data profile to show what's collected vs missing.
"""

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_profile import DataProfile

logger = logging.getLogger(__name__)

FORM_MAPPINGS_DIR = Path(__file__).parent / "form_mappings"

# Map sections to human-readable categories
SECTION_CATEGORIES = {
    "personal_info": "Personal Information",
    "income": "Income",
    "expenses": "Monthly Expenses",
    "assets": "Assets & Property",
    "debts": "Debts",
    "means_test": "Means Test",
    "tax_data": "Tax Data",
    "tax_data_2025": "Tax Data (2025)",
    "bank_data": "Bank Accounts",
    "credit_counseling": "Credit Counseling",
    "computed_totals": "Computed Totals",
    "creditors_secured": "Secured Creditors",
    "creditors_priority": "Priority Unsecured Creditors",
    "creditors_nonpriority": "Nonpriority Unsecured Creditors",
    "exemptions": "Exemptions",
    "contracts_leases": "Contracts & Leases",
    "codebtors": "Codebtors",
    "financial_history": "Financial History",
    "filing_decisions": "Filing Decisions",
    "tracker": "Tracker",
}

# Priority levels based on form requirements
PRIORITY_REQUIRED = "Critical"
PRIORITY_OPTIONAL = "Optional"
PRIORITY_COMPUTED = "Computed"  # auto-calculated from other fields


def load_all_form_mappings() -> Dict[str, dict]:
    """Load all form mapping JSON files. Returns {form_id: mapping_data}."""
    mappings = {}
    for f in sorted(FORM_MAPPINGS_DIR.glob("form_*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
                form_id = data.get("form_id", f.stem.replace("form_", ""))
                mappings[form_id] = data
        except Exception as e:
            logger.warning(f"Failed to load form mapping {f}: {e}")
    return mappings


def _get_profile_value(profile: DataProfile, profile_path: str) -> Tuple[Optional[Any], Optional[str], bool]:
    """
    Look up a value in the data profile by dotted path.
    Returns (value, source, is_verified).
    """
    if not profile_path:
        return None, None, False

    field = profile.get_field_by_path(profile_path)
    if field and field.value is not None:
        return field.value, field.source, field.is_verified
    return None, None, False


def build_tracker_items(project_name: str = "Chapter_7_Assistant") -> List[Dict[str, str]]:
    """
    Build the full list of tracker items by scanning all form mappings.
    Deduplicates fields that appear in multiple forms (same profile_path or human_label).

    Returns list of dicts with keys:
        category, data_item, forms, current_value, source, verified, priority, profile_path, sensitive
    """
    mappings = load_all_form_mappings()
    profile = DataProfile(project_name)
    profile.load()

    # Deduplicate by profile_path (if set) or by human_label
    # Track which forms need each item
    seen = {}  # key -> tracker_item dict
    form_lists = {}  # key -> set of form_ids

    for form_id, mapping in mappings.items():
        form_name = mapping.get("form_name", form_id)
        form_short = form_id.upper()

        for field in mapping.get("fields", []):
            human_label = field.get("human_label", field.get("pdf_field", "Unknown"))
            profile_path = field.get("profile_path")
            is_required = field.get("required", False)
            is_sensitive = field.get("sensitive", False)
            default_value = field.get("default_value")
            notes = field.get("notes", "")

            # Determine dedup key
            if profile_path:
                dedup_key = profile_path
            else:
                # Normalize label for dedup (e.g. "Debtor 1 name" appears in every form)
                dedup_key = f"label:{human_label.lower().strip()}"

            # Look up current value
            value, source, verified = _get_profile_value(profile, profile_path) if profile_path else (None, None, False)

            # If no profile value but has default, use that
            if value is None and default_value is not None:
                value = default_value
                source = "default"
                verified = False

            # Determine category from profile_path section
            if profile_path:
                section = profile_path.split(".")[0]
                category = SECTION_CATEGORIES.get(section, section.replace("_", " ").title())
            else:
                # Infer category from form type
                form_to_category = {
                    "101": "Personal Information",
                    "106ab": "Assets & Property",
                    "106c": "Exemptions",
                    "106d": "Secured Creditors",
                    "106ef": "Unsecured Creditors",
                    "106g": "Contracts & Leases",
                    "106h": "Codebtors",
                    "106i": "Income",
                    "106j": "Monthly Expenses",
                    "106sum": "Summary",
                    "106dec": "Declarations",
                    "107": "Financial History",
                    "108": "Statement of Intention",
                    "121": "Personal Information",
                    "122a_1": "Means Test",
                    "122a_2": "Means Test",
                }
                category = form_to_category.get(form_id, "Other")

            # Determine priority
            if notes and ("sum of" in notes.lower() or "copy from" in notes.lower()):
                priority = PRIORITY_COMPUTED
            elif is_required:
                priority = PRIORITY_REQUIRED
            else:
                priority = PRIORITY_OPTIONAL

            if dedup_key in seen:
                # Just add form to the list
                form_lists[dedup_key].add(form_short)
                # Upgrade priority if any form marks it required
                if priority == PRIORITY_REQUIRED and seen[dedup_key]["priority"] != PRIORITY_REQUIRED:
                    seen[dedup_key]["priority"] = PRIORITY_REQUIRED
            else:
                seen[dedup_key] = {
                    "category": category,
                    "data_item": human_label,
                    "forms": "",  # filled in later
                    "current_value": str(value) if value is not None else "(missing)",
                    "source": source or "",
                    "verified": "Yes" if verified else ("Needs Verify" if value is not None else ""),
                    "priority": priority,
                    "profile_path": profile_path or "",
                    "sensitive": "Yes" if is_sensitive else "",
                    "notes": notes,
                }
                form_lists[dedup_key] = {form_short}

    # Fill in form lists
    for key, item in seen.items():
        item["forms"] = ", ".join(sorted(form_lists[key]))

    # Sort by: priority (Critical first), then category, then data_item
    priority_order = {PRIORITY_REQUIRED: 0, PRIORITY_OPTIONAL: 1, PRIORITY_COMPUTED: 2}
    items = sorted(
        seen.values(),
        key=lambda x: (priority_order.get(x["priority"], 9), x["category"], x["data_item"])
    )

    return items


def generate_tracker_csv(project_name: str = "Chapter_7_Assistant") -> str:
    """Generate a CSV string of all tracker items."""
    items = build_tracker_items(project_name)

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["category", "data_item", "forms", "current_value", "source", "verified", "priority", "profile_path", "sensitive", "notes"],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(items)

    return output.getvalue()


def get_tracker_status(project_name: str = "Chapter_7_Assistant") -> Dict[str, Any]:
    """
    Return completion statistics for the filing tracker.
    Used by the /tracker/status API endpoint and the "what's missing" chat interceptor.
    """
    items = build_tracker_items(project_name)

    total = len(items)
    filled = sum(1 for i in items if i["current_value"] != "(missing)")
    missing = total - filled
    verified = sum(1 for i in items if i["verified"] == "Yes")
    needs_verify = sum(1 for i in items if i["verified"] == "Needs Verify")

    # Critical items
    critical_items = [i for i in items if i["priority"] == PRIORITY_REQUIRED]
    critical_total = len(critical_items)
    critical_filled = sum(1 for i in critical_items if i["current_value"] != "(missing)")
    critical_missing = critical_total - critical_filled

    # By-category breakdown
    categories = {}
    for item in items:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "filled": 0, "missing": 0, "missing_items": []}
        categories[cat]["total"] += 1
        if item["current_value"] != "(missing)":
            categories[cat]["filled"] += 1
        else:
            categories[cat]["missing"] += 1
            categories[cat]["missing_items"].append(item["data_item"])

    # Overall completion percentage
    pct = round(filled / total * 100, 1) if total > 0 else 0

    return {
        "total_fields": total,
        "filled": filled,
        "missing": missing,
        "verified": verified,
        "needs_verify": needs_verify,
        "completion_pct": pct,
        "critical": {
            "total": critical_total,
            "filled": critical_filled,
            "missing": critical_missing,
        },
        "by_category": categories,
    }


def format_missing_summary(project_name: str = "Chapter_7_Assistant") -> str:
    """
    Format a human-readable summary of what's missing, for chat display.
    Used by the data collection interceptor in groq_agent.py.
    """
    status = get_tracker_status(project_name)

    lines = []
    lines.append(f"## Filing Completion: {status['completion_pct']}%")
    lines.append(f"**{status['filled']}** of **{status['total_fields']}** data items collected")
    lines.append(f"- Verified: {status['verified']}")
    lines.append(f"- Needs verification: {status['needs_verify']}")
    lines.append(f"- Missing: {status['missing']}")
    lines.append("")

    if status["critical"]["missing"] > 0:
        lines.append(f"### Critical Missing Items ({status['critical']['missing']})")
        lines.append("")

    for cat, info in sorted(status["by_category"].items()):
        if info["missing"] > 0:
            pct = round(info["filled"] / info["total"] * 100) if info["total"] > 0 else 0
            lines.append(f"**{cat}** ({pct}% complete — {info['missing']} missing)")
            for item_name in info["missing_items"][:5]:  # Show up to 5 per category
                lines.append(f"  - {item_name}")
            if len(info["missing_items"]) > 5:
                lines.append(f"  - ... and {len(info['missing_items']) - 5} more")
            lines.append("")

    lines.append("---")
    lines.append("Say **\"let's collect [category]\"** to start gathering data for a specific section.")
    lines.append("Or provide data directly: *\"my rent is 1850, electric 120\"*")

    return "\n".join(lines)
