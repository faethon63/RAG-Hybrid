"""
Data Profile Module
Structured storage for extracted personal/financial data with source citations.
Every field carries provenance: source document, page, line, confidence, and verification status.

Used by the Chapter 7 bankruptcy form-filling system to ensure data integrity.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_project_kb_path

logger = logging.getLogger(__name__)


class DataField:
    """A single data point with full provenance tracking."""

    def __init__(
        self,
        value: Any = None,
        source: str = "",
        page: Optional[int] = None,
        line: str = "",
        confidence: float = 0.0,
        verified_by: Optional[str] = None,
        extracted_at: Optional[str] = None,
        notes: str = "",
    ):
        self.value = value
        self.source = source  # exact filename
        self.page = page  # page number in source
        self.line = line  # form line / field label
        self.confidence = confidence  # 0.0 - 1.0
        self.verified_by = verified_by  # null until user confirms
        self.extracted_at = extracted_at or datetime.now().isoformat()
        self.notes = notes

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "page": self.page,
            "line": self.line,
            "confidence": self.confidence,
            "verified_by": self.verified_by,
            "extracted_at": self.extracted_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataField":
        return cls(
            value=data.get("value"),
            source=data.get("source", ""),
            page=data.get("page"),
            line=data.get("line", ""),
            confidence=data.get("confidence", 0.0),
            verified_by=data.get("verified_by"),
            extracted_at=data.get("extracted_at"),
            notes=data.get("notes", ""),
        )

    @property
    def is_verified(self) -> bool:
        return self.verified_by is not None

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.8

    def __repr__(self):
        status = "verified" if self.is_verified else f"conf={self.confidence:.0%}"
        return f"DataField({self.value!r}, source={self.source!r}, {status})"


# Sensitive fields that should NEVER be auto-filled without user input at fill time
SENSITIVE_FIELDS = {
    "personal_info.ssn",
    "personal_info.spouse_ssn",
    "personal_info.bank_account_numbers",
    "personal_info.credit_card_numbers",
}


class DataProfile:
    """
    Structured profile of all extractable data for bankruptcy form filling.
    Organized by sections matching bankruptcy form categories.
    """

    SECTIONS = [
        "personal_info",
        "income",
        "expenses",
        "assets",
        "debts",
        "means_test",
        "tax_data",
        "tax_data_2025",
        "bank_data",
        "credit_counseling",
        "computed_totals",
    ]

    def __init__(self, project_name: str = "Chapter_7_Assistant"):
        self.project_name = project_name
        self._data: Dict[str, Dict[str, DataField]] = {s: {} for s in self.SECTIONS}
        self._metadata = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "extraction_runs": [],
        }

    @property
    def _profile_path(self) -> Path:
        base = Path(get_project_kb_path()) / self.project_name
        base.mkdir(parents=True, exist_ok=True)
        return base / "data_profile.json"

    @staticmethod
    def _flatten_dict(d: dict, prefix: str = "") -> dict:
        """Recursively flatten nested dicts into dotted field names.
        e.g. {"a": {"b": 1}} -> {"a.b": 1}
        """
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(DataProfile._flatten_dict(v, key))
            else:
                result[key] = v
        return result

    def set_field(self, section: str, field_name: str, field: DataField) -> None:
        """Set a field value with provenance."""
        if section not in self.SECTIONS:
            raise ValueError(f"Unknown section: {section}. Valid: {self.SECTIONS}")
        self._data[section][field_name] = field
        self._metadata["updated_at"] = datetime.now().isoformat()

    def get_field(self, section: str, field_name: str) -> Optional[DataField]:
        """Get a field by section and name."""
        return self._data.get(section, {}).get(field_name)

    def get_field_by_path(self, path: str) -> Optional[DataField]:
        """Get a field by dot-separated path like 'personal_info.full_name'."""
        parts = path.split(".", 1)
        if len(parts) != 2:
            return None
        return self.get_field(parts[0], parts[1])

    def get_all_fields(self) -> Dict[str, Dict[str, DataField]]:
        """Get all fields organized by section."""
        return self._data

    def get_section(self, section: str) -> Dict[str, DataField]:
        """Get all fields in a section."""
        return self._data.get(section, {})

    def get_unverified_fields(self) -> List[tuple]:
        """Return list of (section, field_name, DataField) for unverified fields."""
        results = []
        for section, fields in self._data.items():
            for name, field in fields.items():
                if not field.is_verified:
                    results.append((section, name, field))
        return results

    def get_low_confidence_fields(self, threshold: float = 0.8) -> List[tuple]:
        """Return fields with confidence below threshold."""
        results = []
        for section, fields in self._data.items():
            for name, field in fields.items():
                if field.confidence < threshold:
                    results.append((section, name, field))
        return results

    def get_sensitive_fields(self) -> List[tuple]:
        """Return fields marked as sensitive (never auto-fill)."""
        results = []
        for section, fields in self._data.items():
            for name, field in fields.items():
                path = f"{section}.{name}"
                if path in SENSITIVE_FIELDS:
                    results.append((section, name, field))
        return results

    def get_field_count(self) -> Dict[str, int]:
        """Return count of fields per section."""
        return {s: len(f) for s, f in self._data.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the profile status."""
        total = sum(len(f) for f in self._data.values())
        verified = sum(1 for s, n, f in self.get_unverified_fields())
        low_conf = len(self.get_low_confidence_fields())
        return {
            "total_fields": total,
            "unverified": total - (total - verified),
            "verified": total - verified,
            "low_confidence": low_conf,
            "sections": self.get_field_count(),
            "updated_at": self._metadata.get("updated_at"),
        }

    def record_extraction_run(self, document: str, extractor: str, fields_extracted: int):
        """Record an extraction run for audit trail."""
        self._metadata["extraction_runs"].append({
            "document": document,
            "extractor": extractor,
            "fields_extracted": fields_extracted,
            "timestamp": datetime.now().isoformat(),
        })

    def save(self) -> Path:
        """Save profile to disk."""
        data = {
            "metadata": self._metadata,
            "sections": {},
        }
        for section, fields in self._data.items():
            data["sections"][section] = {
                name: field.to_dict() for name, field in fields.items()
            }

        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._profile_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data profile to {self._profile_path}")
        return self._profile_path

    def load(self) -> bool:
        """Load profile from disk. Returns True if loaded successfully."""
        if not self._profile_path.exists():
            logger.info(f"No existing profile at {self._profile_path}")
            return False

        try:
            with open(self._profile_path) as f:
                data = json.load(f)

            self._metadata = data.get("metadata", self._metadata)

            if "sections" in data:
                # DataField format: {"sections": {"personal_info": {"full_name": {"value": ..., "confidence": ...}}}}
                for section, fields in data["sections"].items():
                    if section in self.SECTIONS:
                        self._data[section] = {
                            name: DataField.from_dict(field_data)
                            for name, field_data in fields.items()
                        }
            else:
                # Flat format: {"personal_info": {"full_name": "GEORGE..."}, "tax_data_2025": {...}}
                # Flatten nested dicts and wrap plain values into DataField objects
                source = "data_profile"
                if self._metadata.get("documents_used"):
                    source = ", ".join(self._metadata["documents_used"][:3])
                for section in self.SECTIONS:
                    if section in data:
                        flat = self._flatten_dict(data[section])
                        self._data[section] = {
                            name: DataField(value=val, source=source, confidence=0.9)
                            for name, val in flat.items()
                        }

            total = sum(len(f) for f in self._data.values())
            logger.info(f"Loaded data profile from {self._profile_path} ({total} fields)")
            return True
        except Exception as e:
            logger.error(f"Failed to load data profile: {e}")
            return False

    def to_dict(self) -> dict:
        """Serialize entire profile for API responses."""
        result = {"metadata": self._metadata, "sections": {}}
        for section, fields in self._data.items():
            result["sections"][section] = {
                name: field.to_dict() for name, field in fields.items()
            }
        return result

    def verify_field(self, section: str, field_name: str, verified_by: str = "user") -> bool:
        """Mark a field as verified by user."""
        field = self.get_field(section, field_name)
        if field:
            field.verified_by = verified_by
            field.confidence = 1.0
            self._metadata["updated_at"] = datetime.now().isoformat()
            return True
        return False

    def update_field_value(self, section: str, field_name: str, new_value: Any, verified_by: str = "user") -> bool:
        """Update a field value (user correction) and mark as verified."""
        field = self.get_field(section, field_name)
        if field:
            field.value = new_value
            field.verified_by = verified_by
            field.confidence = 1.0
            field.notes = f"User-corrected from {field.value}"
            self._metadata["updated_at"] = datetime.now().isoformat()
            return True
        return False
