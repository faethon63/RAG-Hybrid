"""
Audit Rules Module
Cross-document consistency checks, arithmetic verification, and audit logging
for bankruptcy form filling.

Verification Layers:
1. Cross-Document Consistency - tax return income vs bank deposits vs Schedule I
2. Arithmetic Verification - CMI calculation, asset/debt totals, means test math
3. Post-Fill Read-Back - filled PDF re-read and compared to expected values
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import get_project_kb_path
from data_profile import DataProfile, DataField

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """
    Cross-document consistency rules.
    Checks that values from different sources agree within tolerance.
    """

    @staticmethod
    def check_income_consistency(profile: DataProfile) -> List[Dict[str, Any]]:
        """
        Compare income across sources:
        - Tax return annual income / 12 ≈ Schedule I monthly income (15% tolerance)
        - Bank deposits over 6 months ≈ reported 6-month income (20% tolerance)
        """
        issues = []

        # Get annual income from tax data
        agi_field = profile.get_field("tax_data", "agi")
        annual_income = _parse_dollar(agi_field.value) if agi_field else None

        # Get wages from tax data
        wages_field = profile.get_field("tax_data", "wages")
        wages = _parse_dollar(wages_field.value) if wages_field else None

        # Get self-employment income
        se_field = profile.get_field("tax_data", "schedule_c_net")
        se_income = _parse_dollar(se_field.value) if se_field else None

        # Get monthly income from income section (if Schedule I has been populated)
        monthly_field = profile.get_field("income", "monthly_gross")
        monthly_income = _parse_dollar(monthly_field.value) if monthly_field else None

        # Check 1: Annual income / 12 vs monthly income (15% tolerance for self-employment)
        if annual_income and monthly_income:
            expected_monthly = annual_income / 12
            tolerance = 0.15 if se_income else 0.10
            diff_pct = abs(expected_monthly - monthly_income) / max(expected_monthly, 1)

            if diff_pct > tolerance:
                issues.append({
                    "rule": "income_annual_vs_monthly",
                    "severity": "warning",
                    "message": f"Annual income ({_fmt(annual_income)}) / 12 = {_fmt(expected_monthly)}/mo, "
                               f"but Schedule I shows {_fmt(monthly_income)}/mo ({diff_pct:.0%} difference)",
                    "sources": [
                        agi_field.source if agi_field else "unknown",
                        monthly_field.source if monthly_field else "Schedule I",
                    ],
                    "tolerance": f"{tolerance:.0%}",
                })

        # Check 2: Bank deposits vs reported income
        # Aggregate deposits from bank_data section
        bank_deposits = {}
        for name, field in profile.get_section("bank_data").items():
            if "total_deposits" in name:
                amount = _parse_dollar(field.value)
                if amount:
                    bank_deposits[field.source] = amount

        if bank_deposits and annual_income:
            total_deposits = sum(bank_deposits.values())
            num_months = len(bank_deposits)
            if num_months > 0:
                annualized_deposits = (total_deposits / num_months) * 12
                tolerance = 0.20
                diff_pct = abs(annualized_deposits - annual_income) / max(annual_income, 1)

                if diff_pct > tolerance:
                    issues.append({
                        "rule": "income_vs_bank_deposits",
                        "severity": "warning",
                        "message": f"Bank deposits ({num_months} months) annualize to {_fmt(annualized_deposits)}, "
                                   f"but tax return shows AGI of {_fmt(annual_income)} ({diff_pct:.0%} difference)",
                        "sources": list(bank_deposits.keys()) + [agi_field.source if agi_field else "unknown"],
                        "tolerance": f"{tolerance:.0%}",
                        "notes": "Some variance is normal (transfers, loans, gifts are deposits but not income)",
                    })

        # Check 3: Wages + SE income should approximate total income
        if wages and se_income and annual_income:
            combined = wages + se_income
            diff = abs(combined - annual_income)
            # Allow for other income sources
            if diff > annual_income * 0.05 and combined > annual_income:
                issues.append({
                    "rule": "income_components_vs_total",
                    "severity": "info",
                    "message": f"Wages ({_fmt(wages)}) + SE income ({_fmt(se_income)}) = {_fmt(combined)}, "
                               f"but AGI is {_fmt(annual_income)}. Difference may be deductions or other adjustments.",
                    "sources": [
                        wages_field.source if wages_field else "unknown",
                        se_field.source if se_field else "unknown",
                    ],
                })

        return issues

    @staticmethod
    def check_asset_debt_totals(profile: DataProfile) -> List[Dict[str, Any]]:
        """Check that schedule subtotals sum to 106Sum totals."""
        issues = []

        # Check total assets
        total_assets_field = profile.get_field("assets", "total_estimated")
        if total_assets_field:
            total_assets = _parse_dollar(total_assets_field.value)
            # Sum individual asset categories
            asset_sum = 0
            for name, field in profile.get_section("assets").items():
                if name.startswith("schedule_") and name != "total_estimated":
                    val = _parse_dollar(field.value)
                    if val:
                        asset_sum += val

            if asset_sum > 0 and total_assets:
                diff_pct = abs(asset_sum - total_assets) / max(total_assets, 1)
                if diff_pct > 0.01:  # 1% tolerance for rounding
                    issues.append({
                        "rule": "asset_totals_mismatch",
                        "severity": "error",
                        "message": f"Asset schedule subtotals ({_fmt(asset_sum)}) don't match "
                                   f"total estimated assets ({_fmt(total_assets)})",
                    })

        # Check total debts
        total_debts_field = profile.get_field("debts", "total_estimated")
        if total_debts_field:
            total_debts = _parse_dollar(total_debts_field.value)
            debt_sum = 0
            for name, field in profile.get_section("debts").items():
                if name.startswith("schedule_") and name != "total_estimated":
                    val = _parse_dollar(field.value)
                    if val:
                        debt_sum += val

            if debt_sum > 0 and total_debts:
                diff_pct = abs(debt_sum - total_debts) / max(total_debts, 1)
                if diff_pct > 0.01:
                    issues.append({
                        "rule": "debt_totals_mismatch",
                        "severity": "error",
                        "message": f"Debt schedule subtotals ({_fmt(debt_sum)}) don't match "
                                   f"total estimated debts ({_fmt(total_debts)})",
                    })

        return issues


class ArithmeticVerifier:
    """Verify arithmetic in means test and income calculations."""

    @staticmethod
    def verify_cmi(profile: DataProfile) -> List[Dict[str, Any]]:
        """
        Verify Current Monthly Income (CMI) calculation.
        CMI = total income over 6 calendar months before filing / 6
        """
        issues = []

        cmi_field = profile.get_field("means_test", "current_monthly_income")
        if not cmi_field:
            return issues

        cmi = _parse_dollar(cmi_field.value)
        if not cmi:
            return issues

        # Try to verify from individual monthly incomes
        monthly_incomes = []
        for name, field in profile.get_section("means_test").items():
            if name.startswith("month_") and "income" in name:
                val = _parse_dollar(field.value)
                if val is not None:
                    monthly_incomes.append(val)

        if len(monthly_incomes) == 6:
            expected_cmi = sum(monthly_incomes) / 6
            diff = abs(expected_cmi - cmi)
            if diff > 1.0:  # Allow $1 rounding
                issues.append({
                    "rule": "cmi_calculation",
                    "severity": "error",
                    "message": f"CMI should be {_fmt(expected_cmi)} (sum of 6 months / 6), "
                               f"but form shows {_fmt(cmi)}. Difference: {_fmt(diff)}",
                    "formula": f"({' + '.join(_fmt(m) for m in monthly_incomes)}) / 6 = {_fmt(expected_cmi)}",
                })

        return issues

    @staticmethod
    def verify_means_test(profile: DataProfile) -> List[Dict[str, Any]]:
        """Verify means test arithmetic (Form 122A-1)."""
        issues = []

        # Annualized CMI
        cmi_field = profile.get_field("means_test", "current_monthly_income")
        annual_field = profile.get_field("means_test", "annualized_income")

        if cmi_field and annual_field:
            cmi = _parse_dollar(cmi_field.value)
            annual = _parse_dollar(annual_field.value)
            if cmi and annual:
                expected = cmi * 12
                if abs(expected - annual) > 1.0:
                    issues.append({
                        "rule": "annualized_income",
                        "severity": "error",
                        "message": f"Annualized income should be CMI ({_fmt(cmi)}) x 12 = {_fmt(expected)}, "
                                   f"but form shows {_fmt(annual)}",
                    })

        return issues


class AuditLogger:
    """Generates and stores audit logs for form fill operations."""

    def __init__(self, project_name: str = "Chapter_7_Assistant"):
        self.project_name = project_name
        self._audit_dir = Path(get_project_kb_path()) / project_name / "audit"
        self._audit_dir.mkdir(parents=True, exist_ok=True)

    def create_audit_report(
        self,
        form_id: str,
        fill_plan: Dict[str, Any],
        consistency_issues: List[Dict[str, Any]],
        arithmetic_issues: List[Dict[str, Any]],
        opus_audit: Optional[Dict[str, Any]] = None,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a comprehensive audit report."""
        report = {
            "audit_id": f"audit_{form_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "form_id": form_id,
            "timestamp": datetime.now().isoformat(),
            "fill_plan": fill_plan,
            "consistency_checks": {
                "issues": consistency_issues,
                "errors": [i for i in consistency_issues if i.get("severity") == "error"],
                "warnings": [i for i in consistency_issues if i.get("severity") == "warning"],
                "pass": all(i.get("severity") != "error" for i in consistency_issues),
            },
            "arithmetic_checks": {
                "issues": arithmetic_issues,
                "errors": [i for i in arithmetic_issues if i.get("severity") == "error"],
                "pass": all(i.get("severity") != "error" for i in arithmetic_issues),
            },
            "opus_audit": opus_audit,
            "post_fill_verification": verification_result,
            "overall_pass": (
                all(i.get("severity") != "error" for i in consistency_issues) and
                all(i.get("severity") != "error" for i in arithmetic_issues) and
                (opus_audit is None or opus_audit.get("approved", False)) and
                (verification_result is None or verification_result.get("verified", False))
            ),
        }

        # Save to disk
        path = self._audit_dir / f"{report['audit_id']}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Audit report saved to {path}")

        report["path"] = str(path)
        return report

    def list_audits(self, form_id: Optional[str] = None) -> List[Dict[str, str]]:
        """List existing audit reports."""
        audits = []
        pattern = f"audit_{form_id}_*.json" if form_id else "audit_*.json"
        for path in self._audit_dir.glob(pattern):
            try:
                with open(path) as f:
                    data = json.load(f)
                audits.append({
                    "audit_id": data.get("audit_id"),
                    "form_id": data.get("form_id"),
                    "timestamp": data.get("timestamp"),
                    "overall_pass": data.get("overall_pass"),
                    "path": str(path),
                })
            except Exception:
                pass
        return sorted(audits, key=lambda x: x.get("timestamp", ""), reverse=True)


def run_all_checks(profile: DataProfile) -> Dict[str, Any]:
    """
    Run all consistency and arithmetic checks against a data profile.
    Returns combined results.
    """
    checker = ConsistencyChecker()
    verifier = ArithmeticVerifier()

    income_issues = checker.check_income_consistency(profile)
    asset_debt_issues = checker.check_asset_debt_totals(profile)
    cmi_issues = verifier.verify_cmi(profile)
    means_test_issues = verifier.verify_means_test(profile)

    all_issues = income_issues + asset_debt_issues + cmi_issues + means_test_issues
    errors = [i for i in all_issues if i.get("severity") == "error"]
    warnings = [i for i in all_issues if i.get("severity") == "warning"]

    return {
        "total_issues": len(all_issues),
        "errors": len(errors),
        "warnings": len(warnings),
        "issues": all_issues,
        "pass": len(errors) == 0,
        "summary": (
            f"All checks passed ({len(warnings)} warnings)" if len(errors) == 0
            else f"FAILED: {len(errors)} errors, {len(warnings)} warnings"
        ),
    }


# ======================================================================
# Helpers
# ======================================================================

def _parse_dollar(value: Any) -> Optional[float]:
    """Parse a dollar amount from various formats."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").replace("(", "-").replace(")", "").strip()
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None


def _fmt(amount: Optional[float]) -> str:
    """Format a dollar amount."""
    if amount is None:
        return "N/A"
    if amount < 0:
        return f"-${abs(amount):,.2f}"
    return f"${amount:,.2f}"
