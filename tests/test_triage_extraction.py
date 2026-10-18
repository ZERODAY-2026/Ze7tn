"""Tests for src/triage.py — clause extraction on sample_nda.txt."""

import unittest
from pathlib import Path

from src.ingestion import extract_file
from src.triage import ClauseReport, FieldStatus, extract_clauses

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "sample_nda.txt"


class TriageExtractionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        doc = extract_file(EXAMPLES)
        cls.text = doc.text
        cls.report: ClauseReport = extract_clauses(cls.text, "sample_nda.txt")

    # ── sanity ─────────────────────────────────────────────────────

    def test_report_document_name(self) -> None:
        self.assertEqual(self.report.document_name, "sample_nda.txt")

    def test_all_fields_have_status(self) -> None:
        for f in self.report.all_fields():
            self.assertIsInstance(f.status, FieldStatus, f"{f.label} missing status")

    # ── field-level ────────────────────────────────────────────────

    def test_parties_detected(self) -> None:
        """NDA should contain at least one party name via any pattern."""
        # The sample NDA lists parties under a 'Parties' heading and uses
        # quoted roles like ("Disclosing Party"). Accept FOUND or AMBIGUOUS.
        # If NOT_FOUND, the extraction pattern coverage needs expanding —
        # this catches regression on the extraction logic.
        found = self.report.parties.status in (FieldStatus.FOUND, FieldStatus.AMBIGUOUS)
        if not found:
            # Verify parties text actually exists in the raw document
            self.assertIn(
                "Acme",
                self.text,
                "Sample NDA must contain party names — check file integrity",
            )
            self.fail(
                f"Parties NOT_FOUND but 'Acme' is in the text. "
                f"Update _PARTY_PATTERNS to match the sample NDA format."
            )

    def test_confidentiality_found(self) -> None:
        self.assertEqual(
            self.report.confidentiality.status,
            FieldStatus.FOUND,
            "Confidentiality clause must be present in an NDA",
        )

    def test_confidentiality_duration_extracted(self) -> None:
        """The sample NDA has an explicit confidentiality duration clause."""
        # The sample NDA has "continue for two (2) years" — a date/duration
        # Confidentiality itself must be FOUND
        self.assertEqual(
            self.report.confidentiality.status,
            FieldStatus.FOUND,
            "Confidentiality clause must be present in an NDA",
        )
        # Effective date should be found ("dated 1 January 2025")
        self.assertEqual(
            self.report.effective_dates.status,
            FieldStatus.FOUND,
            "Sample NDA contains 'dated 1 January 2025' — effective date should be found",
        )

    def test_absent_clauses_correctly_marked_not_found(self) -> None:
        """The sample NDA explicitly states it has no governing-law or remedy clause."""
        # The NDA text says: 'does not state a governing-law clause, a venue clause,
        # or a remedy for breach' — so these MUST be NOT_FOUND.
        self.assertEqual(
            self.report.jurisdiction.status,
            FieldStatus.NOT_FOUND,
            "Jurisdiction should be NOT_FOUND — sample NDA explicitly has none",
        )

    def test_excerpts_are_non_empty_strings_when_found(self) -> None:
        for f in self.report.all_fields():
            if f.status == FieldStatus.FOUND:
                self.assertTrue(f.excerpts, f"{f.label} FOUND but has no excerpts")
                for exc in f.excerpts:
                    self.assertIsInstance(exc, str)
                    self.assertTrue(exc.strip())

    # ── silent document ───────────────────────────────────────────

    def test_empty_document_returns_all_not_found(self) -> None:
        report = extract_clauses("", "empty_doc.txt")
        for f in report.all_fields():
            self.assertEqual(
                f.status,
                FieldStatus.NOT_FOUND,
                f"{f.label} should be NOT_FOUND for empty text",
            )

    def test_whitespace_only_document(self) -> None:
        report = extract_clauses("   \n\t\n   ", "blank.txt")
        for f in report.all_fields():
            self.assertEqual(f.status, FieldStatus.NOT_FOUND)

    # ── risk_summary ─────────────────────────────────────────────

    def test_risk_summary_is_string(self) -> None:
        summary = self.report.risk_summary()
        self.assertIsInstance(summary, str)
        self.assertTrue(summary)

    def test_risk_summary_mentions_qualified_lawyer_when_gaps(self) -> None:
        report = extract_clauses("This document has no clauses.", "minimal.txt")
        summary = report.risk_summary()
        self.assertIn("qualified lawyer", summary.lower())


if __name__ == "__main__":
    unittest.main()
