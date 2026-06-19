"""Tests for cross-case (multi-document) data contracts.

Focused regression coverage for the data-frame column-ordering invariant: the
R bridge binds rows positionally, so every case's row must expose the model's
variables in one fixed key order regardless of the order the LLM emitted codings.
"""

from __future__ import annotations

from pt.cq_bridge import is_r_available
from pt.multi_pipeline import _apply_confidence_threshold
from pt.schemas_multi import CaseBinarization, VariableCoding


_ORDER = ["fiscal_crisis", "state_breakdown", "revolution"]


def _coding(name: str, value: int, confidence: float = 0.9) -> VariableCoding:
    return VariableCoding(
        variable_name=name, value=value, confidence=confidence, justification="x"
    )


def _case(case_id: str, pairs: list[tuple[str, int]]) -> CaseBinarization:
    return CaseBinarization(
        case_id=case_id,
        source_file=f"{case_id}.txt",
        codings=[_coding(n, v) for n, v in pairs],
    )


class TestToRowOrdering:
    def test_canonical_order_independent_of_coding_order(self):
        # Two cases code the same variables but in different emission order.
        a = _case("A", [("fiscal_crisis", 1), ("state_breakdown", 0), ("revolution", 1)])
        b = _case("B", [("state_breakdown", 1), ("fiscal_crisis", 0), ("revolution", 1)])
        ra, rb = a.to_row(_ORDER), b.to_row(_ORDER)
        assert list(ra.keys()) == _ORDER
        assert list(rb.keys()) == _ORDER
        # Values stay bound to the correct variable, not the positional slot.
        assert rb["fiscal_crisis"] == 0
        assert rb["state_breakdown"] == 1

    def test_uncoded_variable_becomes_none(self):
        c = _case("C", [("revolution", 1)])
        assert c.to_row(_ORDER) == {
            "fiscal_crisis": None,
            "state_breakdown": None,
            "revolution": 1,
        }

    def test_without_order_preserves_legacy_behavior(self):
        a = _case("A", [("state_breakdown", 0), ("fiscal_crisis", 1)])
        assert list(a.to_row().keys()) == ["state_breakdown", "fiscal_crisis"]


class TestConfidenceThresholdOrdering:
    def test_threshold_rows_share_fixed_order(self):
        a = _case("A", [("fiscal_crisis", 1), ("state_breakdown", 0), ("revolution", 1)])
        b = _case("B", [("revolution", 1), ("fiscal_crisis", 0), ("state_breakdown", 1)])
        rows, _ = _apply_confidence_threshold([a, b], threshold=0.5, variable_order=_ORDER)
        assert list(rows[0].keys()) == _ORDER
        assert list(rows[1].keys()) == _ORDER

    def test_low_confidence_coding_becomes_na(self):
        a = _case("A", [("fiscal_crisis", 1)])
        a.codings[0].confidence = 0.2
        rows, n_na = _apply_confidence_threshold([a], threshold=0.5, variable_order=_ORDER)
        assert rows[0]["fiscal_crisis"] is None
        assert n_na == 1


def test_is_r_available_returns_bool():
    # Used to separate intended graceful degradation (R absent) from real CQ
    # failures (R present but the run errored, which must fail loud).
    assert isinstance(is_r_available(), bool)
