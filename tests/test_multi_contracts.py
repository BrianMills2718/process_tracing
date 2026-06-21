"""Tests for cross-case (multi-document) data contracts.

Focused regression coverage for the data-frame column-ordering invariant: the
R bridge binds rows positionally, so every case's row must expose the model's
variables in one fixed key order regardless of the order the LLM emitted codings.
"""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from pt.cq_bridge import is_r_available
from pt.multi_pipeline import _apply_confidence_threshold, _run_single_case, run_multi_pipeline
from pt.pass_binarize import (
    _BinarizationResponse,
    _binarization_response_model,
    _validate_binarization_contract,
)
from pt.schemas_multi import (
    CaseBinarization,
    CausalEdgeSpec,
    CausalModelSpec,
    CausalQueriesResult,
    CausalVariable,
    VariableCoding,
)


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


def _model() -> CausalModelSpec:
    return CausalModelSpec(
        name="m",
        description="d",
        outcome_variable="revolution",
        variables=[
            CausalVariable(
                name="fiscal_crisis",
                description="fiscal crisis present",
                description_zero="fiscal crisis absent",
            ),
            CausalVariable(
                name="state_breakdown",
                description="state breakdown present",
                description_zero="state breakdown absent",
            ),
            CausalVariable(
                name="revolution",
                description="revolution present",
                description_zero="revolution absent",
            ),
        ],
        edges=[
            CausalEdgeSpec(parent="fiscal_crisis", child="state_breakdown"),
            CausalEdgeSpec(parent="state_breakdown", child="revolution"),
        ],
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

    def test_rejects_unknown_variable_when_ordered(self):
        c = _case("C", [("fiscal_crisis", 1), ("state_breakdown", 0), ("ghost", 1)])
        with pytest.raises(ValueError, match="not in model"):
            c.to_row(_ORDER)


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


class TestBinarizeResponseSchema:
    def test_excludes_system_assigned_fields(self):
        # case_id/source_file are system-assigned and must not be in the schema
        # the LLM is asked to fill (CLAUDE.md: exclude system ids from LLM schema).
        props = set(_BinarizationResponse.model_json_schema()["properties"])
        assert props == {"codings", "analyst_notes"}

    def test_dynamic_schema_enumerates_variables_and_evidence_ids(self):
        schema = _binarization_response_model(
            variable_names=["fiscal_crisis", "revolution"],
            evidence_ids=["evi_1", "evi_2"],
        ).model_json_schema()
        schema_text = str(schema)
        assert "'enum': ['fiscal_crisis', 'revolution']" in schema_text
        assert "'enum': ['evi_1', 'evi_2']" in schema_text


class TestCrossCaseValidation:
    def test_causal_model_rejects_duplicate_variables(self):
        with pytest.raises(ValidationError):
            CausalModelSpec(
                name="m",
                description="d",
                outcome_variable="x",
                variables=[
                    CausalVariable(name="x", description="one", description_zero="zero"),
                    CausalVariable(name="x", description="one", description_zero="zero"),
                ],
                edges=[],
            )

    def test_causal_model_rejects_invalid_r_identifier(self):
        with pytest.raises(ValidationError):
            CausalVariable(name="bad-name", description="one", description_zero="zero")

    def test_variable_coding_rejects_non_binary_value(self):
        with pytest.raises(ValidationError):
            VariableCoding(
                variable_name="fiscal_crisis",
                value=2,
                confidence=0.9,
                justification="x",
            )

    def test_case_binarization_rejects_duplicate_codings(self):
        with pytest.raises(ValidationError):
            _case("A", [("fiscal_crisis", 1), ("fiscal_crisis", 0)])

    def test_binarize_contract_rejects_unknown_model_variable(self):
        extraction = _minimal_result().extraction
        codings = [
            _coding("fiscal_crisis", 1),
            _coding("state_breakdown", 0),
            _coding("revolution", 1),
            _coding("ghost", 1),
        ]
        with pytest.raises(ValueError, match="not in model"):
            _validate_binarization_contract(
                case_id="A",
                codings=codings,
                causal_model=_model(),
                extraction=extraction,
            )

    def test_binarize_contract_rejects_unknown_evidence_id(self):
        extraction = _minimal_result().extraction
        codings = [
            _coding("fiscal_crisis", 1),
            _coding("state_breakdown", 0),
            VariableCoding(
                variable_name="revolution",
                value=1,
                confidence=0.9,
                justification="x",
                evidence_ids=["evi_missing"],
            ),
        ]
        with pytest.raises(ValueError, match="unknown evidence ids"):
            _validate_binarization_contract(
                case_id="A",
                codings=codings,
                causal_model=_model(),
                extraction=extraction,
            )


def _minimal_result():
    from pt.bayesian import run_bayesian_update
    from pt.schemas import (
        AbsenceResult, Evidence, EvidenceLikelihood, ExtractionResult, Hypothesis,
        HypothesisLikelihood, HypothesisSpace, HypothesisVerdict, ProcessTracingResult,
        SynthesisResult, TestingResult,
    )
    ext = ExtractionResult(summary="s", evidence=[
        Evidence(id="evi_1", description="d", source_text="q")])
    hs = HypothesisSpace(research_question="rq", hypotheses=[Hypothesis(
        id="h1", description="d", source="text", theoretical_basis="t",
        causal_mechanism="m", observable_predictions=[])])
    testing = TestingResult(evidence_likelihoods=[EvidenceLikelihood(
        evidence_id="evi_1", justification="j", hypothesis_likelihoods=[
            HypothesisLikelihood(hypothesis_id="h1", relative_likelihood=1.0,
                                 diagnostic_type="straw_in_the_wind")])])
    synth = SynthesisResult(verdicts=[HypothesisVerdict(
        hypothesis_id="h1", status="supported", key_evidence_for=[],
        key_evidence_against=[], reasoning="r", steelman="s")],
        comparative_analysis="c", analytical_narrative="n", limitations=[],
        suggested_further_tests=[])
    return ProcessTracingResult(extraction=ext, hypothesis_space=hs, testing=testing,
        absence=AbsenceResult(), bayesian=run_bayesian_update(testing, hypothesis_ids=["h1"]),
        synthesis=synth)


class TestSingleCaseCache:
    """The per-case cache must be keyed on text content AND resolved model,
    not merely the existence of result.json."""

    def _setup(self, tmp_path, text):
        src = tmp_path / "case.txt"
        src.write_text(text, encoding="utf-8")
        return str(src), str(tmp_path / "out")

    def test_hit_only_when_text_and_model_match(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_pipeline(text, **kwargs):
            calls["n"] += 1
            return _minimal_result()

        monkeypatch.setattr("pt.pipeline.run_pipeline", fake_pipeline)
        monkeypatch.setattr("pt.report.generate_report", lambda r: "<html></html>")
        src, out = self._setup(tmp_path, "the text body " * 50)

        _run_single_case(src, out, model="model-a")
        assert calls["n"] == 1
        # Same text + model → cache hit, no recompute.
        _run_single_case(src, out, model="model-a")
        assert calls["n"] == 1
        # cache_meta.json was written alongside the result.
        assert os.path.isfile(os.path.join(out, "cache_meta.json"))

    def test_model_change_invalidates_cache(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        monkeypatch.setattr("pt.pipeline.run_pipeline",
                            lambda text, **k: (calls.__setitem__("n", calls["n"] + 1) or _minimal_result()))
        monkeypatch.setattr("pt.report.generate_report", lambda r: "x")
        src, out = self._setup(tmp_path, "the text body " * 50)
        _run_single_case(src, out, model="model-a")
        _run_single_case(src, out, model="model-b")  # different model → recompute
        assert calls["n"] == 2

    def test_text_change_invalidates_cache(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        monkeypatch.setattr("pt.pipeline.run_pipeline",
                            lambda text, **k: (calls.__setitem__("n", calls["n"] + 1) or _minimal_result()))
        monkeypatch.setattr("pt.report.generate_report", lambda r: "x")
        src, out = self._setup(tmp_path, "original body " * 50)
        _run_single_case(src, out, model="model-a")
        (tmp_path / "case.txt").write_text("edited body " * 50, encoding="utf-8")
        _run_single_case(src, out, model="model-a")  # text changed → recompute
        assert calls["n"] == 2

    def test_legacy_cache_without_meta_recomputes(self, tmp_path, monkeypatch):
        # A result.json with no cache_meta.json (old cache) must not be trusted.
        calls = {"n": 0}
        monkeypatch.setattr("pt.pipeline.run_pipeline",
                            lambda text, **k: (calls.__setitem__("n", calls["n"] + 1) or _minimal_result()))
        monkeypatch.setattr("pt.report.generate_report", lambda r: "x")
        src, out = self._setup(tmp_path, "body " * 80)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(_minimal_result().model_dump(), f)
        _run_single_case(src, out, model="model-a")
        assert calls["n"] == 1  # recomputed despite result.json existing

    def test_theories_change_invalidates_cache(self, tmp_path, monkeypatch):
        calls = {"theories": []}

        def fake_pipeline(text, **kwargs):
            calls["theories"].append(kwargs.get("theories"))
            return _minimal_result()

        monkeypatch.setattr("pt.pipeline.run_pipeline", fake_pipeline)
        monkeypatch.setattr("pt.report.generate_report", lambda r: "x")
        src, out = self._setup(tmp_path, "the text body " * 50)
        _run_single_case(src, out, model="model-a", theories="theory A")
        _run_single_case(src, out, model="model-a", theories="theory B")
        assert calls["theories"] == ["theory A", "theory B"]


class TestCausalQueriesCaseIds:
    def test_multi_pipeline_passes_case_ids_to_cq(self, tmp_path, monkeypatch):
        case_a = tmp_path / "case_a.txt"
        case_b = tmp_path / "case_b.txt"
        case_a.write_text("case a text " * 80, encoding="utf-8")
        case_b.write_text("case b text " * 80, encoding="utf-8")
        calls = []

        monkeypatch.setattr("pt.multi_pipeline._run_single_case", lambda *a, **k: _minimal_result())
        monkeypatch.setattr(
            "pt.pass_binarize.binarize_case",
            lambda case_id, **k: _case(
                case_id,
                [("fiscal_crisis", 1), ("state_breakdown", 0), ("revolution", 1)],
            ),
        )
        monkeypatch.setattr("pt.cq_bridge.is_r_available", lambda: True)

        def fake_cq(causal_model, data_frame, case_ids=None):
            calls.append(case_ids)
            return CausalQueriesResult(
                model_statement=causal_model.dagitty_statement,
                n_cases=len(data_frame),
                population_estimands=[],
                case_level_estimands=[],
            )

        monkeypatch.setattr("pt.cq_bridge.run_causal_queries", fake_cq)

        run_multi_pipeline(
            [str(case_a), str(case_b)],
            str(tmp_path / "out"),
            causal_model=_model(),
        )

        assert calls
        assert calls[0] == ["case_a", "case_b"]
