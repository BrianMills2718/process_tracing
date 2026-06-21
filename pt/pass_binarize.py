"""Pass: Binarize a single case against a causal model specification.

One LLM call per case. Maps extraction results to binary variable codings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from llm_client import render_prompt
from pydantic import BaseModel, Field, create_model

from pt.llm import DEFAULT_MODEL, call_llm
from pt.schemas import BayesianResult, ExtractionResult
from pt.schemas_multi import CaseBinarization, CausalModelSpec, VariableCoding

PROMPTS_DIR = Path(__file__).parent / "prompts"


class _BinarizationResponse(BaseModel):
    """LLM-facing schema. Excludes the system-assigned case_id/source_file so
    the model does not waste tokens generating values that are overwritten."""
    codings: list[VariableCoding]
    analyst_notes: str = ""


def _literal_enum(values: list[str]) -> Any:
    """Build a Literal type from runtime IDs for JSON-schema enum enforcement."""
    if not values:
        raise ValueError("cannot build enum schema from an empty id list")
    return Literal.__getitem__(tuple(values))


def _binarization_response_model(
    *,
    variable_names: list[str],
    evidence_ids: list[str],
) -> type[BaseModel]:
    """Create an LLM-facing schema constrained to this model and extraction."""
    variable_name_type = _literal_enum(variable_names)
    evidence_id_type: Any = _literal_enum(evidence_ids) if evidence_ids else str
    evidence_id_list: Any = list.__class_getitem__(evidence_id_type)

    variable_coding = create_model(
        "VariableCodingResponse",
        variable_name=(
            variable_name_type,
            Field(description="One of the exact variable names in the causal model."),
        ),
        value=(
            Literal[0, 1] | None,
            Field(default=None, description="0, 1, or null when evidence is insufficient."),
        ),
        confidence=(
            float,
            Field(ge=0.0, le=1.0, description="Confidence in this coding."),
        ),
        justification=(
            str,
            Field(description="Justification citing extracted evidence IDs."),
        ),
        evidence_ids=(
            evidence_id_list,
            Field(
                default_factory=list,
                description="Exact evidence IDs from this extraction that support the coding.",
            ),
        ),
    )
    variable_coding_list: Any = list.__class_getitem__(variable_coding)
    return create_model(
        "BinarizationResponse",
        codings=(
            variable_coding_list,
            Field(description="Exactly one coding per causal model variable."),
        ),
        analyst_notes=(
            str,
            Field(default="", description="Notes on difficult coding decisions."),
        ),
    )


def _validate_binarization_contract(
    *,
    case_id: str,
    codings: list[VariableCoding],
    causal_model: CausalModelSpec,
    extraction: ExtractionResult,
) -> None:
    """Fail loud when LLM codings cannot form the model's case row."""
    coded_names = [c.variable_name for c in codings]
    duplicate_names = sorted({name for name in coded_names if coded_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            f"Binarization for {case_id} has duplicate variable codings: {duplicate_names}"
        )

    coded_vars = set(coded_names)
    model_vars = set(causal_model.variable_names)
    missing = sorted(model_vars - coded_vars)
    if missing:
        raise ValueError(
            f"Binarization for {case_id} missing variables: {missing}"
        )
    unknown = sorted(coded_vars - model_vars)
    if unknown:
        raise ValueError(
            f"Binarization for {case_id} includes variables not in model: {unknown}"
        )

    valid_evidence_ids = {e.id for e in extraction.evidence}
    unknown_evidence_ids = sorted({
        evidence_id
        for coding in codings
        for evidence_id in coding.evidence_ids
        if evidence_id not in valid_evidence_ids
    })
    if unknown_evidence_ids:
        raise ValueError(
            f"Binarization for {case_id} cites unknown evidence ids: {unknown_evidence_ids}"
        )


def binarize_case(
    *,
    case_id: str,
    source_file: str,
    extraction: ExtractionResult,
    bayesian: BayesianResult,
    causal_model: CausalModelSpec,
    model: str | None = None,
    trace_id: str | None = None,
) -> CaseBinarization:
    """Binarize one case against a causal model.

    Uses the extraction results and Bayesian posteriors as context.
    Returns binary codings (0/1/null) for each variable in the model.
    """
    # Build variable descriptions for prompt
    var_descriptions = []
    for v in causal_model.variables:
        indicators = "\n".join(f"      - {ind}" for ind in v.observable_indicators) if v.observable_indicators else "      (none specified)"
        var_descriptions.append(
            f"  {v.name}:\n"
            f"    1 = {v.description}\n"
            f"    0 = {v.description_zero}\n"
            f"    Observable indicators:\n{indicators}"
        )
    var_block = "\n\n".join(var_descriptions)

    # Build evidence summary
    evidence_lines = []
    for evi in extraction.evidence:
        evidence_lines.append(
            f"  {evi.id} [{evi.evidence_type}]: {evi.description}\n"
            f"    Source: \"{evi.source_text[:150]}...\""
            if len(evi.source_text) > 150
            else f"  {evi.id} [{evi.evidence_type}]: {evi.description}\n"
            f"    Source: \"{evi.source_text}\""
        )
    evidence_block = "\n".join(evidence_lines)

    # Build Bayesian context
    posterior_lines = []
    for hp in bayesian.posteriors:
        posterior_lines.append(f"  {hp.hypothesis_id}: posterior={hp.final_posterior:.3f} ({hp.robustness})")
    posterior_block = "\n".join(posterior_lines)

    messages = render_prompt(
        PROMPTS_DIR / "binarize.yaml",
        case_id=case_id,
        summary=extraction.summary,
        model_name=causal_model.name,
        model_description=causal_model.description,
        var_descriptions=var_block,
        evidence_block=evidence_block,
        posterior_block=posterior_block,
        outcome_variable=causal_model.outcome_variable,
    )

    if trace_id is None:
        trace_id = hashlib.sha256(case_id.encode()).hexdigest()[:8]
    kwargs: dict = {}
    if model is not None:
        kwargs["model"] = model
    response_model = _binarization_response_model(
        variable_names=causal_model.variable_names,
        evidence_ids=[e.id for e in extraction.evidence],
    )
    raw_result = call_llm(
        messages[0]["content"],
        response_model,
        task=f"process_tracing.binarize.{case_id}",
        trace_id=trace_id,
        **kwargs,
    )
    result = _BinarizationResponse.model_validate(raw_result.model_dump())

    _validate_binarization_contract(
        case_id=case_id,
        codings=result.codings,
        causal_model=causal_model,
        extraction=extraction,
    )

    # Enforce confidence < 0.5 → null
    for coding in result.codings:
        if coding.confidence < 0.5 and coding.value is not None:
            coding.value = None

    # Attach system-assigned case identity (never LLM-provided).
    return CaseBinarization(
        case_id=case_id,
        source_file=source_file,
        codings=result.codings,
        analyst_notes=result.analyst_notes,
    )
