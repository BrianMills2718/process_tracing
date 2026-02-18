"""Pass: Binarize a single case against a causal model specification.

One LLM call per case. Maps extraction results to binary variable codings.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from llm_client import render_prompt

from pt.llm import DEFAULT_MODEL, call_llm
from pt.schemas import BayesianResult, ExtractionResult
from pt.schemas_multi import CaseBinarization, CausalModelSpec, VariableCoding

PROMPTS_DIR = Path(__file__).parent / "prompts"


class _BinarizationResponse(CaseBinarization):
    """Wrapper so the LLM returns the full CaseBinarization structure."""
    pass


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
        source_file=source_file,
    )

    if trace_id is None:
        trace_id = hashlib.sha256(case_id.encode()).hexdigest()[:8]
    kwargs: dict = {}
    if model is not None:
        kwargs["model"] = model
    result = call_llm(
        messages[0]["content"],
        _BinarizationResponse,
        task=f"process_tracing.binarize.{case_id}",
        trace_id=trace_id,
        **kwargs,
    )

    # Validate: every model variable must be coded
    coded_vars = {c.variable_name for c in result.codings}
    model_vars = set(causal_model.variable_names)
    missing = model_vars - coded_vars
    if missing:
        raise ValueError(
            f"Binarization for {case_id} missing variables: {missing}"
        )

    # Enforce confidence < 0.5 → null
    for coding in result.codings:
        if coding.confidence < 0.5 and coding.value is not None:
            coding.value = None

    # Override case_id and source_file to ensure consistency
    result.case_id = case_id
    result.source_file = source_file

    return result
