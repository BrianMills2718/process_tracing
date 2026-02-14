"""Pass: Binarize a single case against a causal model specification.

One LLM call per case. Maps extraction results to binary variable codings.
"""

from __future__ import annotations

from pt.llm import DEFAULT_MODEL, call_llm
from pt.schemas import BayesianResult, ExtractionResult
from pt.schemas_multi import CaseBinarization, CausalModelSpec, VariableCoding


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

    prompt = f"""You are coding a historical case for cross-case comparative analysis.

## Case
ID: {case_id}
Summary: {extraction.summary}

## Causal Model: {causal_model.name}
{causal_model.description}

## Variables to Code

{var_block}

## Evidence from This Case

{evidence_block}

## Bayesian Posteriors (for context)

{posterior_block}

## Instructions

For EACH variable in the causal model, assign a binary coding:
- value=1 if the evidence clearly supports the variable being present/true
- value=0 if the evidence clearly supports the variable being absent/false
- value=null if the evidence is insufficient or ambiguous (code as NA)

RULES:
1. Code from the EXTRACTED EVIDENCE ONLY. Do not use world knowledge about this case.
2. Use the observable_indicators as a guide for what to look for.
3. Cite specific evidence IDs in your justification.
4. Set confidence based on how strong the evidence is:
   - 0.9+ = Multiple clear evidence items directly support this coding
   - 0.7-0.9 = Evidence supports but with some ambiguity
   - 0.5-0.7 = Weak or indirect evidence; coding is a judgment call
   - Below 0.5 = Insufficient evidence; set value to null (NA)
5. Be CONSISTENT: apply the same evidentiary standard across all variables.
6. The outcome variable ({causal_model.outcome_variable}) should almost always be codeable.

Return the coding for case_id="{case_id}" with source_file="{source_file}".
Include analyst_notes summarizing any difficult coding decisions."""

    kwargs: dict = {}
    if model is not None:
        kwargs["model"] = model
    result = call_llm(prompt, _BinarizationResponse, **kwargs)

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
