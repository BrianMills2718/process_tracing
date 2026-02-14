"""Pass: Propose a causal model from N extraction results (data-driven workflow).

Single LLM call examines all extractions and proposes a CausalModelSpec.
"""

from __future__ import annotations

from pt.llm import call_llm
from pt.schemas import ExtractionResult
from pt.schemas_multi import CausalModelSpec


def propose_causal_model(
    extractions: dict[str, ExtractionResult],
    *,
    model: str | None = None,
) -> CausalModelSpec:
    """Propose a causal model from multiple case extractions.

    Args:
        extractions: Map of case_id -> ExtractionResult.
        model: LLM model override.

    Returns:
        A proposed CausalModelSpec derived from the extraction data.
    """
    # Build case summaries
    case_blocks = []
    for case_id, ext in extractions.items():
        mechanisms = [f"    - {m.id}: {m.description}" for m in ext.mechanisms]
        mech_block = "\n".join(mechanisms) if mechanisms else "    (none)"

        hypotheses = [f"    - {h.id}: {h.description}" for h in ext.hypotheses_in_text]
        hyp_block = "\n".join(hypotheses) if hypotheses else "    (none)"

        edges = [f"    - {e.source_id} -> {e.target_id}: {e.relationship}" for e in ext.causal_edges[:15]]
        edge_block = "\n".join(edges) if edges else "    (none)"

        case_blocks.append(
            f"### {case_id}\n"
            f"Summary: {ext.summary}\n"
            f"Mechanisms:\n{mech_block}\n"
            f"Hypotheses in text:\n{hyp_block}\n"
            f"Causal edges (top 15):\n{edge_block}"
        )
    cases_text = "\n\n".join(case_blocks)

    prompt = f"""You are designing a causal model for cross-case comparative analysis.

You have extraction results from {len(extractions)} cases:

{cases_text}

## Task

Propose a causal model (directed acyclic graph) that captures the common causal structure across these cases. The model will be used with CausalQueries (an R package for Bayesian causal inference with binary data).

## Constraints

1. ALL variables must be BINARY (0/1). Define what 1 means and what 0 means.
2. Maximum 10 variables (CausalQueries has exponential complexity).
3. Include exactly ONE outcome variable.
4. The DAG must be ACYCLIC.
5. Variable names must be valid R identifiers (lowercase, underscores, no spaces).
6. Variables must be GENERAL — applicable across all {len(extractions)} cases, not specific to any single case.
7. Include observable_indicators for each variable — concrete things a coder should look for in texts.

## Guidelines

- Derive variables from the ACTUAL mechanisms and causal edges extracted from these cases.
- Do NOT just reproduce a standard theoretical framework (e.g., Skocpol). Ground the model in what the texts actually say.
- Prefer fewer, more clearly defined variables over many vague ones.
- The DAG should represent theoretically meaningful causal pathways, not just correlations.
- Include both proximate causes (triggers) and structural/background conditions.
- If you see a common mediating mechanism across cases, include it as a variable.

Return a CausalModelSpec with a descriptive name and description explaining the theoretical logic."""

    kwargs: dict = {}
    if model is not None:
        kwargs["model"] = model
    result = call_llm(prompt, CausalModelSpec, **kwargs)

    # Validate
    errors = result.validate_dag()
    if errors:
        raise ValueError(f"Proposed model has validation errors: {errors}")

    return result
