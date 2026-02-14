"""Python-side bridge to CausalQueries (R package).

Communicates with R via subprocess + JSON (stdin -> stdout).
No rpy2 dependency — R script is testable independently.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional

from pt.schemas_multi import (
    CausalModelSpec,
    CausalQueriesResult,
    CQCaseLevelResult,
    CQEstimand,
)


class CQBridgeError(Exception):
    """Raised when CausalQueries bridge fails."""


CQ_TIMEOUT = 600  # 10 minutes — Stan can be slow


def _find_rscript() -> str:
    """Find the Rscript executable or raise."""
    path = shutil.which("Rscript")
    if path is None:
        raise CQBridgeError(
            "Rscript not found. Install R: apt install r-base, "
            "then install CausalQueries: Rscript -e 'install.packages(\"CausalQueries\")'"
        )
    return path


def _build_cq_input(
    causal_model: CausalModelSpec,
    data_frame: list[dict[str, int | None]],
    case_ids: list[str] | None = None,
) -> dict:
    """Build the JSON input for the R script."""
    # Default queries: ATE for each non-outcome variable -> outcome
    outcome = causal_model.outcome_variable
    queries = []
    for v in causal_model.variables:
        if v.name != outcome:
            queries.append(f"{outcome}[{v.name}=1] - {outcome}[{v.name}=0]")

    # Case-level queries (for each case, P(cause -> outcome | this case's data))
    case_level_queries = []
    if case_ids:
        for i, case_id in enumerate(case_ids):
            for v in causal_model.variables:
                if v.name != outcome:
                    case_level_queries.append({
                        "case_id": case_id,
                        "row_index": i + 1,  # R is 1-indexed
                        "query": f"{outcome}[{v.name}=1] - {outcome}[{v.name}=0]",
                    })

    return {
        "model_statement": causal_model.dagitty_statement,
        "data": data_frame,
        "queries": queries,
        "case_level_queries": case_level_queries,
        "restrictions": causal_model.restrictions,
        "confounds": [list(pair) for pair in causal_model.confounds],
    }


def _parse_cq_output(raw: dict, n_cases: int, model_statement: str) -> CausalQueriesResult:
    """Parse the JSON output from the R script into typed results."""
    population = []
    for est in raw.get("population_estimands", []):
        population.append(CQEstimand(
            query=est["query"],
            given=est.get("given", ""),
            using=est.get("using", "posteriors"),
            mean=est["mean"],
            sd=est.get("sd"),
            cred_low=est.get("cred_low"),
            cred_high=est.get("cred_high"),
        ))

    case_level = []
    for est in raw.get("case_level_estimands", []):
        case_level.append(CQCaseLevelResult(
            case_id=est["case_id"],
            query=est["query"],
            mean=est["mean"],
            sd=est.get("sd"),
            cred_low=est.get("cred_low"),
            cred_high=est.get("cred_high"),
        ))

    return CausalQueriesResult(
        model_statement=model_statement,
        n_cases=n_cases,
        population_estimands=population,
        case_level_estimands=case_level,
        diagnostics=raw.get("diagnostics", {}),
    )


def run_causal_queries(
    causal_model: CausalModelSpec,
    data_frame: list[dict[str, int | None]],
    case_ids: list[str] | None = None,
) -> CausalQueriesResult:
    """Run CausalQueries via R subprocess.

    Args:
        causal_model: The causal DAG specification.
        data_frame: List of rows (one per case), each a dict of variable_name -> 0/1/None.
        case_ids: Optional list of case IDs (same order as data_frame rows).

    Returns:
        CausalQueriesResult with population and case-level estimands.

    Raises:
        CQBridgeError: If R is not installed, CQ package missing, or Stan fails.
    """
    rscript = _find_rscript()

    # Find the R script
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "cq_runner.R",
    )
    if not os.path.isfile(script_path):
        raise CQBridgeError(f"R script not found: {script_path}")

    cq_input = _build_cq_input(causal_model, data_frame, case_ids)
    input_json = json.dumps(cq_input)

    try:
        proc = subprocess.run(
            [rscript, script_path],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=CQ_TIMEOUT,
        )
    except subprocess.TimeoutExpired as e:
        raise CQBridgeError(f"CausalQueries timed out after {CQ_TIMEOUT}s") from e

    if proc.returncode != 0:
        raise CQBridgeError(
            f"CausalQueries R script failed (exit {proc.returncode}):\n{proc.stderr}"
        )

    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise CQBridgeError(
            f"Failed to parse CQ output as JSON: {e}\nstdout: {proc.stdout[:500]}"
        ) from e

    if "error" in output:
        raise CQBridgeError(f"CausalQueries error: {output['error']}")

    return _parse_cq_output(
        output,
        n_cases=len(data_frame),
        model_statement=causal_model.dagitty_statement,
    )
