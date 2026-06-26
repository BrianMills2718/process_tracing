"""Data contracts for multi-document cross-case analysis."""

from __future__ import annotations

import re
from typing import Literal, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, model_validator


# ── Causal Model Specification ─────────────────────────────────────

_R_IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"
_R_IDENTIFIER_RE = re.compile(_R_IDENTIFIER_PATTERN)


class CausalVariable(BaseModel):
    name: str = Field(
        pattern=_R_IDENTIFIER_PATTERN,
        description="Valid R identifier, e.g. 'fiscal_crisis'",
    )
    description: str = Field(description="What 1 means")
    description_zero: str = Field(description="What 0 means")
    observable_indicators: list[str] = Field(
        default_factory=list,
        description="Concrete things to look for in texts",
    )


class CausalEdgeSpec(BaseModel):
    parent: str = Field(pattern=_R_IDENTIFIER_PATTERN)
    child: str = Field(pattern=_R_IDENTIFIER_PATTERN)


class CausalModelSpec(BaseModel):
    name: str
    description: str
    outcome_variable: str
    variables: list[CausalVariable]
    edges: list[CausalEdgeSpec]
    restrictions: list[str] = Field(
        default_factory=list,
        description="CausalQueries monotonicity constraints, e.g. 'fiscal_crisis[fiscal_crisis=1] > fiscal_crisis[fiscal_crisis=0]'",
    )
    confounds: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of variables that share an unobserved common cause",
    )

    @property
    def variable_names(self) -> list[str]:
        return [v.name for v in self.variables]

    @model_validator(mode="after")
    def _valid_dag_contract(self) -> "CausalModelSpec":
        errors = self.validate_dag()
        if errors:
            raise ValueError(f"invalid causal model: {errors}")
        return self

    @property
    def dagitty_statement(self) -> str:
        """Generate a DAGitty/CausalQueries model statement like 'X -> M -> Y'."""
        parts: list[str] = []
        for edge in self.edges:
            parts.append(f"{edge.parent} -> {edge.child}")
        return "; ".join(parts)

    def validate_dag(self) -> list[str]:
        """Return a list of validation errors (empty = valid)."""
        errors: list[str] = []
        variable_names = self.variable_names
        names = set(variable_names)

        duplicates = sorted({name for name in variable_names if variable_names.count(name) > 1})
        if duplicates:
            errors.append(f"duplicate variable names: {duplicates}")

        invalid_names = sorted(name for name in variable_names if not _R_IDENTIFIER_RE.match(name))
        if invalid_names:
            errors.append(f"invalid R identifiers: {invalid_names}")

        if self.outcome_variable not in names:
            errors.append(f"outcome_variable '{self.outcome_variable}' not in variables")

        for edge in self.edges:
            if edge.parent not in names:
                errors.append(f"edge parent '{edge.parent}' not in variables")
            if edge.child not in names:
                errors.append(f"edge child '{edge.child}' not in variables")

        # Check acyclicity via topological sort
        adj: dict[str, list[str]] = {n: [] for n in names}
        for edge in self.edges:
            if edge.parent in adj:
                adj[edge.parent].append(edge.child)
        visited: set[str] = set()
        in_stack: set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for child in adj.get(node, []):
                if child in in_stack:
                    return True
                if child not in visited and _has_cycle(child):
                    return True
            in_stack.discard(node)
            return False

        for node in names:
            if node not in visited:
                if _has_cycle(node):
                    errors.append("DAG contains a cycle")
                    break

        if len(self.variables) > 12:
            errors.append(f"Too many variables ({len(self.variables)}); CausalQueries max ~10-12")

        return errors

    @classmethod
    def from_yaml(cls, path: str) -> CausalModelSpec:
        """Load a causal model specification from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Write causal model specification to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="python"),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )


# ── Binarization ───────────────────────────────────────────────────


class VariableCoding(BaseModel):
    variable_name: str = Field(pattern=_R_IDENTIFIER_PATTERN)
    value: Literal[0, 1] | None = Field(
        default=None,
        description="0, 1, or null (NA if insufficient evidence)",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this coding (0.0-1.0)",
    )
    justification: str = Field(
        description="Cites specific evidence IDs from the extraction",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence IDs that support this coding",
    )


class CaseBinarization(BaseModel):
    case_id: str
    source_file: str
    codings: list[VariableCoding]
    analyst_notes: str = ""

    @model_validator(mode="after")
    def _unique_codings(self) -> "CaseBinarization":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for coding in self.codings:
            if coding.variable_name in seen:
                duplicates.add(coding.variable_name)
            seen.add(coding.variable_name)
        if duplicates:
            raise ValueError(f"duplicate variable codings: {sorted(duplicates)}")
        return self

    def to_row(
        self, variable_order: Optional[list[str]] = None
    ) -> dict[str, Optional[int]]:
        """Convert to a flat dict suitable for a data frame row.

        When ``variable_order`` is given (the causal model's canonical variable
        list), the returned dict has exactly those keys in that order, with any
        uncoded variable set to None. This guarantees every case produces rows
        with an identical, fixed column order — without it, rows are keyed in
        whatever order the LLM emitted codings, which can silently transpose
        columns when the data frame is bound row-wise in R (rbind is positional).
        """
        coded: dict[str, Optional[int]] = {}
        for coding in self.codings:
            if coding.variable_name in coded:
                raise ValueError(
                    f"duplicate variable coding in {self.case_id}: {coding.variable_name}"
                )
            coded[coding.variable_name] = coding.value
        if variable_order is None:
            return coded
        allowed = set(variable_order)
        unknown = sorted(set(coded) - allowed)
        if unknown:
            raise ValueError(
                f"binarization for {self.case_id} has variables not in model: {unknown}"
            )
        return {name: coded.get(name) for name in variable_order}


# ── CausalQueries Results ─────────────────────────────────────────


class CQEstimand(BaseModel):
    query: str = Field(description="CQ query string")
    given: str = ""
    using: str = Field(description="'priors' or 'posteriors'")
    mean: float
    sd: Optional[float] = None
    cred_low: Optional[float] = None
    cred_high: Optional[float] = None


class CQCaseLevelResult(BaseModel):
    case_id: str
    query: str
    mean: float
    sd: Optional[float] = None
    cred_low: Optional[float] = None
    cred_high: Optional[float] = None


class CausalQueriesResult(BaseModel):
    model_statement: str
    n_cases: int
    population_estimands: list[CQEstimand]
    case_level_estimands: list[CQCaseLevelResult] = Field(default_factory=list)
    diagnostics: dict = Field(default_factory=dict)


# ── Sensitivity Analysis ──────────────────────────────────────────


class BinarizationSensitivityRun(BaseModel):
    confidence_threshold: float = Field(
        description="Codings below this threshold become NA",
    )
    data_frame: list[dict[str, Optional[int]]] = Field(
        description="Binarized data at this threshold",
    )
    cq_result: Optional[CausalQueriesResult] = None
    n_na_codings: int = Field(
        description="How many codings became NA at this threshold",
    )


class BinarizationSensitivity(BaseModel):
    runs: list[BinarizationSensitivityRun]
    stable_estimands: list[str] = Field(
        default_factory=list,
        description="Queries whose CQ posteriors vary < 0.1 across thresholds",
    )
    fragile_estimands: list[str] = Field(
        default_factory=list,
        description="Queries whose CQ posteriors vary >= 0.1 across thresholds",
    )


# ── Cross-Case Eligibility ────────────────────────────────────────


class VariableEligibility(BaseModel):
    variable_name: str
    n_cases: int = Field(description="Total number of cases evaluated")
    n_coded: int = Field(description="Cases with non-None coding")
    n_na: int = Field(description="Cases with None (insufficient evidence)")
    n_zero: int = Field(description="Cases coded 0")
    n_one: int = Field(description="Cases coded 1")
    mean_confidence: float = Field(description="Mean coding confidence across cases")
    varies: bool = Field(description="True when at least one 0 and one 1 appear")
    warnings: list[str] = Field(default_factory=list)


class CrossCaseEligibility(BaseModel):
    """Eligibility assessment for cross-case quantitative estimation.

    Computed deterministically from binarizations before CausalQueries is
    called. eligible_for_cq=False blocks the CQ bridge; warnings appear
    even when eligible.
    """

    n_cases: int
    outcome_variable: str
    outcome_n_coded: int = Field(description="Cases with non-None outcome coding")
    outcome_n_zero: int
    outcome_n_one: int
    outcome_varies: bool = Field(
        description="True when outcome has at least one 0 and one 1 across cases",
    )
    variable_checks: list[VariableEligibility]
    n_variables_with_variation: int = Field(
        description="Count of non-outcome variables that vary across cases",
    )
    eligible_for_cq: bool = Field(
        description="True when minimum criteria for CausalQueries are met",
    )
    ineligible_reasons: list[str] = Field(
        description="Specific reasons CQ is blocked (empty when eligible)",
        default_factory=list,
    )
    warnings: list[str] = Field(
        description="Non-blocking issues that may limit estimation quality",
        default_factory=list,
    )


# ── Top-Level Result ──────────────────────────────────────────────


class MultiDocResult(BaseModel):
    causal_model: CausalModelSpec
    case_results: dict[str, str] = Field(
        description="case_id -> path to result.json",
    )
    binarizations: list[CaseBinarization]
    data_frame: list[dict[str, Optional[int]]] = Field(
        description="Binarized data: list of rows (one per case)",
    )
    cq_result: Optional[CausalQueriesResult] = None
    sensitivity: Optional[BinarizationSensitivity] = None
    eligibility: Optional[CrossCaseEligibility] = Field(
        default=None,
        description="Cross-case eligibility assessment; populated whenever binarization completes",
    )
    workflow: str = Field(description="'theory_driven' or 'data_driven'")
