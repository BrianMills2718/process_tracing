"""Data contracts for multi-document cross-case analysis."""

from __future__ import annotations

from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ── Causal Model Specification ─────────────────────────────────────


class CausalVariable(BaseModel):
    name: str = Field(description="Valid R identifier, e.g. 'fiscal_crisis'")
    description: str = Field(description="What 1 means")
    description_zero: str = Field(description="What 0 means")
    observable_indicators: list[str] = Field(
        default_factory=list,
        description="Concrete things to look for in texts",
    )


class CausalEdgeSpec(BaseModel):
    parent: str
    child: str


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
        names = set(self.variable_names)

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
    variable_name: str
    value: Optional[int] = Field(
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

    def to_row(self) -> dict[str, Optional[int]]:
        """Convert to a flat dict suitable for a data frame row."""
        return {c.variable_name: c.value for c in self.codings}


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
    workflow: str = Field(description="'theory_driven' or 'data_driven'")
