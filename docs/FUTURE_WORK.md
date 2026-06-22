# Future Work

This is the current roadmap for `process_tracing`. It records work that remains
after the inference-core rebuild and report-audit work. Historical phase plans and
superseded status reports live under `docs/archive/`.

## Current Baseline

Implemented in the reference pipeline:

- LLM-first extraction, hypothesis generation, testing, absence review,
  synthesis, and optional refinement.
- Coherent evidence-by-hypothesis likelihood vectors.
- Deterministic Bayesian support update in log space.
- Researcher priors, prior sensitivity, LR sensitivity, and robustness labels.
- Explicit residual hypothesis in the update.
- Interpretive-evidence caps and relevance gating.
- Dependence clusters with scalar partial pooling.
- Source-text hash validation for `--from-result`.
- PhD-style output audit, evidence triage, and `make audit-result`.
- Temporal report view and interactive causal network with top-driver,
  background-driver, additional-link, temporal-conflict, and isolated-node
  controls.
- Cross-case path through `pt.multi` and the CausalQueries bridge.

## Highest-Value Next Work

| Priority | Work | Why it matters | Acceptance check |
|---:|---|---|---|
| 1 | Source-packet workflow | The current pipeline can be academically capped by single-text or broad-overview source scope. It needs an agent-drivable way to assemble primary sources, rival secondary accounts, dates, and source-genre metadata before running inference. | A documented source packet can be passed to the pipeline; the report audit distinguishes corpus limits from report/model failures. |
| 2 | Hypothesis partition audit | Broad, overlapping, or complementary hypotheses still undermine comparative support. | A review artifact freezes the research question, hypothesis menu, residual, and pairwise discriminators before testing. |
| 3 | Dependence and trace-production upgrade | Scalar dependence clusters reduce double-counting but do not model per-hypothesis redundancy, solicitation, preservation, false-positive channels, or shared model error. | Planted duplicate/source-lineage tests plus a report section showing why evidence was pooled or left independent. |
| 4 | Observability-weighted absence | Absence findings are qualitative and excluded from the update. | Missing predicted traces carry source-genre observability bands and remain clearly separated from evidence of world-absence. |
| 5 | Auditor ablation benchmark | Architecture is auditable, but methodological validity is not yet empirically demonstrated. | Frozen benchmark cases compare narrative-only, dependence-pooling, and audit-enabled variants with calibration/discrimination metrics. |

## Methodology Extensions

- **Qualitative structural critic:** implement the white paper's local causal-graph
  critic as a categorical, directional audit of posterior-moving evidence. The
  graph should identify missing pathways, confounds, and too-strong likelihood
  claims; it should not compute likelihood magnitudes without parameters.
- **Prior provenance audit:** record which sources informed researcher priors and
  prevent the same material from being counted again as likelihood evidence.
- **Cap/floor sensitivity grid:** report whether rankings survive alternative
  LR caps and floors, not only top-driver perturbations.
- **Cross-case eligibility checks:** before invoking CausalQueries, enforce case
  comparability and outcome/covariate variation so the formal path is not fed an
  all-ones design.
- **Quantitative feedback loop:** use cross-case findings to propose sharper
  within-case traces, and use process-tracing outputs to refine cross-case model
  structure and measurement.

## Product and Operations Work

- Add agent-drivable JSON endpoints for report inspection and audit results.
- Make source-packet construction a CLI/Make target rather than an ad hoc doc.
- Preserve run metadata: model, prompts, priors, source hashes, audit version,
  and commit SHA in `result.json`.
- Add report-regression fixtures for network layout and audit sections.
- Build a small curated benchmark set with public sources and expected critique
  targets.

## Not In Scope For This Repo

- Generic qualitative coding or thematic coding workflows. Those belong in the
  separate qualitative-coding project.
- Claims of identified causal effects from one text. Single-text output remains
  comparative support over a stated hypothesis set.
- Treating human judgment as the only route to rigor. Human review is valuable
  for direction, accountability, and validation; routine process-tracing labor
  should be decomposed into auditable agent and deterministic steps.
