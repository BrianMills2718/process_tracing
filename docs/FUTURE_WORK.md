# Future Work

This is the current roadmap for `process_tracing`. It records work that remains
after the inference-core rebuild and report-audit work. Historical phase plans and
superseded status reports live under `docs/archive/`.

Current SOTA+ recovery source of truth:

- `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md`
- `docs/plans/002_sota_plus_recovery_plan.md`
- `docs/plans/003_sota_plus_execution_master_plan.md`

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
- Slice 0 agentic assistant harness for typed source-packet drafts through
  `llm_client` `workspace_agent`; deterministic contract tests are in place and
  live provider smoke is opt-in with `PT_RUN_LIVE_AGENT_TESTS=1`.
- Slice 1 source-packet contract accepted by the pipeline via `--source-packet`;
  result/report/audit outputs preserve source counts, source kinds, high-priority
  gaps, packet limitations, and the rule that packet metadata is not evidence.
- Source-packet coverage verification: packet sources can define exact
  `text_markers`; `result.json`, report, and `make audit-result` show which
  packet sources appear in the input text, which produced extracted evidence,
  and which evidence remains unassigned.
- Deterministic verdict calibration: synthesis status labels are downgraded
  when they overstate computed comparative support; low-posterior hypotheses can
  still receive steelman reasoning, but not a misleading "supported" label.

## Highest-Value Next Work

| Priority | Work | Why it matters | Acceptance check |
|---:|---|---|---|
| 1 | Hypothesis partition audit | Broad, overlapping, or complementary hypotheses still undermine comparative support. | A review artifact freezes the research question, hypothesis menu, residual, and pairwise discriminators before testing. |
| 2 | Dependence and trace-production upgrade | Scalar dependence clusters reduce double-counting but do not model per-hypothesis redundancy, solicitation, preservation, false-positive channels, or shared model error. | Planted duplicate/source-lineage tests plus a report section showing why evidence was pooled or left independent. |
| 3 | Observability-weighted absence | Absence findings are qualitative and excluded from the update. | Missing predicted traces carry source-genre observability bands and remain clearly separated from evidence of world-absence. |
| 4 | Source acquisition and missing-source resolution | The packet is now an accepted contract and coverage is reported, but the tool does not yet acquire missing source classes such as private correspondence. | The input corpus is assembled or extended from packet sources with source-level provenance, and high-priority packet gaps are resolved or explicitly accepted. |
| 5 | Auditor ablation benchmark | Architecture is auditable, but methodological validity is not yet empirically demonstrated. | Frozen benchmark cases compare narrative-only, dependence-pooling, and audit-enabled variants with calibration/discrimination metrics. |

For the current Brumaire benchmark, the active audit blocker is **source
acquisition and high-priority missing-source resolution**: the packet-source
coverage report is complete for accepted sources, but the packet still declares
an unresolved high-priority private-correspondence gap. For the broader
architecture roadmap, the next planned method slice remains the hypothesis
partition audit from Plan 003. Every slice must include live non-mocked E2E
testing, review/critique, cleanup, and a commit gate before the next slice
starts.

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
- Add source acquisition and missing-source resolution on top of the accepted
  source-packet contract and coverage report.
- Keep agent harness integration behind `llm_client`; this repo should not call
  Codex, Claude Code, provider SDKs, or assistant subprocesses directly.
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
