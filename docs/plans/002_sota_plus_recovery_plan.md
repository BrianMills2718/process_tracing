# Plan 002 - SOTA+ Recovery And Thin-Slice Operating Model

**Status:** Planned
**Created:** 2026-06-22
**Context:** The project has a strong methodology north star, but prior work let
documentation, report polish, and implementation slices drift away from a
benchmark-enforced SOTA+ path.

## Goal

Recover the SOTA+ vision and make it operational. The project should automate
process tracing at PhD / think-tank / academic quality by exceeding current
practice, not merely digitizing it. Current practice is constrained by manual
qualitative labor and limited technical mixed-methods implementation; this repo
should treat those as bottlenecks to remove.

## Required Reading

- `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md`
- `docs/PROJECT_THEORY_AND_GOALS.md`
- `docs/WHITEPAPER_optimal_automated_process_tracing.md`
- `docs/OUTPUT_QUALITY_RUBRIC.md`
- `docs/ontology.md`

## Postmortem

| Failure | What happened | Prevention gate |
|---|---|---|
| Vision not enforced | SOTA+ appeared in docs but not as a slice-level acceptance gate. | Every plan must name the SOTA frontier it advances and the benchmark/failure mode it tests. |
| Presentation substituted for inference | Report/network polish improved inspectability but did not always improve source scope, hypothesis design, diagnosticity, or validation. | A UI/report slice cannot claim quality improvement unless it changes an inference check, exposes a failure mode, or blocks an overclaim. |
| No frozen benchmark | Grades and "optimal" claims were too easy to inflate. | Maintain frozen benchmark cases with expected critique targets and caps. |
| Current/future drift | Methodology optimum, current code, and old sprint notes blurred together. | Active docs must distinguish implemented / partial / planned; stale docs go to `docs/archive/`. |
| Thin slices lost north-star coupling | Small implementation steps were not always tied to a named SOTA capability. | Each slice must update the capability ladder and add a deterministic or audit check. |

## SOTA+ Capability Ladder

Each capability has to be independently testable. A slice can be small, but it
must move one row upward.

| Capability | Current state | SOTA+ target | Gate |
|---|---|---|---|
| Source packet | Manual/ad hoc source selection; report can be capped by source scope. | Agent builds primary/secondary source packet with provenance, dates, genre, reliability, and coverage map. | Source packet schema + one benchmark packet. |
| Research question/focal window | LLM can choose, user can pin. | Frozen question, focal decision window, outcome, and scope before testing. | Partition artifact exists before Pass 3. |
| Hypothesis partition | Prompt rules + optional review. | MECE-ish rival set, explicit residual, split/merge audit, pairwise discriminators. | Partition audit blocks broad/overlapping hypotheses. |
| Extraction/provenance | Source-grounded evidence and source hash. | Evidence/event/mechanism/source metadata with quote, date confidence, source genre, and trace-production relevance. | Extraction fixture checks provenance completeness. |
| Diagnostic testing | Coherent likelihood vectors and Van Evera labels. | Pre-specified hoop/smoking-gun/discriminator tests with source-scope assumptions. | Diagnostic matrix shows at least one targeted discriminator per rival pair or caps grade. |
| Dependence | Scalar dependence clusters. | Source-lineage graph, per-hypothesis redundancy, partial pooling, planted duplicate defense. | Duplicate/shared-source benchmark prevents raw-count inflation. |
| Absence/observability | Qualitative absence only. | Source-silence model with observability bands and genre expectations. | Missing trace includes where it should appear and why. |
| Structural critic | Not built. | Local causal graph critic flags confounds, missing pathways, and too-strong likelihood claims; estimator re-elicits magnitude. | Audit-on/off ablation exists. |
| Synthesis calibration | Synthesis plus PhD audit caps. | Synthesis cannot outrun support, robustness, source scope, or diagnosticity. | Report audit blocks overclaim language. |
| Cross-case bridge | `pt.multi` + CausalQueries bridge. | Eligibility gates, case comparability checks, and iterative process tracing -> quantitative feedback. | CausalQueries only runs when variation/measurement gates pass. |
| Validation | Software tests and report audit. | Frozen benchmark suite with expected critique targets, calibration/discrimination metrics, and ablations. | No "A" or SOTA+ claim without benchmark result. |

## Plan

Execute the recovery in thin slices, beginning with the source-packet contract.
Each slice must update the SOTA+ capability ladder, add or revise a benchmark
fixture/audit artifact, and leave `make check` green. Do not start a downstream
slice if the current slice reveals that the source packet, hypothesis partition,
or benchmark acceptance criteria are underspecified.

## Thin-Slice Roadmap

### Slice 1 - Source Packet Contract

Build a source-packet schema and one benchmark packet for the Brumaire/French
Directory case. Include source type, date coverage, expected observability, rival
secondary interpretations, and known gaps.

Acceptance:

- `docs/source_packets/18_BRUMAIRE_RESEARCH_DESIGN.md` is either upgraded or a
  new packet is created with the schema.
- Pipeline/report audit can distinguish "report/model failure" from "source
  corpus cap."
- `make check` passes.

### Slice 2 - Hypothesis Partition Audit

Add a pre-testing artifact that freezes research question, focal window,
hypotheses, residual, split/merge warnings, and pairwise discriminators.

Acceptance:

- Broad or complementary hypotheses are flagged before Pass 3.
- Each rival pair has at least one proposed discriminator or the output is capped.
- Review artifact is preserved in `result.json` or a sidecar JSON.

### Slice 3 - Source-Lineage Dependence Benchmark

Build a deterministic fixture with duplicate/shared-source evidence and assert
the update cannot treat those items as independent corroboration.

Acceptance:

- Benchmark demonstrates lower effective evidence count for duplicates.
- Report explains why evidence was pooled or left independent.
- `make check` includes the benchmark.

### Slice 4 - Observability-Weighted Absence Prototype

Extend absence findings with source-genre observability: where a missing trace
should appear, why this corpus should contain it, and how damaging the omission
is under each hypothesis.

Acceptance:

- Absence entries include observability rationale and expected source location.
- Bayesian update remains unchanged unless a separate explicit method gate is
  passed.

### Slice 5 - Structural Critic Ablation Hook

Implement a cheap top-driver structural critic that flags confounds,
too-strong likelihoods, void links, and confirmed links. The critic changes
numbers only through re-elicitation, not direct graph computation.

Acceptance:

- Pipeline can run critic on/off.
- Report shows critic decisions and changed likelihoods.
- Benchmark records whether the critic catches planted confounds.

## Operating Rules Going Forward

- No SOTA+ claim without a named external SOTA baseline and a benchmark result.
- No "optimal" claim unless remaining caps are external-data limits rather than
  model/report failures.
- No quality-improvement slice without one of: source scope, hypothesis
  partition, diagnosticity, dependence, observability, structural critique,
  cross-case eligibility, or validation.
- Human review is useful for accountability and benchmark construction, but it
  is not treated as the permanent ceiling on qualitative inference quality.
- Every slice must be agent-drivable through CLI/API/tests, not only readable in
  the HTML report.

## Immediate Next Slice

Start with **Slice 1 - Source Packet Contract**. It has the highest leverage
because weak source scope poisons every later step and currently imposes the most
frequent academic cap.
