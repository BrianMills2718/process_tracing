# Plan 003 - SOTA+ Execution Master Plan

**Status:** In Progress
**Progress:** Slices 0-1 implemented; Slice 1b source-coverage and verdict-calibration hardening completed with live E2E evidence; Slice 1c adds trace-derived source-acquisition targets so the pipeline can identify which missing evidence would most clarify the process trace.
**Type:** implementation
**Priority:** Critical
**Blocked By:** None
**Blocks:** All SOTA+ implementation slices

---

## Gap

**Current:** Plan 002 defines the SOTA+ recovery model and a high-level
thin-slice roadmap. It does not yet give agents an unambiguous end-to-end
execution contract for every slice.

**Target:** A long-horizon execution plan that makes each slice independently
reviewable, E2E-testable, critique-gated, cleanup-gated, and traceable to the
PhD / think-tank / academic-quality goal.

**Why:** The prior failure mode was not lack of ambition. It was allowing
implementation, report polish, and methodology claims to drift apart. This plan
turns the SOTA+ vision into a repeatable operating loop.

---

## References Reviewed

- `docs/plans/002_sota_plus_recovery_plan.md` - SOTA+ capability ladder and
  recovery postmortem.
- `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md` - external SOTA frontier
  and gaps.
- `docs/PROJECT_THEORY_AND_GOALS.md` - project intent, invariants, and
  implemented/planned ledger.
- `docs/WHITEPAPER_optimal_automated_process_tracing.md` - methodology
  optimum.
- `docs/OUTPUT_QUALITY_RUBRIC.md` - A-grade standard, academic evidence caps,
  and iteration protocol.
- `docs/ontology.md` - analytic object model and report-network semantics.
- `docs/ARCHITECTURE.md` - active design-plan architecture artifact with
  boundary, domain-model, and typed data-flow diagrams.
- `docs/source_packets/18_BRUMAIRE_RESEARCH_DESIGN.md` - first public
  source-packet benchmark candidate and E2E run command.
- `docs/plans/005_interactive_trace_execution_host.md` - planned stage-by-stage
  host for running and inspecting the pipeline interactively.
- `~/projects/.claude/skills/design-plan/SKILL.md` - modality-aware
  planning protocol used to separate specifiable contracts from exploratory
  benchmark calibration, require vertical risk-ordered slices, and keep a live
  concern register.

---

## Design-Plan Modality Diagnosis

This project is hybrid.

**Deductive / plan-first surfaces:** schema contracts, CLI/Make entry points,
`llm_client` routing, provenance requirements, no-direct-subprocess policy,
deterministic Bayesian math, report-audit structure, artifact persistence,
markdown/link/typing checks, and stop/go gates. These can be specified before
implementation and must have tests.

**Exploratory / ladder surfaces:** exact benchmark score thresholds, calibration
of PhD-quality grades across cases, the practical usefulness of the structural
critic, and how much source expansion is enough for different historical
questions. These should not get fake precision. Each needs an instrument:
frozen benchmark cases, audit scorecards, ablations, and concrete failure
examples that can be inspected when a metric moves.

**Hybrid rule:** every exploratory surface must still emit a concrete artifact
that agents and humans can review. Every aggregate score must step down to the
case, evidence item, hypothesis pair, prompt, source, or report section that
caused it.

**Application to this plan:** this is a risk-ordered skeleton, not a fake
precision script for the entire long-term program. Slices 0 and 1 are the next
execution-ready slices. Slices 2-10 define the directional path, risks, and
expected artifacts, but each must be refreshed into an execution-ready child
plan after the prior slice's E2E run, independent critique, and cleanup. Later
slice details may change when earlier benchmark readouts reveal a better order
or a failed assumption.

The live concern register for this plan is
`docs/plans/sota_plus_concern_register.md`. Concerns, audit findings,
ambiguities, and recommendations should go there when they arise rather than
remaining in chat. Each slice boundary must triage the register before the next
slice begins.

The active architecture artifact for deductive surfaces is
`docs/ARCHITECTURE.md`. It is required reading before changing source packets,
the inference pipeline, acquisition planning, report/workbench behavior, or any
cross-component data contract.

---

## Files Affected

This master plan governs later work. It does not itself require production-code
changes.

Expected files touched by future slices:

- `pt/assistant.py`, `pt/cli_assistant.py`, `pt/schemas.py`, `pt/pipeline.py`,
  `pt/pass_*.py`, `pt/bayesian.py`, `pt/report.py`, `pt/multi_pipeline.py`
- `pt/prompts/*.yaml`
- `tests/test_*`
- `docs/source_packets/*`
- `docs/benchmarks/*` or `tests/fixtures/*`
- `docs/OUTPUT_QUALITY_RUBRIC.md`, `docs/ontology.md`, `docs/FUTURE_WORK.md`
- `Makefile`

---

## Universal Slice Contract

No implementation slice is complete until all gates below pass.

| Gate | Required evidence |
|---|---|
| Contract | Typed input/output artifact exists or an existing schema is extended truthfully. Open exploratory fields use broad types and explicit notes rather than fake precision. |
| Unit/integration tests | Deterministic tests cover the new contract, failure mode, and at least one regression case. |
| Live non-mocked E2E | One command exercises the slice on a real process-tracing case with real LLM/provider calls, no mocks, and writes `result.json` plus `report.html` artifacts. Deterministic tests are necessary but insufficient for this gate. |
| Audit/review | `make audit-result` is run against the live generated JSON/report/sidecar and reviewed against `docs/OUTPUT_QUALITY_RUBRIC.md` plus this plan's slice criteria. |
| Independent adversarial critique | A fresh reviewer pass identifies what still fails from a PhD methods standpoint and whether the next step is code, prompt, source, benchmark, or scope. Self-review alone does not satisfy this gate. |
| Concern register triage | New concerns and audit findings are added to `docs/plans/sota_plus_concern_register.md`; every open item is dispositioned as resolved, mitigated, accepted, escalated, or deferred to a named slice. |
| Mockup/notebook gate | Significant UI surfaces and cross-seam contracts include a static mockup or concrete input/output examples before implementation. Non-trivial contract work includes a planning notebook, or the slice plan records an explicit waiver and rationale. |
| Cleanup | Stale docs are updated or archived; generated artifacts stay out of git unless intentionally curated; `make check` passes. |
| Commit | Verified work is committed before moving to the next slice. |

Stop rule: do not advance to a downstream slice when the current slice reveals
that source scope, hypothesis partition, benchmark expectations, or artifact
contracts are underspecified. Update this plan or the relevant slice plan first.

Acceptable independent critique sources, in descending preference: a separate
agent run through the agentic assistant harness, a Claude/Codex second-opinion
pass through `llm_client`, a human review, or a deliberately separate
adversarial run with a new trace and written limitation that no external
reviewer was available. The critique artifact must be saved or summarized in
the slice notes before cleanup.

Each slice handoff must update the next 1-2 slices from directional skeleton to
execution-ready detail using this shape:

```text
Slice N - <one-line outcome>
  advances:        capability-ladder row and long-term goal step
  vertical scope:  smallest end-to-end demonstrable behavior
  de-risks:        unknown or boundary attacked by doing this now
  success:         deductive test/gate or exploratory readout
  review focus:    what the independent adversarial pass should try to break
  cleanup:         debt/docs/refactor to clear before Slice N+1
  done-when:       test/readout passes; audit findings dispositioned; cleanup
                   done; concern register triaged
```

---

## Master E2E Commands

These commands are the baseline harness. Slices may add commands, but they may
not remove these as release gates unless the plan is updated.

```bash
make check
```

```bash
python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir /tmp/pt_smoke/brumaire_packet \
  --research-question "Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup and the Consulate rather than a stable parliamentary republic, a revived Jacobin-dominated republic, or a royalist restoration?" \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --max-budget 1.0

make audit-result RESULT=/tmp/pt_smoke/brumaire_packet/result.json \
  REPORT=/tmp/pt_smoke/brumaire_packet/report.html \
  FOCAL_YEAR=1799
```

```bash
python -m pt input_text/revolutions/french_revolution.txt \
  --output-dir /tmp/pt_smoke/french_revolution \
  --research-question "Why did the French Revolution radicalize?" \
  --max-budget 1.0
```

Cross-case gates begin only after Slice 9:

```bash
python -m pt.multi input_text/revolutions/french_revolution.txt \
  input_text/american_revolution/american_revolution.txt \
  --output-dir /tmp/pt_smoke/multi_case \
  --skip-cq
```

---

## Slice Roadmap

Slices 0-1 are execution-ready. Slices 2-10 are risk-ordered directional
slices: keep their goals and gates, but refresh their implementation details
after the previous slice's E2E readout and independent critique.

### Slice 0 - Agentic Assistant Harness

**Goal:** Add the narrow assistant surface needed to make process-tracing labor
agent-drivable through `llm_client`, not through direct Codex/Claude Code glue.

**Implementation scope:**

- Add a process-tracing assistant wrapper over `llm_client`
  `execution_mode="workspace_agent"`.
- Select backend by config/CLI from `codex*` or `claude-code*`.
- Emit a typed assistant artifact for source-packet drafting.
- Preserve backend, cwd, `task`, `trace_id`, `max_budget`, prompt/spec
  reference, and output path.

**E2E test:** run the assistant on the Brumaire source-packet task and validate
the typed artifact without requiring the downstream pipeline to accept it yet.

**Review/critique:** inspect whether the artifact is a research product or only
a transcript. It must name sources, dates, genres, observability expectations,
rival interpretations, and gaps.

**Cleanup:** add or update Make help; verify no direct `codex`, `claude-code`,
provider SDK, or assistant subprocess dependency appears in `pt/`.

**Success criteria:** a reviewer can rerun the assistant task and see a typed,
budgeted, observable source-packet draft.

### Slice 1 - Source Packet Contract And Benchmark

**Status:** Implemented in the Slice 1 source-packet contract commit.

**Goal:** Make source scope explicit before inference.

**Implementation scope:**

- Define a source-packet schema or document contract.
- Upgrade the Brumaire packet into the first benchmark packet.
- Preserve source group, date coverage, genre, reliability/limitations,
  observability expectations, and missing-source gaps.
- Allow the pipeline/report audit to distinguish corpus caps from model/report
  failures.

**E2E test:** run the Brumaire packet command in this plan, then run
`make audit-result`.

**Review/critique:** judge whether multiple source groups actually affect
extraction, top drivers, absence interpretation, and synthesis caveats.

**Cleanup:** archive or update any source-scope docs that imply a single-text
demo can support PhD-grade claims.

**Success criteria:** source-scope caps in the report name concrete missing
source classes, or clear because the packet covers the needed source genres for
the stated question.

**Implemented artifacts:** `pt/source_packet.py`, `--source-packet`,
`make source-packet-run`, `docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json`,
`ProcessTracingResult.source_packet`, `ProcessTracingResult.source_coverage`,
report source-packet and source-coverage tables, and audit source-scope cap
distinctions.

**Active hardening slice:** Pass 1 extraction must receive the accepted
source-packet contract before hypothesis generation. For every accepted source
whose exact marker appears in the assembled input, extraction should produce
source-specific evidence when the source contains relevant causal,
institutional, temporal, agency, mechanism, constraint, interpretive, or
absence-relevant traces. Legal or constitutional source genres count as
evidence-bearing when they specify institutional powers, constraints,
succession rules, office design, or procedural veto points. This does not
complete Slice 3 metadata modeling; it is the minimal bridge needed so Slice 1
coverage reports reflect extraction behavior rather than only downstream
auditing.

### Slice 2 - Research Question And Hypothesis Partition Gate

**Goal:** Freeze the focal question, outcome window, rival hypotheses, residual,
and pairwise discriminators before testing.

**Implementation scope:**

- Add a partition artifact before Pass 3.
- Flag broad, overlapping, complementary, or absorptive hypotheses.
- Require at least one discriminator for each important rival pair or cap the
  output.

**E2E test:** run Brumaire and French Revolution cases and verify the partition
artifact is persisted in `result.json` or a sidecar JSON.

**Review/critique:** adversarially ask whether the winning hypothesis could
absorb the rivals, whether the residual is meaningful, and whether the focal
window matches the outcome.

**Cleanup:** update prompts/docs so hypothesis generation and review language
matches the partition contract.

**Success criteria:** downstream testing cannot proceed silently with a broad
or overlapping hypothesis menu.

### Slice 3 - Extraction Provenance And Source Metadata

**Goal:** Raise extraction from quote preservation to source-aware trace
production.

**Implementation scope:**

- Extend evidence/event metadata with source group, source genre, approximate
  date confidence, and trace-production relevance.
- Keep raw source quote/provenance intact.
- Avoid using source metadata as likelihood evidence until a later explicit
  method gate.

**E2E test:** Brumaire source groups appear in extracted evidence and report
triage; source hashes still protect `--from-result`.

**Review/critique:** sample top drivers and missing traces to ensure they can
be traced back to source group and genre.

**Cleanup:** update ontology and report labels if new metadata changes network
interpretation.

**Success criteria:** a reader can tell which sources produced the evidence
that moved support and which source genres could or could not reveal missing
traces.

### Slice 4 - Diagnostic Test Matrix

**Goal:** Turn likelihood scoring into explicit Van-Evera-style diagnostic
tests against rival hypotheses.

**Implementation scope:**

- Represent hoop, smoking-gun, straw-in-the-wind, and discriminator tests as
  structured artifacts.
- Tie each important rival pair to at least one proposed discriminator or cap
  the grade.
- Preserve coherent likelihood vectors; do not return to pairwise LRs.

**E2E test:** Brumaire report shows at least one targeted discriminator per
central rival pair or explains the cap.

**Review/critique:** judge whether high-impact evidence is actually diagnostic
or merely background/context.

**Cleanup:** update rubric caps and report audit wording to use the new matrix.

**Success criteria:** an A-level claim is impossible without displayed,
source-grounded discriminators.

### Slice 5 - Source-Lineage Dependence Benchmark

**Goal:** Prevent raw-count inflation from duplicate, shared-source, or
same-event evidence.

**Implementation scope:**

- Add planted duplicate/shared-source fixtures.
- Extend dependence artifacts toward source lineage and per-hypothesis
  redundancy where feasible.
- Keep deterministic Bayesian behavior tested in `pt/bayesian.py`.

**E2E test:** a fixture demonstrates that duplicate/shared-source items produce
lower effective evidence than independent corroboration.

**Review/critique:** inspect whether dependence decisions are explained at the
source/evidence level rather than hidden in one scalar.

**Cleanup:** update report triage and ontology if dependence categories change.

**Success criteria:** duplicate evidence cannot materially inflate support
without the audit showing why it was treated as independent.

### Slice 6 - Observability-Weighted Absence

**Goal:** Treat absence as source silence with observability, not as proof of
world absence.

**Implementation scope:**

- Add source-genre observability bands and expected source locations for
  missing traces.
- Keep absence outside the Bayesian update unless a separate explicit method
  gate is approved.
- Show where a missing trace should have appeared and why.

**E2E test:** Brumaire absence findings name the source genre where each missing
trace would be expected.

**Review/critique:** check damaging absence claims for genre overreach.

**Cleanup:** revise report/audit language so absence caps are clear and do not
overstate what the corpus can prove.

**Success criteria:** absence findings improve research guidance without
silently becoming likelihood evidence.

### Slice 7 - Structural Critic Ablation

**Goal:** Use causal graphs as qualitative critics of likelihood claims, not as
unparameterized likelihood calculators.

**Implementation scope:**

- Add a critic pass that flags confounds, missing pathways, void links,
  too-strong likelihood claims, and confirmed links.
- Route any numeric change through re-elicitation.
- Add on/off configuration and ablation capture.

**E2E test:** run Brumaire with critic off and on; preserve both results and the
critic delta.

**Review/critique:** judge whether the critic catches planted confounds or only
adds generic warnings.

**Cleanup:** report only critic findings that can be traced to evidence,
hypotheses, or causal edges.

**Success criteria:** the critic improves at least one frozen benchmark failure
without creating unsupported likelihood magnitudes.

### Slice 8 - Synthesis Calibration And Report Regression

**Goal:** Ensure narrative conclusions cannot outrun evidence, support,
robustness, source scope, or diagnosticity.

**Implementation scope:**

- Strengthen report audit gates for overclaim language.
- Add regression fixtures for report sections, network semantics, and evidence
  triage.
- Preserve isolated-node disclosure and temporal layout interpretation.

**E2E test:** generated report and audit fail when synthesis overstates weak or
fragile support.

**Review/critique:** read the report as a hostile PhD reviewer and identify the
strongest overclaim still present.

**Cleanup:** remove stale report terminology and archive superseded screenshots
or generated examples.

**Success criteria:** report polish cannot increase the grade unless the
underlying inference artifact supports it.

### Slice 9 - Cross-Case Eligibility And Quantitative Bridge

**Goal:** Integrate process tracing with cross-case/quantitative methods only
when the data support it.

**Implementation scope:**

- Add eligibility checks for case comparability, outcome variation,
  measurement variation, and variable support before CausalQueries/QCA/stat
  handoff.
- Preserve the distinction between single-text comparative support and
  cross-case estimands.
- Feed cross-case results back into proposed within-case traces.

**E2E test:** `python -m pt.multi ... --skip-cq` runs, and CausalQueries is
blocked or allowed based on explicit eligibility artifacts.

**Review/critique:** inspect whether the bridge is learning from cases or merely
formalizing arbitrary binary codings.

**Cleanup:** update multi-case docs/report language to prevent single-case
causal-effect claims.

**Success criteria:** cross-case integration is agent-drivable and gated, and it
does not launder weak within-case evidence into quantitative certainty.

### Slice 10 - Frozen Benchmark Suite And Graduation Gate

**Goal:** Make SOTA+/A-level claims benchmark-backed rather than conversational.

**Implementation scope:**

- Build a small public benchmark suite with expected critique targets,
  planted failure modes, and score caps.
- Track calibration/discrimination metrics and ablation results.
- Define what counts as PhD-review-ready under available corpus versus blocked
  by external evidence.

**E2E test:** one command runs all frozen benchmarks, records scores, and links
every failure to a concrete artifact.

**Review/critique:** compare benchmark outputs to Plan 002 SOTA frontier and
the output rubric; update thresholds only with evidence from cases.

**Cleanup:** archive obsolete benchmarks and remove any docs that claim SOTA+
without benchmark evidence.

**Success criteria:** no "A", "optimal", or SOTA+ claim is possible without a
passing benchmark record and an explanation of remaining external-data limits.

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|---|---|---|
| `tests/test_assistant.py` | | Slice 0 assistant harness contract, artifact persistence, dependency boundary, CLI errors, and opt-in live smoke. |
| `tests/test_source_packet.py` | | Slice 1 source-packet loading, assistant-artifact compatibility, and summary metadata. |
| tests/test_source_packet.py | test_source_packet_summary_preserves_scope_metadata | Source-gap disposition summaries preserve unresolved high-priority gap status. |
| `tests/test_source_coverage.py` | | Slice 1 packet-source marker coverage for input text, extracted evidence, missing sources, and unconfigured sources. |
| `tests/test_extraction_quality.py` | | Slice 1b extraction prompt/schema preserve source markers and source-packet marker coverage so packet coverage can be measured on live output. |
| tests/test_extraction_quality.py | test_extraction_contract_preserves_source_markers_in_prompt_and_schema | Source-marker preservation prompt/schema regression. |
| tests/test_extraction_quality.py | test_extraction_contract_uses_source_packet_for_marker_coverage | Source-packet marker coverage prompt regression. |
| tests/test_pipeline_integration.py | TestReportConsistency::test_source_packet_context_reaches_extraction_pass | Source-aware extraction regression: the accepted source packet reaches Pass 1 before hypothesis generation. |
| tests/test_pipeline_integration.py | TestReportConsistency::test_source_packet_is_visible_in_report_and_audit | Report/audit expose source-gap dispositions and unresolved high-priority gap count. |
| tests/test_source_acquisition.py | test_acquisition_plan_prioritizes_unresolved_source_gaps_and_absences | Trace-derived acquisition agenda ranks unresolved source gaps and damaging absences ahead of lower-value corroboration. |
| tests/test_source_acquisition.py | test_source_acquisition_cli_writes_json_plan | Agent-drivable CLI writes a machine-readable acquisition plan from `result.json` and source packet context. |
| tests/test_workbench.py | test_workbench_http_exposes_button_and_json_endpoint | Local workbench exposes a click-to-enrich UI and JSON endpoint for the same acquisition flow. |
| tests/test_architecture_docs.py | test_architecture_doc_has_required_design_plan_diagrams | Active architecture doc preserves boundary, domain-model, and data-flow Mermaid diagrams. |
| tests/test_pipeline_integration.py | TestVectorCompleteness::test_repairs_overlapping_clusters_once_with_validation_feedback | Live E2E regression: Pass 3 makes one explicit validation-repair call when dependence clusters overlap, then still fails loud if invalid. |
| `tests/test_cli_source_packet.py` | | Slice 1 CLI `--source-packet` plumbing without an LLM call. |
| `tests/test_pass_refine.py` | | Regression coverage for the live Slice 1 refinement failure: Pass 5 must not put evidence-to-hypothesis support links into causal edges. |
| `tests/test_verdict_calibration.py` | | Slice 1b deterministic synthesis status calibration against computed posteriors. |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|---|---|
| `tests/test_pt_llm.py` | Existing LLM boundary behavior remains intact. |
| `tests/test_pipeline_integration.py` | Pipeline/report integration, source-packet prompt context, result metadata, report visibility, and audit cap behavior remain covered. |

## Required Tests By Slice

| Slice | Deterministic tests | E2E artifact |
|---|---|---|
| 0 | assistant wrapper contract; no direct assistant subprocess imports/calls | typed source-packet assistant artifact |
| 1 | source-packet schema/validator; corpus-cap audit behavior | Brumaire packet result + audit |
| 2 | partition artifact validation; broad/overlap failure cases | partition sidecar or `result.json` section |
| 3 | provenance metadata validation; source hash regression | evidence inventory with source groups |
| 4 | diagnostic matrix validation; discriminator cap | report diagnostic matrix |
| 5 | duplicate/shared-source pooling fixture | dependence benchmark output |
| 6 | observability-band validation; damaging absence cap | absence observability report section |
| 7 | critic categories and on/off ablation capture | critic delta artifact |
| 8 | report/audit regression tests | report + audit failure/pass examples |
| 9 | cross-case eligibility gates | multi-case eligibility artifact |
| 10 | benchmark runner and scorecard schema | benchmark scorecard |

---

## Per-Slice Review Checklist

Use this checklist after each E2E run.

- Does the slice move a named row in the Plan 002 capability ladder?
- Did the E2E command exercise the new behavior rather than only unit tests?
- Is there a typed artifact a reviewer can inspect without reading code?
- Can every aggregate score or warning step down to a source, evidence item,
  hypothesis pair, case, prompt, or report section?
- Did the output improve process-tracing validity, or only presentation?
- Who performed the independent adversarial critique, and where is the critique
  artifact or summary?
- Were all new concerns and critique findings added to
  `docs/plans/sota_plus_concern_register.md` and dispositioned?
- What is the strongest PhD-level critique remaining?
- Is the next action code, prompt, source collection, benchmark design, or
  documentation cleanup?
- Are any docs now stale or overclaiming?
- Did the next 1-2 slice specs get refreshed based on the observed readout?

---

## Definition Of Long-Term Success

The long-term goal is met only when all of the following are true:

- Source packets, hypothesis partitioning, diagnostic testing, dependence,
  absence, structural critique, synthesis, and cross-case eligibility are
  agent-drivable and typed.
- The report and JSON can be reviewed as a transparent process-tracing artifact,
  not merely an LLM essay.
- Benchmark cases demonstrate that the system catches known academic failure
  modes: weak source scope, broad hypotheses, background top drivers, duplicate
  evidence, absence overreach, causal confounds, fragile winners, and cross-case
  ineligibility.
- A-level/PhD-quality claims are tied to benchmark records and source-scope
  limits, not visual polish.
- Humans direct research questions and adjudicate high-stakes interpretation,
  but routine process-tracing labor is decomposed into auditable agent and
  deterministic steps.

---

## Acceptance Criteria For This Plan

- [x] Plan 003 is linked from the plan index and active roadmap docs.
- [x] Plan 003 defines slice order, dependencies, E2E tests, critique gates,
  cleanup gates, and success criteria.
- [x] Plan 003 distinguishes execution-ready next slices from directional
  future skeleton slices to avoid fake precision.
- [x] Plan 003 has a live concern register and makes register triage part of
  every slice's definition of done.
- [x] Future implementation work uses this plan as the stop/go checklist.
- [x] `make check` passes after documentation updates.
