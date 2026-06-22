# Process Tracing Project Dossier

Wiki home: http://localhost:8088/index.php/Project_Wiki

## Goal And Why

Process Tracing is an analyst-methods system for turning historical or source
text into auditable causal analysis: rival explanations, diagnostic evidence,
absence checks, comparative support updates, and inspectable reports.

The portfolio claim is not "LLM summarization." The claim is that an AI system
can help execute a disciplined causal workflow when the workflow forces evidence
records, hypothesis partitions, likelihood-vector testing, deterministic support
updates, source-scope caveats, and report audits into explicit artifacts.

## Portfolio Claim

This is a core CIA-analyst portfolio project. It shows Brian's bridge between
computational social science, AI engineering, and analyst tradecraft:

- method-aware LLM pipeline rather than generic qualitative coding
- structured Pydantic contracts at LLM boundaries
- deterministic Bayesian update and sensitivity logic outside the LLM
- provenance, source-scope, absence-of-evidence, and report-quality controls
- reviewer-facing artifacts that separate workflow evidence from historical
  truth claims

Lead with the workflow and evidence discipline. Do not lead with model choice,
framework mechanics, or claims that the French Directory conclusion is
historically definitive.

## Current Status

Status: active core portfolio project.

The methodology and architecture are strong enough to present as an auditable
inference system. The project is not yet validated as a PhD-level automated
historical analyst. Current SOTA+ work is strengthening source packets, source
coverage, hypothesis partition audit, benchmark repair, and report critique.

Implemented capabilities include source-grounded extraction, hypothesis
generation, coherent evidence-by-hypothesis likelihood vectors, residual
hypothesis handling, dependence pooling, absence evaluation, deterministic
support updates, verdict calibration, HTML report generation, and output-quality
audit. Several validation and source-acquisition surfaces remain partial or
planned.

## Methodology Spine

Start with:

- `docs/METHODOLOGY.md` - concise method and modality map
- `docs/adr/0001_process_tracing_methodology_spine.md` - durable methodology
  decision record
- `docs/PROJECT_THEORY_AND_GOALS.md` - current capability ledger and claim
  discipline
- `docs/WHITEPAPER_optimal_automated_process_tracing.md` - methodology
  north-star
- `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md` - external SOTA map

## Modality Split

Deductive and plan-first parts:

- schema contracts for pipeline outputs
- deterministic Bayesian math and report rendering
- truth-in-labeling around comparative support
- manifest/report publication surfaces

Exploratory and ladder parts:

- source-packet adequacy and source acquisition
- hypothesis partition quality thresholds
- source-production and dependence modeling
- validation benchmark construction and PhD-quality thresholds
- agentic critique tasks routed through `llm_client`

The project should not fake precision on exploratory surfaces. It should expose
readouts, audit failures, and source gaps, then promote stable findings into
contracts and gates.

## Artifacts And Evidence

Read these in order for a portfolio review:

1. `docs/REVIEWER_WALKTHROUGH.md`
2. `docs/portfolio/FRENCH_DIRECTORY_COLLAPSE_CASE_BUNDLE.md`
3. `docs/ARTIFACTS.md`
4. `docs/VALIDATION.md`
5. `docs/CONCERNS.md`

Generated `output/` artifacts are local build products and are not committed.
For a live demonstration, regenerate `result.json` and `report.html` from a
public input text and run the quality audit.

## Known Limits

- The current reviewer case is a public historical example, not classified or
  operational intelligence work.
- The French Directory bundle demonstrates workflow shape, not a settled
  historical finding.
- Source scope, hypothesis partition quality, and dependence modeling still
  constrain academic-strength claims.
- Methodology and report evidence are stronger than product polish.
- Full validation requires multiple cases, frozen benchmark criteria, and
  adversarial/human review.

## Next Slices

1. Finish the active SOTA+ source-packet/source-coverage lane and commit cleanly
   without mixing implementation state into portfolio narrative.
2. Build a regenerated public demo package: command transcript, `result.json`,
   `report.html`, audit output, and a short reviewer path from source text to
   final report.
3. Add hypothesis-partition and benchmark critique readouts before claiming
   stronger methodological validity.

## Concern Register

The portfolio-level concern register is `docs/CONCERNS.md`. The active
implementation concern register is
`docs/plans/sota_plus_concern_register.md`.
