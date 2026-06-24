# Process Tracing Artifact Register

Wiki home: http://localhost:8088/index.php/Project_Wiki

This register lists portfolio evidence, not every local file. The question for
each artifact is what claim it supports and how a reviewer should inspect it.

| Artifact | Type | Path | Claim Supported | Audience | Status | Verification |
|---|---|---|---|---|---|---|
| Project dossier | Dossier | `PROJECT.md` | Concise entrypoint for goals, portfolio claim, status, limits, and next slices. | Portfolio reviewer, agent | current | Markdown link check. |
| Reviewer walkthrough | Portfolio guide | `docs/REVIEWER_WALKTHROUGH.md` | Shows the intended review path and analyst-facing differentiator. | Portfolio reviewer | current | Published through `docs/wiki_manifest.yaml`. |
| French Directory case bundle | Portfolio evidence | `docs/portfolio/FRENCH_DIRECTORY_COLLAPSE_CASE_BUNDLE.md` | Demonstrates rival hypotheses, evidence records, absence checks, and support updates on public historical material. | CIA/analyst reviewer | current, compact excerpt | Regeneration command documented; generated `output/` files are local-only. |
| Methodology whitepaper | Whitepaper | `docs/WHITEPAPER_optimal_automated_process_tracing.md` | Explains the quality-optimal architecture and the methodological rationale. | Methodology reviewer | current north-star | Draft status; not itself an empirical validation. |
| Theory/goals ledger | Project doc | `docs/PROJECT_THEORY_AND_GOALS.md` | Separates implemented, partial, planned, and not-claimed capabilities. | Agent, technical reviewer | current but actively edited | Read with git status; active SOTA+ work may update it. |
| Methodology dossier | Methodology | `docs/METHODOLOGY.md` | Maps goals, borrow-vs-build, modality split, failure modes, and promotion rule. | Portfolio reviewer, agent | current | Markdown link check. |
| Methodology ADR | ADR | `docs/adr/0001_process_tracing_methodology_spine.md` | Records the decision to frame this as process-tracing methodology infrastructure. | Agent, reviewer | current | Markdown link check. |
| Output quality rubric | Validation guide | `docs/OUTPUT_QUALITY_RUBRIC.md` | Defines report-quality scoring and academic evidence caps. | Methodology reviewer | current | Used by `make audit-result`. |
| Ontology | Project doc | `docs/ontology.md` | Defines analytic objects and report-network semantics. | Technical reviewer | current | Published through wiki manifest. |
| SOTA review | Research synthesis | `docs/research/PROCESS_TRACING_SOTA_REVIEW_2026.md` | Maps external SOTA and the SOTA+ opportunity. | Technical/methodology reviewer | current as of 2026-06-22 | Links to primary external references. |
| SOTA+ recovery plan | Plan | `docs/plans/002_sota_plus_recovery_plan.md` | Defines thin-slice operating model and recovery gates. | Agent, reviewer | in progress context | Plan index tracks status. |
| SOTA+ execution plan | Plan | `docs/plans/003_sota_plus_execution_master_plan.md` | Long-horizon SOTA+ implementation roadmap. | Agent | active, dirty during this dossier slice | Do not treat as final until committed. |
| Concern registers | Risk register | `docs/CONCERNS.md`; `docs/plans/sota_plus_concern_register.md` | Keeps portfolio and implementation risks out of chat. | Agent, reviewer | current | Triage at slice boundaries. |
| Evidence notes | Evidence bundle | `evidence/current/Evidence_Plan003*.md` | Shows active SOTA+ implementation and validation evidence records. | Technical reviewer | current | Published by glob in wiki manifest; historical evidence lives in `~/archive/process_tracing/raw/repo-historical-evidence-2026-06-24/`. |
| Local result/report | Generated output | `output/<run>/result.json`; `output/<run>/report.html` | Demonstrates the actual pipeline output shape. | Demo reviewer | local-only, regenerate before sharing | Run `python -m pt ...`, then `make audit-result RESULT=... REPORT=...`. |
| Source-packet examples | Source design | `docs/source_packets/18_BRUMAIRE_RESEARCH_DESIGN.md`; `docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json` | Demonstrates source-scope design direction. | Methodology reviewer | active/in-progress | Do not overclaim until source acquisition/coverage gates pass. |

## Local Generated Artifacts

Generated `output/` artifacts are intentionally not tracked in git. For a
portfolio demo, regenerate them from public input text, keep the command
transcript, and record model/provider/date/audit grade in a short evidence note.

Minimum demo package:

1. input text path and source description
2. command used to run the pipeline
3. `result.json`
4. `report.html`
5. `make audit-result` output
6. short caveat stating what the run does and does not prove
