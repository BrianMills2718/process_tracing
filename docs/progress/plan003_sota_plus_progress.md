# Plan 003 SOTA+ Progress

## Objective

Make the process-tracing pipeline defensible at PhD / think-tank / academic quality through thin, live-verified slices. The current slice hardens source-packet evidence coverage so reports cannot imply a source-informed design when extracted evidence is not traceable to packet sources.

## Acceptance Criteria

- [x] Deterministic source-packet coverage is persisted in `result.json`, visible in `report.html`, and included in `audit_result_quality.py`.
- [x] Extraction contracts preserve source markers in evidence `source_text` so packet coverage can be measured against real live output.
- [x] `make check` and `make plan-tests PLAN=3` pass.
- [x] A live non-mocked Brumaire source-packet E2E run is executed after implementation, audited, and recorded.
- [x] Source-packet context reaches Pass 1 extraction, accepted marker-bearing sources are represented in extracted evidence or remain visible as explicit coverage failures, and the behavior is verified by deterministic tests plus a fresh live non-mocked E2E run.
- [x] Verified source-aware extraction work is committed and pushed.
- [x] Source-gap disposition metadata distinguishes acquired, partially mitigated, unresolved, unavailable, and accepted-limit source gaps in packet summary, report, and audit.
- [x] The Brumaire source packet is expanded with private-planning memoir evidence and rerun through live non-mocked E2E.
- [x] Verified source expansion work is committed and pushed.
- [x] Trace-derived source-acquisition targets rank missing evidence by inferential payoff and can drive live `open_web_retrieval` searches.
- [x] Local workbench exposes the acquisition agenda and a click-to-enrich action through an agent-drivable JSON API.
- [x] Active architecture docs satisfy design-plan diagram requirements for deductive surfaces.

## Constraints

- Live non-mocked E2E is mandatory for implementation slices.
- Source-packet metadata is provenance and scope control, not likelihood evidence.
- Missing high-priority source classes must cap claims instead of being hidden by report language.

## Current Phase

Plan 005 has now moved from proposal to implementation: the interactive trace execution host ships typed stage replay, run inspection, and live non-mocked verification, while later visual-audit slices remain open. At the methodology level, the active frontier is now explicit: source packet -> source-design engine, stronger trace-production modeling, richer dependence handling, and a tighter within-case -> cross-case bridge. `docs/SOTA_PLUS_TARGET_ARCHITECTURE.md` now captures the end-goal boundary/domain/data-flow package for that target. Plan 006 is implemented and folded back into the roadmap as the source-design engine slice, with `pt/source_design.py` and the acquisition/workbench payloads carrying the new typed state.

## Completed

- Slice 1 baseline source-packet contract, Brumaire benchmark packet, and report/audit source-scope cap were committed in `559fbb1`.
- Initial Slice 1b implementation added source coverage models, report table, audit category fields, and Brumaire packet `source_id` / `text_markers`.
- Live Slice 1b Brumaire run `output/live_plan003_slice1b_final_20260622_002` represented 5/5 packet sources in extracted evidence and assigned 48/49 evidence items.
- Fresh dual-track audit live run `output/live_plan003_dual_audit_20260623_001` represented 4/5 packet sources in extracted evidence and correctly classified Source C's missing extracted evidence as a given-source cap, separate from the private-correspondence claim-scope cap.
- Deterministic verdict calibration removed low-posterior "supported" overclaims from the live artifact; the remaining audit caps are split into given-source robustness and broader claim-scope source gaps.
- Source-aware extraction implementation now passes source-packet context into Pass 1 and adds deterministic regression coverage for the prompt contract and pipeline wiring.
- Live retry `output/live_plan003_source_aware_extract_20260623_002` completed end to end. Pass 3 validation repair fired once on overlapping dependence clusters and succeeded. Source coverage is 5/5 packet sources, Source C has 4 evidence items, assigned evidence is 43/43, given-source grade is `A (100/100)`, and claim-scope grade remains `B (82/100)` because the private-correspondence gap is unresolved.
- Source expansion slice added structured source-gap dispositions and Source F, Bourrienne's private-secretary memoir, to partially mitigate the private-planning gap without pretending direct correspondence is resolved.
- Live source-expansion run `output/live_plan003_source_expansion_20260623_001` completed end to end in `1112.2s`. Source coverage is 6/6 packet sources, Source F has 8 evidence items, assigned evidence is 50/50, given-source grade is `A (100/100)`, and claim-scope grade remains `B (82/100)` because the high-priority correspondence gap is only partially mitigated. Top support shifted to `h1` at `0.550`, showing the new source changed the inference rather than only increasing evidence volume.
- Source-acquisition slice added `pt.source_acquisition`, `scripts/source_acquisition_plan.py`, and `make source-acquisition`. The planner derives next-source targets from unresolved source gaps, damaging absences, prior/posterior sensitivity, and top-driver corroboration needs.
- Workbench slice added `pt.workbench` and `make workbench`. Puppeteer verified the local UI loads, renders the acquisition targets, and the `Enrich Top Targets` button saves `output/source_acquisition/workbench_latest.json` with six extracted hits across the top two targets.
- Interactive trace execution host slice added `pt.trace_host`, stage-by-stage JSON endpoints in `pt.workbench`, and a live non-mocked Brumaire run at `output/workbench_runs/run_20260624_200616_d4f6` that completed through refine (`result.json`, `report.html`, and `refinement.json`).
- Design-plan cleanup added `docs/ARCHITECTURE.md` with Mermaid boundary, domain-model, and data-flow diagrams plus typed contract and failure-path tables.
- Plan 005 added the pre-implementation design artifacts for the interactive trace execution host: `docs/plans/005_interactive_trace_execution_host.md`, `docs/plans/005_interactive_trace_execution_host_mockup.html`, and `docs/plans/005_interactive_trace_execution_host_contracts.ipynb`.
- Plan 006 added the execution-ready source-design-engine child plan and its required seam mockup / contract notebook artifacts.
- Plan 006 implementation added the typed `SourceDesignState` loop, action records, review decisions, live Brumaire retrieval run, and docs sync.

## Next

Review the Plan 005 mockup. In parallel, Plan 006 is ready to drive the next non-UI methodology slice around source-design completion. Keep `docs/ARCHITECTURE.md` and `docs/SOTA_PLUS_TARGET_ARCHITECTURE.md` synchronized with any new boundary, schema, or data-flow contract.
