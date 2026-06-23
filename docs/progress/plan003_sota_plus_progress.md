# Plan 003 SOTA+ Progress

## Objective

Make the process-tracing pipeline defensible at PhD / think-tank / academic quality through thin, live-verified slices. The current slice hardens source-packet evidence coverage so reports cannot imply a source-informed design when extracted evidence is not traceable to packet sources.

## Acceptance Criteria

- [x] Deterministic source-packet coverage is persisted in `result.json`, visible in `report.html`, and included in `audit_result_quality.py`.
- [x] Extraction contracts preserve source markers in evidence `source_text` so packet coverage can be measured against real live output.
- [x] `make check` and `make plan-tests PLAN=3` pass.
- [x] A live non-mocked Brumaire source-packet E2E run is executed after implementation, audited, and recorded.
- [x] Verified work is committed and pushed.

## Constraints

- Live non-mocked E2E is mandatory for implementation slices.
- Source-packet metadata is provenance and scope control, not likelihood evidence.
- Missing high-priority source classes must cap claims instead of being hidden by report language.

## Current Phase

Slice 1b complete and pushed in `f18f46d`; dual-track audit correction pushed in `58d5e05` and `4ab3ef2`. The audit shows source material known to the grader, then separates given-source critique caps from claim-scope caps. Fresh live run `output/live_plan003_dual_audit_20260623_001` shows the current active blockers: accepted-source coverage for Source C, lack of decisive tests, and broader claim-scope private-correspondence gap.

## Completed

- Slice 1 baseline source-packet contract, Brumaire benchmark packet, and report/audit source-scope cap were committed in `559fbb1`.
- Initial Slice 1b implementation added source coverage models, report table, audit category fields, and Brumaire packet `source_id` / `text_markers`.
- Live Slice 1b Brumaire run `output/live_plan003_slice1b_final_20260622_002` represented 5/5 packet sources in extracted evidence and assigned 48/49 evidence items.
- Fresh dual-track audit live run `output/live_plan003_dual_audit_20260623_001` represented 4/5 packet sources in extracted evidence and correctly classified Source C's missing extracted evidence as a given-source cap, separate from the private-correspondence claim-scope cap.
- Deterministic verdict calibration removed low-posterior "supported" overclaims from the live artifact; the remaining audit caps are split into given-source robustness and broader claim-scope source gaps.

## Next

Next slice should first address the given-source caps: ensure each accepted packet source produces extracted evidence or is explicitly dispositioned, then add/identify decisive discriminating traces. Source acquisition or disposition of the high-priority missing private-correspondence class is a separate claim-scope lane before broader publication-strength claims.
