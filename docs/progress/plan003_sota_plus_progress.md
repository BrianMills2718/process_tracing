# Plan 003 SOTA+ Progress

## Objective

Make the process-tracing pipeline defensible at PhD / think-tank / academic quality through thin, live-verified slices. The current slice hardens source-packet evidence coverage so reports cannot imply a source-informed design when extracted evidence is not traceable to packet sources.

## Acceptance Criteria

- [x] Deterministic source-packet coverage is persisted in `result.json`, visible in `report.html`, and included in `audit_result_quality.py`.
- [x] Extraction contracts preserve source markers in evidence `source_text` so packet coverage can be measured against real live output.
- [x] `make check` and `make plan-tests PLAN=3` pass.
- [x] A live non-mocked Brumaire source-packet E2E run is executed after implementation, audited, and recorded.
- [x] Source-packet context reaches Pass 1 extraction, accepted marker-bearing sources are represented in extracted evidence or remain visible as explicit coverage failures, and the behavior is verified by deterministic tests plus a fresh live non-mocked E2E run.
- [ ] Verified source-aware extraction work is committed and pushed.

## Constraints

- Live non-mocked E2E is mandatory for implementation slices.
- Source-packet metadata is provenance and scope control, not likelihood evidence.
- Missing high-priority source classes must cap claims instead of being hidden by report language.

## Current Phase

Source-aware extraction coverage is implemented and live-verified. Slice 1b complete and pushed in `f18f46d`; dual-track audit correction pushed in `58d5e05` and `4ab3ef2`. Fresh live run `output/live_plan003_source_aware_extract_20260623_002` clears the accepted-source coverage blocker: Source C is covered, 5/5 packet sources have extracted evidence, and the given-source audit grade is `A (100/100)`. The active blocker is now claim-scope: unresolved high-priority private-correspondence evidence.

## Completed

- Slice 1 baseline source-packet contract, Brumaire benchmark packet, and report/audit source-scope cap were committed in `559fbb1`.
- Initial Slice 1b implementation added source coverage models, report table, audit category fields, and Brumaire packet `source_id` / `text_markers`.
- Live Slice 1b Brumaire run `output/live_plan003_slice1b_final_20260622_002` represented 5/5 packet sources in extracted evidence and assigned 48/49 evidence items.
- Fresh dual-track audit live run `output/live_plan003_dual_audit_20260623_001` represented 4/5 packet sources in extracted evidence and correctly classified Source C's missing extracted evidence as a given-source cap, separate from the private-correspondence claim-scope cap.
- Deterministic verdict calibration removed low-posterior "supported" overclaims from the live artifact; the remaining audit caps are split into given-source robustness and broader claim-scope source gaps.
- Source-aware extraction implementation now passes source-packet context into Pass 1 and adds deterministic regression coverage for the prompt contract and pipeline wiring.
- Live retry `output/live_plan003_source_aware_extract_20260623_002` completed end to end. Pass 3 validation repair fired once on overlapping dependence clusters and succeeded. Source coverage is 5/5 packet sources, Source C has 4 evidence items, assigned evidence is 43/43, given-source grade is `A (100/100)`, and claim-scope grade remains `B (82/100)` because the private-correspondence gap is unresolved.

## Next

Commit and push the verified source-aware extraction slice. Next, stop optimizing the given-source report and address claim scope: either acquire/disposition the high-priority private-correspondence source class or add a structured source-gap disposition artifact that makes publication-strength claims impossible until unresolved source classes are explicitly accepted or resolved.
