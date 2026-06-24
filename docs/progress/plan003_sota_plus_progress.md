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

## Constraints

- Live non-mocked E2E is mandatory for implementation slices.
- Source-packet metadata is provenance and scope control, not likelihood evidence.
- Missing high-priority source classes must cap claims instead of being hidden by report language.

## Current Phase

Claim-scope source expansion is live-verified and landed. Source-aware extraction coverage was committed and pushed in `9ac37fc`. Fresh live run `output/live_plan003_source_expansion_20260623_001` expanded the packet to 6 sources, covered Source F with 8 evidence items, rendered the source-gap disposition table, and kept the high-priority correspondence gap honestly marked as `partially_mitigated`.

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

## Next

Next highest-value work is either direct correspondence acquisition or a hypothesis-partition gate. If staying within current public-source constraints, add the partition gate next because the live runs keep producing merge suggestions and prior-sensitive winners.
