# Plan 005 / Slice 005d — Visual Audit Concern Register

Live concern register for `docs/plans/005_interactive_trace_execution_host.md`, Slice 005d (visual audit panels).

Append new concerns as new rows. Update status and disposition in place when triaged. Never delete entries. Triage at every slice boundary.

## Disposition Values

- `open` — not yet addressed.
- `resolved` — fixed and verified.
- `mitigated` — reduced enough to proceed, with remaining risk stated.
- `accepted` — intentionally left as known risk, with rationale.
- `escalated` — blocks progress until re-planned or decided.
- `deferred` — assigned to a named future slice.

## Register

| ID | Status | Area | Concern | Why it matters | Disposition / next action |
|---|---|---|---|---|---|
| D-001 | resolved | trace_host.py | Refine stage silently overwrites extraction.json, hypothesis_space.json, testing.json, absence.json, bayesian.json, synthesis.json in-place, destroying the pre-refinement state. | Slice 005d's done-when includes a before/after delta board for the refine stage. Without the pre-refine artifacts, the board cannot be built. Also any audit that asks "what changed?" has no baseline. | Fixed in Task 2: save each artifact to output_dir/pre_refine/ before overwriting. Deterministic test added. |
| D-002 | resolved | workbench.py | _html() is 495 lines of inline Python-string HTML/CSS/JS with {{/}} escaping. Adding matrix + bars + provenance + cluster panels will push it to 1000+ unmaintainable lines. | Static analysis, linting, browser DevTools, and isolation testing are all blocked by inline embedding. The {{/}} escaping is a trap — any new JS using template literals must double-escape. | Resolved by ADR-005-HTML: extract to pt/templates/workbench.html (Task 3 + Task 11). |
| D-003 | resolved | workbench.py | No stage-specific view API exists. The UI fetches raw artifact blobs via /artifact?path= and dumps them into a <pre> block. | For a 67 KB testing.json, the browser downloads the full object graph and shows nothing useful. Visual panels require structured projections, not raw JSON. | Resolved by adding GET /api/runs/{run_id}/stages/{stage_id}/view with ViewRenderer projections (Task 12). |
| D-004 | open | workbench UI | Clicking a stage button in the sidebar always sends POST .../stages/{id}/run. There is no affordance to distinguish 'view artifact' from 'execute stage'. | force=false prevents re-execution, but the UX is confusing: 'Running extract...' flashes for every click on a completed stage. Auditors need to browse artifacts without accidentally triggering re-runs. | Deferred to a UX polish pass after panels are shipped. Mitigated: the force=false behavior is correct; the visual cost is one flash. |
| D-005 | accepted | trace_host.py | TraceHostStore uses ThreadingHTTPServer but has no locking on run.json. Concurrent stage runs on the same run_id could corrupt the state file. | For a single-user local tool, concurrent stage execution is unlikely. But the server is multi-threaded and the design does not prevent it. | Accepted as a known limitation for the single-user local use case. Mitigation: each stage sets status=running at the start; a second concurrent call would see running and fail stage-order validation, which partially serializes access. Document in plan. |
| D-006 | resolved | methodology / UI | The h2 posterior of 0.994 is labeled 'fragile' in the Bayesian result. Without a clear visual treatment, users may read 0.994 as a decisive settled result. | The CLAUDE.md 'load-bearing known limitation' explicitly says fragile high-posterior results must be read as rankings, not settled conclusions. If the UI shows a big green bar for h2, it creates exactly the overclaim the methodology warns against. | Resolved in Task 14: renderSupport() fires a prominent warning banner when any bar with posterior > 0.5 has robustness == 'fragile'. Task 18 legibility readout confirmed the banner fires for the Brumaire run (h2 post=0.994, fragile). |
| D-007 | resolved | view rendering | 59-item evidence matrix may be illegible at full scroll. Filtering, sorting, or pagination may be required but cannot be determined until the matrix is rendered with real data. | This is the exploratory part of Slice 005d. Over-specifying the filtering behavior up front would be faking precision. | Resolved in Task 18 legibility readout: 59 rows sorted by discrimination (most-discriminating first) is manageable for auditing. Caption shows "59 items · N below threshold". Top rows carry the causal story; auditors skim remaining. No pagination or filter controls needed for the current use case. |
| D-008 | open | plan hygiene | docs/plans/005_interactive_trace_execution_host.md boundary, domain, and data-flow diagrams cover the host skeleton (Slices 005b-005c), not the visual audit (Slice 005d). | The skill requires all three diagrams to reflect the system being built. Implementing without updated diagrams means the implementation is unverified against contracts. | Addressed by Tasks 4–6 (diagram updates). Must be completed before Task 11 (implementation). |
