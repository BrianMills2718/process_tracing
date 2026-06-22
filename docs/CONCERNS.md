# Process Tracing Concern Register

Wiki home: http://localhost:8088/index.php/Project_Wiki

This register tracks portfolio-level concerns. Implementation concerns for the
active SOTA+ lane live in `docs/plans/sota_plus_concern_register.md`.

Do not delete rows. Update disposition when triaged.

| ID | Concern | Where | Why It Matters | Disposition | Owner / Next Slice |
|---|---|---|---|---|---|
| PT-PORT-001 | The methodology story is stronger than the empirical validation story. | `docs/VALIDATION.md` | Reviewers may overread architecture as demonstrated performance. | open | Build a frozen benchmark and adversarial/human review package before stronger claims. |
| PT-PORT-002 | The French Directory case bundle is compact and public-source; it demonstrates workflow shape, not historical truth. | `docs/portfolio/FRENCH_DIRECTORY_COLLAPSE_CASE_BUNDLE.md` | The portfolio should not oversell one public historical example. | accepted | Keep caveats visible; regenerate a fuller demo package before external sharing. |
| PT-PORT-003 | Source packets and source coverage are active in-progress work. | `docs/source_packets/`; Plan 003 | Source-scope improvements are promising but not yet the final portfolio endpoint. | open | Finish the active source-packet/source-coverage lane and commit it cleanly. |
| PT-PORT-004 | Hypothesis partition quality remains a hard open surface. | `docs/METHODOLOGY.md`; Plan 003 | A coherent posterior over a weak hypothesis set can still mislead. | open | Add partition audit readouts and benchmark failures before claiming stronger validity. |
| PT-PORT-005 | Generated output artifacts are local-only. | `output/` | Reviewers need a reproducible demo package, not a path that may not exist. | open | Create a regenerated public demo package with command transcript, result, report, audit output, and caveats. |
| PT-PORT-006 | Product polish is behind method architecture. | HTML report and CLI workflows | CIA/portfolio reviewers may need a clear, short inspection path. | mitigated | Use `PROJECT.md`, `docs/REVIEWER_WALKTHROUGH.md`, and the case bundle as the first-reader path. |
