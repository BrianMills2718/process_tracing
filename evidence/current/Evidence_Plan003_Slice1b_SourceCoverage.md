# Evidence - Plan 003 Slice 1b Source Coverage And Calibration

## Slice

Plan 003 Slice 1b: source-packet coverage verification and report/synthesis hardening.

## Implemented

- Added deterministic packet-source coverage in `pt/source_coverage.py`.
- Extended source candidates with `source_id` and exact `text_markers`.
- Persisted `ProcessTracingResult.source_coverage`.
- Rendered packet-source coverage in `report.html` and audited it in
  `make audit-result`.
- Tightened extraction source-fidelity instructions and schema descriptions so
  live evidence `source_text` preserves packet source markers.
- Added report control-character sanitization so live model text cannot place
  NUL/C0 characters into HTML.
- Added deterministic synthesis verdict calibration in `pt/verdict_calibration.py`
  so very-low-posterior hypotheses cannot remain labeled `supported`.
- Added dual-track audit semantics: the grader sees `source_material_context`,
  then receives a given-source grade and a separate claim-scope grade.

## Deterministic Verification

```bash
make check
```

Result: `180 passed, 2 skipped`; mypy clean; LLM compliance 100%; markdown
links OK; `AGENTS.md` sync OK.

```bash
make plan-tests PLAN=3
```

Result: `64 passed, 2 skipped`; all required Plan 003 tests pass.

## Live Non-Mocked E2E

Command:

```bash
PYTHONPATH=. python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir output/live_plan003_slice1b_final_20260622_002 \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --model gpt-5-mini \
  --max-budget 2.0
```

Result: completed end to end in `810.5s`, writing:

- `output/live_plan003_slice1b_final_20260622_002/result.json`
- `output/live_plan003_slice1b_final_20260622_002/report.html`
- `output/live_plan003_slice1b_final_20260622_002/refinement.json`

Live run summary:

- Model: `openrouter/openai/gpt-5-mini`
- Refined: yes
- Extracted after refinement: 49 evidence items
- Packet-source coverage: 5/5 packet sources represented in extracted evidence
- Assigned evidence: 48/49 evidence items assigned to packet sources
- Missing/input-only/unconfigured sources: none
- Report NUL count after sanitizer/regeneration: 0
- Verdict calibration: no supported/strongly-supported verdict below `0.10`

Audit after deterministic verdict calibration and report regeneration:

```bash
make audit-result \
  RESULT=output/live_plan003_slice1b_final_20260622_002/result.json \
  REPORT=output/live_plan003_slice1b_final_20260622_002/report.html \
  FOCAL_YEAR=1799
```

Result after the dual-track audit correction:

- Given-source grade: `B (84/100)`
- Claim-scope grade: `B (82/100)`
- Report-surface score: `100/100`
- Conditional cap: high-support fragile winner
- Claim-scope cap: unresolved high-priority source gap

Resolved blockers:

- Packet coverage cap cleared: accepted packet sources now have evidence
  coverage.
- Verdict-calibration cap cleared: no supported/strongly-supported verdict has
  comparative support below `0.10`.
- Report safety issue cleared: generated HTML contains no embedded NUL bytes.

Remaining blocker:

- The source packet still declares an unresolved high-priority source gap:
  private correspondence among conspirators. The correct next iteration is
  source acquisition or explicit scholarly disposition of that gap, not report
  polish.
- The final live run's leading hypothesis is high-support but fragile, so the
  next evidence slice also needs fewer, stronger discriminating traces and a
  dependence-cluster review.
