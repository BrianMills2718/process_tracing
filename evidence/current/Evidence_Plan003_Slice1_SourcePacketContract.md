# Evidence - Plan 003 Slice 1 Source Packet Contract

## Slice

Plan 003 Slice 1: Source Packet Contract And Benchmark.

## Implemented

- Added `pt/source_packet.py` with a reusable source-packet contract, loader,
  summary model, and prompt-context renderer.
- Reused that contract from the assistant harness, so Slice 0 assistant artifacts
  can be consumed directly by the pipeline.
- Added `--source-packet` to `python -m pt` and `make source-packet-run`.
- Added `ProcessTracingResult.source_packet` summary metadata.
- Passed source-packet context into Pass 2 hypothesis generation while warning
  that packet metadata is not itself evidence.
- Added a report source-packet table and audit distinctions for:
  - no packet,
  - packet present but thin/gapped/limited,
  - packet present but synthesis/report still using stale single-source caveats.
- Added `docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json` as the concrete
  Brumaire benchmark contract.

## Verification

Live non-mocked E2E is mandatory for this slice. The initial committed version
had deterministic coverage only; the live run below caught and drove a required
Pass 5 refinement fix.

Targeted tests:

```bash
PYTHONPATH=. pytest tests/test_cli_source_packet.py tests/test_source_packet.py tests/test_pipeline_integration.py -q --tb=short
```

Result: `46 passed`.

Earlier targeted slice set:

```bash
PYTHONPATH=. pytest tests/test_source_packet.py tests/test_assistant.py tests/test_pipeline_integration.py -q --tb=short
```

Result: `50 passed, 1 skipped`.

Type check:

```bash
mypy pt --ignore-missing-imports
```

Result: `Success: no issues found in 26 source files`.

Full project check:

```bash
make check
```

Result after the live-E2E refinement fix: `174 passed, 2 skipped`; mypy clean; real LLM compliance 100%;
markdown links OK; `AGENTS.md` sync OK.

Plan gate:

```bash
make plan-tests PLAN=3
```

Result after adding the Pass 5 live-failure regression: `55 passed, 2 skipped`;
all required Plan 003 tests pass.

Packet load smoke:

```bash
PYTHONPATH=. python - <<'PY'
from pt.source_packet import load_source_packet
packet = load_source_packet('docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json')
print(packet.to_summary('docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json').model_dump())
PY
```

Result: loaded 5 source candidates, 5 source groups, 2 known gaps, and 1
high-priority gap.

Failed live E2E attempt, default Gemini model:

```bash
PYTHONPATH=. python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir output/live_plan003_slice1_brumaire_20260622_074438 \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --max-budget 1.0
```

Result: failed during refinement application:
`ValueError: Refinement: new causal edge endpoint 'h3' is not a known node`.
The live refinement emitted evidence-to-hypothesis support/challenge links in
`new_causal_edges`. That is not a valid causal-process edge and would pollute
the causal graph if accepted. Fix: tightened `NewCausalEdge` descriptions,
added a valid causal-edge endpoint inventory to Pass 5, explicitly forbade
hypothesis IDs such as `h1`/`h3` in `new_causal_edges`, and added
`tests/test_pass_refine.py`.

Second live E2E attempt, default Gemini model:

```bash
PYTHONPATH=. python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir output/live_plan003_slice1_brumaire_retry_20260622_075023 \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --max-budget 1.0
```

Result: stopped and rerouted because Gemini returned `429 RESOURCE_EXHAUSTED`
for the free-tier `gemini-2.5-flash` quota.

Successful live non-mocked E2E:

```bash
PYTHONPATH=. python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir output/live_plan003_slice1_brumaire_openrouter_20260622_075208 \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --model gpt-5-mini \
  --max-budget 2.0
```

Result: completed end to end in `1191.5s`, writing:

- `output/live_plan003_slice1_brumaire_openrouter_20260622_075208/result.json`
- `output/live_plan003_slice1_brumaire_openrouter_20260622_075208/report.html`
- `output/live_plan003_slice1_brumaire_openrouter_20260622_075208/refinement.json`

Live run summary:

- Model: `openrouter/openai/gpt-5-mini`
- Refined: yes
- Hypotheses: 6
- Evidence after refinement: 53
- Top hypothesis after refinement: `h4`
- Top comparative support: `0.555154`
- Source packet persisted: 5 source groups, 2 known gaps, 1 high-priority gap
- Refinement applied: 2 new evidence items, 2 reinterpretations, 1 spurious
  causal edge removed, 3 new causal edges, 3 hypothesis refinements

Live audit:

```bash
make audit-result \
  RESULT=output/live_plan003_slice1_brumaire_openrouter_20260622_075208/result.json \
  REPORT=output/live_plan003_slice1_brumaire_openrouter_20260622_075208/report.html \
  FOCAL_YEAR=1799
```

Result: `B (82/100)`, with report-surface score `100/100` and academic evidence
cap `82/100`. Remaining blocker: the source packet is present but still has a
high-priority missing source class and unresolved packet limitations, so the
audit correctly refuses to upgrade the live run to an `A`.

## Critique

This slice makes source scope explicit and reviewable before inference, but it
does not yet automate source acquisition or prove that every packet source
contributed extractable evidence. The report and prompt therefore state that
packet metadata is not evidence. Concern C-006 records the remaining source
coverage problem.

The packet can pin the research question and shape hypothesis generation, but
Pass 1 extraction still operates on the supplied input text. A future
source-acquisition/coverage slice must build the input corpus from packet
sources and report source-group evidence coverage.

The live E2E also exposed that Pass 5 needed stronger constraints separating
causal-process edges from evidence-hypothesis diagnostic links. That is now
covered by prompt/schema text and `tests/test_pass_refine.py`, but future
schema work should consider a first-class diagnostic-link refinement artifact
instead of asking the model to express everything through causal edges,
hypothesis refinements, and analyst notes.
