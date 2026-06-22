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

Result: `173 passed, 2 skipped`; mypy clean; real LLM compliance 100%;
markdown links OK; `AGENTS.md` sync OK.

Plan gate:

```bash
make plan-tests PLAN=3
```

Result after removing duplicate plan-test listing: `54 passed, 2 skipped`;
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
