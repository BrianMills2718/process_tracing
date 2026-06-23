# Evidence - Plan 003 Source-Aware Extraction Coverage

## Scope

This slice hardens the accepted-source coverage failure found in
`output/live_plan003_dual_audit_20260623_001`: Source C
(`Constitution of the Year VIII`) was present in the input packet text but
produced zero extracted evidence. The fix is intentionally narrow. Pass 1
extraction now receives the accepted source-packet context and is instructed to
preserve source-marker coverage while still treating only the input text as
evidence.

This is not full source acquisition and not the full Slice 3 source-metadata
model. It is a bridge ensuring downstream packet-source coverage reflects the
actual extraction contract.

## Deterministic Verification

Pending full gate run.

Targeted regression command:

```bash
PYTHONPATH=. pytest tests/test_extraction_quality.py \
  tests/test_pipeline_integration.py::TestReportConsistency::test_source_packet_context_reaches_extraction_pass -q
```

Result: `5 passed`.

Coverage:

- `test_extraction_contract_uses_source_packet_for_marker_coverage` verifies
  the Pass 1 prompt includes the accepted source-packet contract, the
  metadata-is-not-evidence warning, marker coverage requirements, and
  legal/constitutional/procedural source relevance.
- `TestReportConsistency.test_source_packet_context_reaches_extraction_pass`
  verifies `run_pipeline(..., source_packet=...)` passes the packet context
  into `run_extract` before hypothesis generation and still builds source
  coverage from the resulting extraction.

The first live attempt also exposed a separate Pass 3 validity failure after
refinement: the LLM assigned `evi_evening_session_chazal_proposal` to multiple
dependence clusters. The deterministic validator correctly failed loud. The
slice now includes a bounded validation-repair loop for Pass 3 and the
regression test
`TestVectorCompleteness.test_repairs_overlapping_clusters_once_with_validation_feedback`.
This repair is not a silent fallback: it reprompts once with the exact
validation error and still raises if the corrected result violates the
deterministic matrix or cluster invariants.

## Live Non-Mocked E2E

First attempt:

```bash
PYTHONPATH=. python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir output/live_plan003_source_aware_extract_20260623_001 \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --model gpt-5-mini \
  --max-budget 2.0
```

Result: failed during refined Pass 3 with:

```text
ValueError: testing: evidence in multiple dependence clusters: ['evi_evening_session_chazal_proposal']
```

Required retry command shape:

```bash
PYTHONPATH=. python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir output/live_plan003_source_aware_extract_20260623_002 \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --model gpt-5-mini \
  --max-budget 2.0

make audit-result RESULT=output/live_plan003_source_aware_extract_20260623_001/result.json \
  REPORT=output/live_plan003_source_aware_extract_20260623_002/report.html \
  FOCAL_YEAR=1799
```

Minimum success readout: Source C is no longer silently `input_only`. If Source
C remains uncovered, the next slice should add a structured per-source
extraction disposition instead of hiding the source's evidentiary value.

Retry result:

- Output directory: `output/live_plan003_source_aware_extract_20260623_002`
- Pipeline completed in `1126.5s`
- Refinement triggered the new Pass 3 validation-repair path once:
  `testing: evidence in multiple dependence clusters:
  ['evi_barras_resignation_persuasion']`
- The repair response succeeded: `43/43` evidence items vectorized across
  `6` hypotheses with `6` dependence clusters covering `41` items.
- Source coverage: `5/5` packet sources represented in extracted evidence.
- Assigned evidence: `43/43`; unassigned evidence: `0`.

Source C (`Constitution of the Year VIII`) is now covered with four evidence
items:

- `evi_year_viii_filtered_lists`
- `evi_year_viii_first_concent_power`
- `evi_year_viii_conservative_senate`
- `evi_year_viii_emergency_clause`

Audit:

```bash
make audit-result RESULT=output/live_plan003_source_aware_extract_20260623_002/result.json \
  REPORT=output/live_plan003_source_aware_extract_20260623_002/report.html \
  FOCAL_YEAR=1799
```

Result:

- Given-source grade: `A (100/100)`
- Claim-scope grade: `B (82/100)`
- Report-surface score: `100/100`
- Conditional cap: `100/100`
- Claim-scope cap: `82/100`
- Optimality status: `optimal_given_accepted_sources`
- Remaining claim-scope blocker: unresolved high-priority private
  correspondence gap.

Browser sanity check:

- Headless Chromium opened
  `output/live_plan003_source_aware_extract_20260623_002/report.html`.
- The rendered packet-source coverage table shows
  `source_c_constitution_year_viii` as `covered`, with `11` input marker hits
  and `4` evidence IDs.

## Adversarial Critique

This slice clears the specific accepted-source extraction failure. It does not
prove publication-strength historical truth. The strongest remaining PhD-level
critique is claim-scope, not given-source coherence: the packet still declares a
high-priority missing private correspondence source class among Sieyes,
Bonaparte, Lucien, Murat, and allied conspirators. Because the current top
hypothesis turns on agency, coercion, and sequencing, private planning traces
could still materially revise the relative support between Bonaparte-centered,
Sieyes-centered, and elite-negotiated variants.

The next optimal slice should not keep polishing the given-source report. It
should either acquire/disposition the high-priority private-correspondence gap
or introduce a structured source-gap disposition artifact that explicitly marks
which publication-strength claims remain blocked by unavailable source classes.
