# Evidence - Plan 003 Source-Gap Disposition And Source Expansion

## Scope

This slice addresses the remaining claim-scope blocker from
`output/live_plan003_source_aware_extract_20260623_002`: unresolved
high-priority private correspondence among Sieyes, Bonaparte, Lucien, Murat,
and allied conspirators.

The goal is not to pretend the gap is solved. The goal is to make source-gap
status explicit and agent-drivable, then expand the Brumaire packet with
available private-planning memoir evidence.

## Implementation

- Added `SourceGapDisposition` to `pt/source_packet.py`.
- `SourcePacketSummary` now stores `source_gap_dispositions` and
  `unresolved_high_priority_gap_count`.
- `report.html` now renders a Source Gap Dispositions table.
- `audit_result_quality.py` now reports source-gap dispositions and uses
  unresolved high-priority gaps for claim-scope caps.
- Added Source F: Louis Antoine Fauvelet de Bourrienne, *Memoirs of Napoleon
  Bonaparte*, Volume 3, public-domain text hosted from Project Gutenberg
  material by Britannica.
- Updated the assembled Brumaire corpus with Source F private-planning and
  coercion-sequence traces.

## Source Search Notes

Confirmed useful source:

- Bourrienne memoir HTML at
  `https://cdn.britannica.com/primary_source/gutenberg/PGCC_classics/nb03v.htm`.
  Relevant passages include the limited disclosure of Bonaparte's plans,
  Bonaparte's strategic choice to join Sieyes, role distribution before
  Brumaire, deception of Barras, generals assembled at Rue de la Victoire, and
  Lucien/Murat's coercive sequence on 19 Brumaire.

Null/limited result:

- A public Napoleon letters/documents PDF route was checked but did not return
  usable Brumaire correspondence within this slice. The packet therefore marks
  the private-correspondence gap as `partially_mitigated`, not `acquired`.

## Deterministic Verification

Focused command:

```bash
python -m json.tool docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json >/dev/null
PYTHONPATH=. pytest tests/test_source_packet.py \
  tests/test_pipeline_integration.py::TestReportConsistency::test_source_packet_is_visible_in_report_and_audit -q
PYTHONPATH=. mypy pt --ignore-missing-imports
```

Result:

- JSON validation passed.
- Focused pytest: `4 passed`.
- Mypy: `Success: no issues found in 28 source files`.

## Live Non-Mocked E2E

Command:

```bash
PYTHONPATH=. python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir output/live_plan003_source_expansion_20260623_001 \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --model gpt-5-mini \
  --max-budget 2.0

make audit-result RESULT=output/live_plan003_source_expansion_20260623_001/result.json \
  REPORT=output/live_plan003_source_expansion_20260623_001/report.html \
  FOCAL_YEAR=1799
```

Minimum success readout:

- Source F is represented in extracted evidence.
- Report and audit show the source-gap disposition table.
- The high-priority correspondence gap remains explicitly
  `partially_mitigated` unless direct correspondence is actually acquired.

Result:

- Output directory: `output/live_plan003_source_expansion_20260623_001`
- Pipeline completed in `1112.2s`.
- Source coverage: `6/6` packet sources represented in extracted evidence.
- Assigned evidence: `50/50`; unassigned evidence: `0`.
- Source F evidence count: `8`.
- Refined Pass 3 triggered validation repair once for overlapping dependence
  cluster evidence `evi_false_majority_commissioning`; the repair succeeded
  with `50/50` evidence vectors and `7` dependence clusters covering `49`
  items.

Source F extracted evidence:

- `evi_bourrienne_coup_confided_few`
- `evi_bonaparte_initially_consider_director`
- `evi_bonaparte_choice_sieyes_over_barras`
- `evi_roles_distributed_before_coup`
- `evi_generals_preassembled_18brumaire`
- `evi_false_majority_commissioning`
- `evi_bonaparte_proclamation_dictated`
- `evi_bourrienne_deception_barras`

Audit:

```bash
make audit-result RESULT=output/live_plan003_source_expansion_20260623_001/result.json \
  REPORT=output/live_plan003_source_expansion_20260623_001/report.html \
  FOCAL_YEAR=1799
```

Result:

- Given-source grade: `A (100/100)`
- Claim-scope grade: `B (82/100)`
- Source count: `6`
- Sources with evidence: `6`
- High-priority gap count: `1`
- Unresolved high-priority gap count: `1`
- Source gap disposition:
  `Private correspondence among Sieyes, Bonaparte, Lucien, Murat, and allied
  conspirators: partially_mitigated (sources=source_f_bourrienne_memoirs)`
- Top hypothesis after source expansion: `h1`, support `0.550`, moderate,
  prior-sensitive.

Browser sanity check:

- Headless Chromium opened the new report.
- The rendered Source Gap Dispositions table shows the private-correspondence
  gap as `partially_mitigated`, with Source F as the relevant source and the
  claim implication that direct correspondence remains unresolved.
- The rendered Packet Source Coverage table shows Source F as `covered` with
  `11` input marker hits and `8` evidence IDs.

## Adversarial Critique

The slice improves the source base and changes the live inference, but it does
not clear publication-strength source scope. Bourrienne is valuable because he
is an insider private-secretary memoirist, and the pipeline extracted the right
kind of planning-sequence evidence. It remains retrospective, self-positioned,
and not equivalent to contemporaneous private correspondence among the named
conspirators. The correct next move is not more report iteration; it is either
direct correspondence acquisition or an explicit decision to keep claims scoped
to a public-source plus memoir packet.
