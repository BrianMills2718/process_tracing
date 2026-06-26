# Plan 007: Workbench Export V1

**Status:** Planned
**Type:** implementation
**Priority:** High
**Blocked By:** `ISSUE-002` for final `make check` reliability
**Blocks:** `mixed_methods_workbench` PT adapter and fixture-backed walking skeleton

---

## Gap

**Current:** The pipeline writes `result.json` and `report.html`. `result.json`
is the internal `ProcessTracingResult` artifact: it contains raw testing
matrices, Bayesian internals, optional critic/refinement state, and internal
schema details that are free to evolve.

**Target:** Add a versioned, workbench-safe export named `pt_export_v1.json`.
The export should expose source scope, hypotheses, comparative support,
absence findings, verdicts, diagnostic caveats, and artifact provenance without
requiring downstream consumers to import `pt.schemas` or parse internal
`result.json`.

**Why:** `mixed_methods_workbench` needs process-tracing outputs, but the
workbench must not couple to PT internals or flatten comparative support into
generic qualitative confidence. A versioned export lets PT own its method
semantics while giving the workbench a stable artifact seam.

---

## References Reviewed

- `CLAUDE.md` - future adapter surface explicitly says the workbench must not
  import `pt.schemas` or parse `result.json` directly, and names
  `pt_export_v1.json` as the future seam.
- `pt/schemas.py` - defines `ProcessTracingResult`, `HypothesisSpace`,
  `BayesianResult`, `AbsenceResult`, `SynthesisResult`, `DiagnosticMatrix`, and
  the internal result fields that should be summarized rather than exported raw.
- `pt/source_packet.py` - defines `SourcePacketSummary`,
  `SourceCoverageReport`, and source-gap disposition fields needed for claim
  limits.
- `pt/source_coverage.py` - deterministic source-packet coverage over input text
  and extracted evidence.
- `pt/cli.py` - current CLI writes `result.json` and `report.html`.
- `docs/plans/003_sota_plus_execution_master_plan.md` - universal slice
  contract and live E2E/audit expectations.
- `ISSUES.md` - `ISSUE-002` documents the current `make check` runtime surface
  problem.
- `~/projects/mixed_methods_workbench/docs/plans/002_engine_stability_and_integration_readiness.md`
  - names `pt_export_v1` as a blocker for the workbench PT adapter.

---

## Files Affected

- `pt/export.py` (new; export models, builder, JSON writer, CLI `main`)
- `tests/test_export.py` (new deterministic tests)
- `Makefile` (optional; add an agent-drivable `export-workbench` target)
- `CLAUDE.md` (update when implemented)
- `docs/ARTIFACTS.md` (document export artifact)
- `docs/VALIDATION.md` (document verification command)
- `docs/plans/CLAUDE.md` (mark this plan complete when done)
- `ISSUES.md` (move related issue state when done)

Do not modify `mixed_methods_workbench` from this plan. The workbench should
consume the export only after this repo proves it.

---

## Export Contract Sketch

Producer model should be strict: `ConfigDict(extra="forbid")`.

Consumer models in downstream repos may use `extra="ignore"` for forward
compatibility.

Required top-level fields:

- `schema_version`: literal `pt_export_v1`
- `producer`: repo name, git commit when available, export command, created_at
- `artifact_provenance`: `result_json_path`, `result_json_sha256`,
  optional `report_html_path`, optional `report_html_sha256`,
  `source_text_sha256`, optional `source_packet_path`
- `research_question`
- `source_scope`: source-packet summary, source coverage, known gaps,
  source-gap dispositions, limitations, and derived claim limits
- `hypotheses`: ids, descriptions, causal mechanisms, observable predictions
- `comparative_support`: ranked per-hypothesis support entries with:
  `hypothesis_id`, `rank`, `comparative_support_weight`, `robustness`,
  `sensitivity_low`, `sensitivity_high`, `rank_stable`,
  `top_driver_evidence_ids`
- `evidence`: evidence ids, descriptions, quote/source text, evidence type,
  source group/genre, date metadata, and trace-production relevance
- `absence_findings`: hypothesis id, prediction id, missing evidence, severity,
  would-be-extractable flag, expected source genre/location, reasoning
- `verdicts`: hypothesis id, calibrated status, key evidence for/against,
  steelman, reasoning, posterior robustness
- `diagnostic_caveats`: partition audit cap, pairs without discriminators,
  diagnostic grade cap, source-coverage gaps, unresolved high-priority gaps
- `limitations`: combined synthesis, source-scope, partition, diagnostic, and
  export limitations

Forbidden export semantics:

- no generic `confidence` field;
- no `probability_of_truth` label;
- no raw likelihood matrix as the public workbench contract;
- no source-packet metadata treated as evidence by itself;
- no absence finding included in Bayesian support;
- no workbench import of `pt.schemas`.

## Plan

### Steps

1. Create strict export models in `pt/export.py`.
2. Implement `build_export_v1(result: ProcessTracingResult, *, result_path:
   Path | None, report_path: Path | None) -> ProcessTracingExportV1`.
3. Implement deterministic claim-limit derivation:
   - no source packet -> explicit source-scope limitation;
   - unresolved high-priority source gaps -> explicit claim cap;
   - source coverage missing/input-only/unconfigured rows -> explicit caveats;
   - partition or diagnostic grade caps -> explicit caveats.
4. Implement JSON writer and `python -m pt.export RESULT --output
   pt_export_v1.json --report REPORT`.
5. Optionally add `make export-workbench RESULT=... REPORT=... OUT=...`.
6. Add deterministic tests over a minimal `ProcessTracingResult` fixture.
7. Run a live Brumaire or French Revolution command, then export the generated
   result and run `make audit-result` against the original result/report.
8. Update docs and issue/plan status.

### Pre-Made Decisions

- The export is a projection of `ProcessTracingResult`, not a replacement for
  `result.json`.
- The export may include comparative support weights, but labels must say
  comparative support, not probability or truth confidence.
- `SourceCoverageReport` and `SourcePacketSummary` are source-scope evidence
  for claim limits, not causal evidence for a hypothesis.
- The export should be usable without `report.html`, but if a report path is
  supplied it must be hashed into provenance.
- Missing source packet should not crash export by default; it should emit an
  explicit limitation. Workbench policy may later reject no-packet exports.

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|---|---|---|
| `tests/test_export.py` | `test_export_v1_preserves_source_scope_and_claim_limits` | Source packet summary, coverage gaps, and limitations survive export. |
| `tests/test_export.py` | `test_export_v1_labels_comparative_support_not_confidence` | Export contains comparative-support labels and no generic confidence/probability-of-truth fields. |
| `tests/test_export.py` | `test_export_v1_summarizes_hypotheses_verdicts_and_absence` | Hypotheses, calibrated verdicts, and absence findings map into the public export. |
| `tests/test_export.py` | `test_export_v1_does_not_export_raw_testing_matrix` | Raw `testing.evidence_likelihoods` is not exposed as the workbench contract. |
| `tests/test_export.py` | `test_export_cli_writes_valid_json_with_hashes` | `python -m pt.export` writes deterministic JSON with artifact hashes. |
| `tests/test_export.py` | `test_export_v1_no_source_packet_has_explicit_limitation` | No-packet runs export only with an explicit source-scope limitation. |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|---|---|
| `tests/test_source_packet.py` | Source-scope summary and coverage semantics remain intact. |
| `tests/test_source_coverage.py` | Coverage-derived caveats remain deterministic. |
| `tests/test_pipeline_integration.py` | Internal result production still works. |
| `tests/test_cli_source_packet.py` | CLI source-packet plumbing remains unchanged. |

Use repo-local verification while `ISSUE-002` is open:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_export.py tests/test_source_packet.py tests/test_source_coverage.py tests/test_cli_source_packet.py -q --tb=short
PYTHONPATH=. .venv/bin/python -m pytest tests -q --tb=short
```

After `ISSUE-002` is resolved, `make check` is required.

## Live Verification Gate

Plan completion requires one live non-mocked run, because the export is a public
cross-repo seam:

```bash
python -m pt input_text/source_packets/18_brumaire_source_packet.txt \
  --output-dir /tmp/pt_export_v1_brumaire \
  --source-packet docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  --research-question "Why did the French Revolution culminate in Napoleon Bonaparte's 18 Brumaire coup and the Consulate rather than a stable parliamentary republic, a revived Jacobin-dominated republic, or a royalist restoration?" \
  --theories input_text/theories/18_brumaire_rival_frameworks.txt \
  --refine \
  --max-budget 1.0

python -m pt.export /tmp/pt_export_v1_brumaire/result.json \
  --report /tmp/pt_export_v1_brumaire/report.html \
  --output /tmp/pt_export_v1_brumaire/pt_export_v1.json

make audit-result RESULT=/tmp/pt_export_v1_brumaire/result.json \
  REPORT=/tmp/pt_export_v1_brumaire/report.html \
  FOCAL_YEAR=1799
```

Record model, command, output path, export hash, and audit findings before
marking complete.

## Acceptance Criteria

- [ ] `pt_export_v1.json` can be generated from a valid `result.json`.
- [ ] Export models are strict producer models.
- [ ] Export has schema version, producer metadata, artifact paths, hashes, and
      source text hash.
- [ ] Export preserves research question, source scope, hypotheses,
      comparative support, absence findings, verdicts, diagnostic caveats, and
      limitations.
- [ ] Export never labels comparative support as truth probability or generic
      confidence.
- [ ] Export does not expose raw internal testing matrices as the workbench
      contract.
- [ ] No source packet yields an explicit limitation.
- [ ] Deterministic tests pass.
- [ ] Full repo-local tests pass.
- [ ] `make check` passes after `ISSUE-002` is resolved, or current check debt is
      explicitly cited in completion notes.
- [ ] One live E2E run is exported and audited.
- [ ] Docs and plan/issue status are updated.

## Notes

This is intentionally an export-only slice. It should not create a workbench
adapter, change inference behavior, or introduce a UI dependency. The output is
licensed to say "PT results can be consumed by the workbench through a stable
artifact." It does not license methodological-validity claims beyond the
underlying run's evidence and caveats.
