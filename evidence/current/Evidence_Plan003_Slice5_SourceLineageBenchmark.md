# Evidence Note: Plan #3 Slice 5 — Source-Lineage Dependence Benchmark

**Slice**: 5 — Source-Lineage Dependence Benchmark  
**Date**: 2026-06-25  
**Model**: openrouter/openai/gpt-5-mini  
**Output dir**: output/slice5_e2e/

## What was implemented

1. **`lineage_type` field on `EvidenceCluster`** (`pt/schemas.py`) — `Optional[LineageType]` with values `duplicate | shared_source | same_event | same_mechanism | other`. Backward compat: defaults to `None` so existing result.json files load cleanly.

2. **Pass 3 prompt updated** (`pt/prompts/pass3_test.yaml`) — dependence clustering section now requests `lineage_type` with definitions for each value, explaining WHY items are dependent.

3. **Cap-84 in audit script** (`scripts/audit_result_quality.py`) — detects top-driver evidence items sharing a `source_group` (not "Main text") that are not covered by any dependence cluster. Fires as a conditional cap blocking A-grade claims.

4. **Deterministic tests** (`tests/test_pt_bayesian.py`, `TestSourceLineageDependence`, 8 tests):
   - Planted fixture proves duplicates without a cluster inflate support measurably (>0.05 delta)
   - Planted fixture proves full-redundancy cluster (rho=1.0, lineage_type=duplicate) collapses 3 copies to single-item support (exact)
   - Shared-source (rho=0.7) and same-event (rho=0.6) clusters partially correct inflation — monotonicity verified
   - lineage_type roundtrip, default None, invalid literal rejection
   - 5-item full-redundancy cluster: unclustered vs clustered delta >0.05

## E2E run

Command:
```
python -m pt input_text/revolutions/french_revolution.txt \
    --output-dir output/slice5_e2e --model openrouter/openai/gpt-5-mini --json-only
```

Runtime: 705s  
Result: 6 hypotheses, 41 evidence items, 7 dependence clusters covering 23 items

### Cluster output
7 clusters generated, but `lineage_type=None` on all — gpt-5-mini returned null despite the Optional+description schema. The field validates correctly (null is a valid value); this is a model behavior observation, not a schema failure.

### Audit result: **Grade B (80/100)**

Cap-84 fired as intended:
> 1 source group(s) have multiple top-driver evidence items not covered by any dependence cluster: 'French Directory': evi_directory_collapse_arg_suggestion, evi_coup_18_brumaire_1799.

This confirms the source-lineage inflation check is operational and surfacing real findings.

Cap-82 also fired (7 rival pairs no discriminators), cap-80 fired (only 2/38 proximate evidence) — these are pre-existing text quality issues with the broad Wikipedia overview text, not regressions.

## Acceptance criteria status

| Criterion | Grade |
|-----------|-------|
| lineage_type field exists on EvidenceCluster, Optional | test (A) |
| Backward compat — old clusters without lineage_type load cleanly | test (A) |
| Pass 3 prompt requests lineage_type with category definitions | schema_validated (B) |
| Cap-84 fires when top-drivers share source_group and no cluster | observed (B) |
| Clustered duplicates give same support as single item (rho=1.0) | test (A) |
| Unclustered duplicates inflate support measurably | test (A) |
| Shared-source partial correction is monotone between extremes | test (A) |

## Known limitation recorded

`lineage_type` is Optional (null default) — stronger models (Gemini-2.5-flash) may populate it; gpt-5-mini does not. To force a value, promote to required (removing Optional). Deferred: backward compat cost and limited audit value until a Gemini run confirms population.
