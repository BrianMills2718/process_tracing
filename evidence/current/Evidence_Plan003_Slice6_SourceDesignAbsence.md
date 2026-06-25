# Evidence Note: Plan #3 Slice 6 — Source-Design Engine and Observability-Weighted Absence

**Slice**: 6 — Source-Design Engine And Observability-Weighted Absence  
**Date**: 2026-06-25  
**Model**: openrouter/openai/gpt-5-mini  
**Output dir**: output/slice6_e2e/

## What was implemented

1. **Two new fields on `AbsenceEvaluation`** (`pt/schemas.py`):
   - `expected_source_genre: Optional[SourceGenre]` — genre of source that would carry this missing trace (uses existing SourceGenre Literal)
   - `expected_source_location: Optional[str]` — concrete collection/document type where the missing trace would appear (e.g. "minutes of the Conseil des Cinq-Cents 1798–1799")
   - Both Optional for backward compat; old result.json files load cleanly

2. **Pass 3b prompt updated** (`pt/prompts/pass3b_absence.yaml`) — new "Source acquisition guidance" section instructs the LLM to populate both fields for every finding, even when `would_be_extractable=False`, with guidance to name specific archives/collections.

3. **Report updated** (`pt/report.py`) — absence table now has "Acquire from" column: genre badge + specific location text. Pre-existing Pyright issue on `pred_desc[:100]` fixed (None-safety).

4. **10 deterministic tests** (`tests/test_extraction_quality.py`, `TestAbsenceAcquisitionFields`):
   - Both fields default to None
   - `expected_source_genre` accepts all valid SourceGenre literals, rejects invalid
   - `expected_source_location` stores and roundtrips string
   - Backward compat: old absence dict without new fields loads cleanly
   - `genre` populated even when `would_be_extractable=False` (acquisition guidance independent of extractability judgment)
   - Prompt contract tests: both field names present in pass3b_absence.yaml

## E2E run

Command:
```
python -m pt input_text/revolutions/french_revolution.txt \
    --output-dir output/slice6_e2e --model openrouter/openai/gpt-5-mini --json-only
```

Runtime: 651s  
Result: 6 hypotheses, 35 evidence items, 5 dependence clusters, 11 absence findings (4 damaging)

### Absence field population
- `expected_source_genre` populated: **11/11**
- `expected_source_location` populated: **11/11**

Sample acquisition pointers produced:
- "contemporary newspapers (Le Moniteur Universel 7–10 Nov 1799), military dispatches"
- "parliamentary minutes (Archives parlementaires), contemporary press dispatches"
- "collections of contemporary pamphlets (Bibliothèque nationale de France)"
- "parliamentary debate transcripts in the Archives parlementaires"

All outputs are specific, actionable acquisition targets, not vague "primary sources".

### Audit result: **Grade B (80/100)**

Cap-82, cap-80, cap-88 fired — all pre-existing text-quality gaps with the broad Wikipedia overview corpus, not regressions from Slice 6. Cap-84 (source-lineage inflation) did not fire — top-drivers were clustered correctly in this run.

## Acceptance criteria status

| Criterion | Grade |
|-----------|-------|
| expected_source_genre field on AbsenceEvaluation, Optional | test (A) |
| expected_source_location field, Optional | test (A) |
| Backward compat — old absence without new fields loads cleanly | test (A) |
| Pass 3b prompt instructs LLM on both fields | test (A) |
| Prompt populates expected_source_genre on all findings | observed (B) — 11/11 |
| Prompt populates expected_source_location with specific targets | observed (B) — 11/11 |
| genre populated regardless of would_be_extractable | test (A) |
| Report shows "Acquire from" column with genre badge + location | schema_validated (B) |
| Absence stays outside Bayesian update | existing — not changed |

## Key design decision recorded

`expected_source_genre` and `expected_source_location` are populated for **all** findings, not just extractable ones. When `would_be_extractable=False` (the current text can't contain this), naming the genre/location is still the research value — it tells the analyst where to look next, not that the current absence is informative.
