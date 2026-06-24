# Evidence - Plan 003 Source-Acquisition Loop

## Scope

This slice turns the prior source-gap finding into an iterative acquisition
loop. The process trace now emits ranked targets for the evidence most likely
to clarify the next run, rather than relying on ad hoc source expansion.

## Implementation

- Added `pt.source_acquisition` with typed `AcquisitionTarget` and
  `AcquisitionPlan` contracts.
- Added `scripts/source_acquisition_plan.py` and `make source-acquisition`.
- Ranking inputs:
  - unresolved source gaps and source-gap dispositions;
  - damaging absence evaluations;
  - prior/posterior sensitivity for the top hypothesis versus runner-up;
  - independent corroboration needs for top-driver evidence.
- Live retrieval uses `open_web_retrieval` with provider credentials from the
  environment, fetches results, and records extraction availability.

## Deterministic Verification

```bash
PYTHONPATH=. pytest tests/test_source_acquisition.py -q
```

Result: `3 passed`.

```bash
make check
```

Result: `186 passed, 2 skipped`; mypy passed; real LLM compliance was 100%;
markdown links and AGENTS sync passed.

```bash
make plan-tests PLAN=3
```

Result: `67 passed, 2 skipped`; all required Plan 003 tests passed.

## Live Non-Mocked Retrieval

Command:

```bash
make source-acquisition \
  RESULT=output/live_plan003_source_expansion_20260623_001/result.json \
  SOURCE_PACKET=docs/source_packets/18_BRUMAIRE_SOURCE_PACKET.json \
  SOURCE_ACQUISITION_OUTPUT=output/source_acquisition/brumaire_plan003_20260624_live.json \
  SOURCE_ACQUISITION_ARGS="--retrieve --max-targets 2 --top-k 3 --queries-per-target 1"
```

Top targets:

1. `acq_gap_1`, score `96`: private planning sequence, actor intent,
   coalition bargaining, and whether Bonaparte converted a civilian plot into
   personalist rule.
2. `acq_sensitivity_top_vs_runner_up`, score `92`: independent trace
   distinguishing `h1` Sieyes-led civilian constitutional revision from `h4`
   Napoleon-centered personalist military takeover.

Retrieved and extracted hits:

| Target | Hit | Extracted text |
|---|---|---:|
| `acq_gap_1` | Jacques Boudon, *Lucien Bonaparte et le coup d'État de Brumaire* | 1,212 chars |
| `acq_gap_1` | Napoleon.org, *18 Brumaire: the context and course of a coup d'État* | 15,430 chars |
| `acq_gap_1` | Britannica archived page, *Coup of 18-19 Brumaire* | 1,994 chars |
| `acq_sensitivity_top_vs_runner_up` | Napoleon.org, *18 Brumaire: the context and course of a coup d'État* | 15,430 chars |
| `acq_sensitivity_top_vs_runner_up` | Britannica archived page, *Coup of 18-19 Brumaire* | 1,994 chars |
| `acq_sensitivity_top_vs_runner_up` | Napoleon Series, *Brumaire Decree* | 5,874 chars |

Machine-readable artifacts:

- `output/source_acquisition/brumaire_plan003_20260624.json`
- `output/source_acquisition/brumaire_plan003_20260624_live.json`

## Adversarial Critique

This improves the research loop, but it does not resolve the high-priority
private-correspondence gap. The top live hits are usable source routes or
adjacent source material, not proof that direct correspondence among Sieyes,
Bonaparte, Lucien, Murat, and allied conspirators has been acquired.

The main methodological gain is that the system now asks a sharper question:
which missing trace would most change the current inference? For this run, the
answer is not generic "more sources." It is private-planning evidence capable
of distinguishing a Sieyes-led civilian constitutional project from a
Napoleon-centered personalist conversion.

Next iteration should either fetch and assess the Lucien Bonaparte hit as a
possible source-packet candidate or run targeted archival queries for
correspondence collections. If no direct correspondence route emerges after
recorded searches, the packet should preserve the claim-scope cap and proceed
to the hypothesis-partition gate.
