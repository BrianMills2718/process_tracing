# ADR 0001: Process-Tracing Methodology Spine

Wiki home: http://localhost:8088/index.php/Project_Wiki

Status: accepted
Date: 2026-06-22

## Context

The project can be misread as generic qualitative coding, summarization, or LLM
research tooling. That framing loses the point. The repo is strongest when
presented as a method-aware causal-analysis system: rival hypotheses,
diagnostic evidence, explicit absence checks, deterministic comparative support
updates, source-scope caveats, and inspectable reports.

The portfolio need is also different from the implementation need. A reviewer
needs a compact path through goals, methodology, artifacts, validation, limits,
and next slices. Agents need the same structure so future work does not drift
back into broad infrastructure or overclaimed demos.

## Decision

Treat this repo as a process-tracing methodology system and maintain a
repo-owned dossier spine:

- `PROJECT.md`
- `docs/METHODOLOGY.md`
- `docs/ARTIFACTS.md`
- `docs/VALIDATION.md`
- `docs/CONCERNS.md`
- `docs/wiki_manifest.yaml`

The portfolio narrative must lead with analyst-method discipline, not generic
LLM tooling. Claims must distinguish implemented, partial, planned, and
not-claimed capabilities.

## Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Present as qualitative-analysis automation | Too broad and undersells the causal-inference structure. |
| Present as AI engineering infrastructure | Accurate for some implementation mechanics, but not the strongest CIA analyst signal. |
| Present the French Directory output as the main claim | Risks confusing workflow evidence with a historical truth claim. |
| Wait until full benchmark validation exists | Would hide valuable current work and block useful portfolio curation. |

## Consequences

- The first-reader path is now explicit and reviewer-oriented.
- Validation gaps stay visible instead of being buried in plans or chat.
- Future implementation should strengthen source scope, hypothesis partition,
  dependence modeling, benchmark evidence, and regenerated demo artifacts.
- Missing evidence should lower the claim, not trigger report polish.

## Supersession Rule

This ADR should be superseded only if the project changes its lead portfolio
claim away from automated process tracing, or if a stronger validated benchmark
package changes the evidence hierarchy.
