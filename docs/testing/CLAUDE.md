# docs/testing

Historical manual and exploratory validation helpers for `process_tracing`.

This subtree is **not** part of the active pytest suite. Several files preserve
old investigation scripts from the pre-`pt` pipeline era and may reference
removed entry points such as `process_trace_advanced.py`.

Active verification lives in `tests/` and runs through `make check`.
Production code lives in `pt/`.

Before reusing anything here, verify it against the current `pt` APIs and move
revived tests into `tests/` as deterministic pytest coverage.
