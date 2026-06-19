# docs/validation

Historical validation utilities and benchmark scripts for `process_tracing`.

This subtree is not part of the default verification gate. Treat scripts here
as archived evidence or starting points for renewed validation work, not as
authoritative current tests.

Active verification lives in `tests/` and runs through `make check`.
Production code lives in `pt/`.

If a validation script becomes current again, update it against the active `pt`
APIs and move the enforced coverage into `tests/`.
