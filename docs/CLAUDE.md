# docs

Maintained documentation for `process_tracing`.

## Route By Question

- active implementation planning -> `plans/`
- active deterministic verification -> `../tests/` via `make check`
- historical manual verification helpers -> `testing/`
- historical validation benchmarks and checks -> `validation/`

## Working Rules

- Keep this parent file at routing level only.
- Historical or superseded material belongs in `archive/`, not in new top-level
  docs files.
- When changing pipeline behavior, update the closest methodological or
  validation doc rather than adding loose status notes here.
