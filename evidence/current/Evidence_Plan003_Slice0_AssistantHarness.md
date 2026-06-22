# Evidence - Plan 003 Slice 0 Assistant Harness

Date: 2026-06-22

## Scope

Implemented the smallest vertical Slice 0: a process-tracing assistant harness
that routes source-packet drafting through `llm_client` `workspace_agent` and
writes a typed JSON artifact. This is not the final source-packet contract; that
is Slice 1.

## Changed Files

- `pt/assistant.py`
- `pt/prompts/assistant_source_packet.yaml`
- `tests/test_assistant.py`
- `pytest.ini`
- `Makefile`
- `README.md`
- `docs/FUTURE_WORK.md`
- `docs/plans/sota_plus_concern_register.md`

## Verification

Focused deterministic tests:

```bash
PYTHONPATH=. pytest tests/test_assistant.py -q --tb=short
```

Result: `5 passed, 1 skipped`.

Type check:

```bash
mypy pt --ignore-missing-imports
```

Result: `Success: no issues found in 25 source files`.

Plan-targeted tests:

```bash
make plan-tests PLAN=3
```

Result: `46 passed, 2 skipped`.

LLM/assistant compliance:

```bash
PYTHONPATH=. python scripts/check_real_compliance.py
```

Result: semantic pipeline passes still route through `pt.llm`; assistant harness
is separately reported as `[OK] pt/assistant.py` for `llm_client`
`workspace_agent` use.

CLI help:

```bash
python -m pt.assistant --help
python -m pt.assistant source-packet --help
make help
```

Result: assistant CLI and `source-packet-draft` Make target are visible.

Invalid-model smoke:

```bash
python -m pt.assistant source-packet \
  --case-name "18 Brumaire" \
  --context docs/source_packets/18_BRUMAIRE_RESEARCH_DESIGN.md \
  --output /tmp/pt_smoke/assistant/should_not_exist.json \
  --model gpt-4o \
  --max-budget 0.25
```

Result: clean exit `1` with no traceback:
`assistant model must be a Codex or Claude Code backend accepted by llm_client`.

Full check:

```bash
make check
```

Result after final documentation updates: `165 passed, 2 skipped`; mypy clean;
LLM compliance 100%; markdown links clean; AGENTS sync clean.

## Independent Critique Attempt

Attempted Claude adversarial review with:

```bash
claude -p "Here is a code diff..." --bare
```

Result: blocked by local Claude authentication:
`Not logged in - Please run /login`.

Per Plan 003's fallback rule, this slice used a deliberately separate
adversarial pass and recorded the limitation here and in the concern register.
The fallback pass found one concrete issue: invalid model errors produced a
traceback. Fixed by making `pt.assistant.main()` catch `AssistantError` and
`ValueError`, return exit code `1`, and print a concise error.

The slice also exposed a repo tooling issue: `make plan-tests PLAN=3` could not
find `003_...` plan files and then did not set `PYTHONPATH` for targeted pytest
runs. Fixed in `scripts/meta/check_plan_tests.py`.

## Remaining Risks

- Live Codex/Claude Code provider smoke was not run. It is available through:
  `PT_RUN_LIVE_AGENT_TESTS=1 pytest tests/test_assistant.py -q`.
- The harness emits a source-packet draft, but the pipeline does not yet consume
  a source-packet contract. This is now C-005 in the concern register and belongs
  to Slice 1.
- Future independent critiques should use authenticated Claude/Codex or the
  assistant harness itself once available in the target environment.
