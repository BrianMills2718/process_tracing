# Plan #N: [Name]

**Status:** Planned
**Type:** implementation  <!-- implementation | design -->
**Priority:** High | Medium | Low
**Blocked By:** None
**Blocks:** None

---

## Gap

**Current:** What exists now

**Target:** What we want

**Why:** Why this matters

---

## References Reviewed

> **REQUIRED:** Cite specific code/docs reviewed before planning.

- `pt/example.py:45-89` - active pipeline implementation, when relevant
- `docs/architecture/current/example.md` - current design
- `CLAUDE.md` - project conventions

---

## Files Affected

> **REQUIRED:** Declare upfront what files will be touched.

- pt/example.py (modify)
- tests/test_feature.py (modify/create)

---

## Plan

### Steps

1. Create X
2. Modify Y
3. Add tests
4. Update docs

---

## Required Tests

### New Tests (TDD)

| Test File | Test Function | What It Verifies |
|-----------|---------------|------------------|
| `tests/test_example.py` | `test_happy_path` | Basic functionality works |
| `tests/test_example.py` | `test_error_case` | Errors handled correctly |

### Existing Tests (Must Pass)

| Test Pattern | Why |
|--------------|-----|
| `tests/test_related.py` | Integration unchanged |

---

## Acceptance Criteria

- [ ] Required tests pass
- [ ] Full test suite passes
- [ ] Deterministic active-code tests pass
- [ ] Type check passes, or current mypy debt is explicitly listed
- [ ] Docs updated

---

## Notes

[Design decisions, alternatives considered, risks]
