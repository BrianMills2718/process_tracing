# Implementation Plans

Track all implementation work here.

## Gap Summary

| # | Name | Priority | Status | Blocks |
|---|------|----------|--------|--------|
| 1 | Tentative Maintenance Roadmap | Medium | 📋 Planned | Future cleanup/refactor implementation plans |
| 2 | SOTA+ Recovery And Thin-Slice Operating Model | Critical | 🚧 In Progress | Source packet contract, hypothesis partition audit, validation benchmark |
| 3 | SOTA+ Execution Master Plan | Critical | 🚧 In Progress | All SOTA+ implementation slices |

## Status Key

| Status | Meaning |
|--------|---------|
| Planned | Ready to implement |
| In Progress | Being worked on |
| Blocked | Waiting on dependency |
| Complete | Implemented and verified |

## Creating a New Plan

1. Copy `TEMPLATE.md` to `NN_name.md`
2. Fill in gap, steps, required tests
3. Add to this index
4. Commit with `[Plan #N]` prefix

## Trivial Changes

Not everything needs a plan. Use `[Trivial]` for:
- Less than 20 lines changed
- No changes to `pt/` (production code)
- No new files created

```bash
git commit -m "[Trivial] Fix typo in README"
```

## Completing Plans

```bash
python scripts/meta/complete_plan.py --plan N
```

This verifies tests pass and records completion evidence.
