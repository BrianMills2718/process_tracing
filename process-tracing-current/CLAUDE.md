# Process Tracing Current Repo Wiki — Wiki Schema

This bundle is a Karpathy-style OKF wiki for the active `process_tracing` repo.
The human curates direction and decides what is authoritative. The LLM maintains
the compiled knowledge layer.

## Purpose

This wiki documents the current repo so agents and humans can orient quickly
without rereading every plan, architecture document, and source file. It is an
orientation and synthesis layer, not the source of truth. Current repo code,
tests, `CLAUDE.md`, and active docs win over this wiki whenever they conflict.

## The Three Layers

- `raw/` - immutable source snapshots or source manifests used for wiki ingests.
  Do not edit an ingested raw source.
- `wiki/` - LLM-maintained markdown knowledge base.
- `CLAUDE.md` - this schema and operating contract.

## Directory Map

```text
raw/
  assets/
wiki/
  index.md
  log.md
  overview.md
  sources/
  architecture/
  plans/
  contracts/
  concepts/
  entities/
  _candidates/
```

## Page Types

Every non-reserved markdown page in `wiki/` must have YAML frontmatter with a
non-empty `type`.

Use these page types first:

- `Overview` - top-level current-state synthesis.
- `Source` - summary of an active repo document, code surface, or test surface.
- `Architecture` - component, boundary, or data-flow explanation.
- `Plan` - active roadmap or thin-slice plan synthesis.
- `Contract` - typed interface, schema, API, CLI, or artifact contract.
- `Concept` - process-tracing or implementation concept.
- `Entity` - named internal system object when a concept page is too broad.
- `Candidate` - unstable or unresolved note awaiting promotion.

Recommended frontmatter:

```yaml
---
type:
title:
description:
tags: []
created:
updated:
sources: []
confidence:
---
```

`confidence` may be `high`, `medium`, `low`, or `speculative`.

## Linking

Use markdown links. Prefer bundle-root-relative paths beginning with `/wiki/`
when linking inside the bundle. Broken links are allowed as work markers, but
must be resolved or triaged during lint.

## Operations

### Ingest

1. Identify the active repo source file, command output, or code surface.
2. Record its path and, when copied into `raw/`, its content hash.
3. Write or update a `wiki/sources/` page.
4. Integrate the implications into architecture, plan, contract, concept, or
   entity pages.
5. Cite source paths on every non-obvious claim.
6. Update `wiki/index.md`.
7. Append the action to `wiki/log.md` with newest date first.

### Query

1. Read `wiki/index.md`.
2. Open the smallest set of relevant pages.
3. Answer with citations to wiki pages and, when current behavior matters, active
   repo source paths.
4. File durable answers back into the wiki and log them.

### Lint

Run deterministic structural lint when available, then inspect for:

- stale current-state claims
- contradictions with active repo docs/code
- pages with missing provenance
- orphan pages
- concepts that deserve their own page

Every lint finding needs an action: fix, promote, archive, or escalate.

## House Rules

- This wiki is current-state orientation only.
- Do not use this wiki as authority over active repo files.
- Do not import archive claims unless they are explicitly promoted into active
  repo docs first.
- User corrections about wiki structure or interpretation get filed back into
  this schema.

