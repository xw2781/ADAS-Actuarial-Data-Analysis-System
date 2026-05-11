# AGENTS.md

This is the ArcRho monorepo root. Use one Git repository here for all ArcRho components.

## Repository Layout
- `frontend/`: current ArcRho desktop/web UI, Electron host, backend service code currently bundled with the frontend app, docs, release fragments, and frontend-specific agent rules.
- `data-engine/`: ArcRho data-engine component.
- `tools/`: repository-level automation, including commit/push helpers for agents.

## Mandatory Read Before Editing
Before making code or documentation edits, run `git branch --show-current`.

If the current branch is `main`, ask before starting non-trivial feature, bugfix, or ArcBot work. 

ArcBot related work should normally use `codex/arcbot`.

Before changing files under `frontend/`, read `frontend/AGENTS.md`.

## Bug Fix Cleanup Review
When fixing a bug, remove clearly obsolete code in the touched area. Ask before broader cleanup or cleanup with behavior risk.
