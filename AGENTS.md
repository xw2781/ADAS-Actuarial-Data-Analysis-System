# AGENTS.md

This is the ArcRho monorepo root. Use one Git repository here for all ArcRho components.

## Repository Layout
- `frontend/`: current ArcRho desktop/web UI, Electron host, backend service code currently bundled with the frontend app, docs, release fragments, and frontend-specific agent rules.
- `data-engine/`: ArcRho data-engine component, including the legacy agent/master/shell Python services previously stored under `backend/`.
- `tools/`: repository-level automation, including commit/push helpers for agents.

## Mandatory Read Before Editing
Before changing files under `frontend/`, read `frontend/AGENTS.md` and follow its contracts and documentation workflow.

When future component-specific `AGENTS.md` files are added, read the nearest one before editing that component.

## Commit and Push Workflow
When the user asks an agent to commit and push ArcRho code:
1. Inspect the final root-level diff/status and make sure the commit scope matches the current conversation.
2. Write a fresh, specific commit message from the actual updates in that conversation. Do not reuse a generic message.
3. Run the root helper from `ArcRho/`:
   `powershell -ExecutionPolicy Bypass -File tools\agent_commit_push.ps1 -Message "Describe the current update"`
4. Use `-DryRun` when reviewing commit scope, `-NoPush` for a local commit only, and a comma-list such as `-Pathspec frontend,data-engine` when intentionally limiting scope.
5. Report the commit hash and push result back to the user.
