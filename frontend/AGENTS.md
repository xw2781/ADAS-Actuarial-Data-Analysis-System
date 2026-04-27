# AGENTS.md

This file defines mandatory guardrails for any code agent working in this repository.
  
## Mandatory Read Before Editing
Before changing frontend, app-server API behavior, or runtime architecture, read:
1. `docs/contracts/frontend_behavior_contract.md`
2. `docs/contracts/business_logic_contract.md`
3. `docs/architecture/architecture_guardrails.md`

These contracts are mandatory whenever a task touches:
- Frontend shell or feature entry/coordinator files under `ui/` (for example shell, dataset, workflow, DFM, or project settings).
- App-server API, service, or runtime config files under `app_server/api/`, `app_server/services/`, or `app_server/config.py`.
- Electron runtime bridge/host files such as `electron_main.js` or `electron_preload.js`.

## Hard Rules (MUST)
1. Keep `adas:*` message names backward-compatible unless all producers/consumers are updated in the same change.
2. Preserve tab dirty-state semantics and close-confirmation behavior.
3. Keep workflow save/load payload compatibility unless explicitly approved to break.
4. Keep router -> service -> config/schema layering; do not move business logic into routers.
5. Any behavior/logic/architecture change must update the corresponding MANUAL doc sections in the same change.
6. Any meaningful user-facing feature, fix, improvement, or breaking change must add a release fragment under `changes/unreleased/` with a short user-facing summary.

## Required Documentation Workflow
After relevant code changes:
1. Update contract docs (or explicitly state "no contract impact").
2. Run `python tools/docs_index_builder.py --write`.
3. Run `python tools/docs_index_builder.py --check`.
4. If `--check` fails, fix docs before finishing.

## Commit and Push Workflow
When the user asks an agent to commit and push frontend code:
1. Inspect the final diff/status first and make sure the commit scope matches the current conversation.
2. Write a fresh, specific commit message from the actual updates in that conversation. Do not reuse a generic message.
3. Use the root helper from `ArcRho/`, for example:
   `powershell -ExecutionPolicy Bypass -File tools\agent_commit_push.ps1 -Message "Reorganize frontend UI entrypoints"`
   The compatibility wrapper at `frontend/tools/agent_commit_push.ps1` delegates to the same root helper.
4. Use `-DryRun` when reviewing commit scope, `-NoPush` when the user asks for a local commit only, `-StageMode none` only when intentionally committing already staged changes, and `-Pathspec frontend` or a comma-list such as `-Pathspec frontend,tools` when intentionally limiting commit scope.
5. Report the commit hash and push result back to the user.

## Decision Priority
When code and docs conflict:
1. Explicit user request in current task.
2. This `AGENTS.md`.
3. Contract documents under `docs/contracts/` and `docs/architecture/`.
4. Generated inventories under `docs/generated/`.

## Change Safety
Before modifying code, stop and double-check with the user if any of the following are detected:
1. The request is unclear or missing important implementation details.
2. The request appears to conflict with standard or best-practice application development.
3. The new request conflicts with existing code logic, contracts, or architecture.
4. The request is likely not the best option for long-term architecture, optimization, or maintainability.

In those cases:
1. Start the user-facing response with `[!!!!!]` to make the triggered safety/contract concern explicit.
2. Call out the concern clearly.
3. Ask targeted clarifying question(s) or propose better options.
4. Proceed only after the user confirms the direction.

If a requested change appears to violate these contracts:
1. Stop and call out the exact contract rule.
2. Propose compliant alternatives.
3. Proceed only after explicit user confirmation for intentional exception.
