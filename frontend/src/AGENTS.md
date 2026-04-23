# AGENTS.md

This file defines mandatory guardrails for any code agent working in this repository.

## Mandatory Read Before Editing
Before changing frontend, backend API behavior, or runtime architecture, read:
1. `docs/contracts/frontend_behavior_contract.md`
2. `docs/contracts/business_logic_contract.md`
3. `docs/architecture/architecture_guardrails.md`

If a task touches any file listed below, these contracts are mandatory:
- `ui_shell.js`
- `workflow_main.js`
- `dataset_main.js`
- `DFM.html`
- `dfm_*.js`
- `project_settings.js`
- `backend/api/*.py`
- `backend/services/*.py`
- `backend/config.py`
- `electron_main.js`
- `electron_preload.js`

## Hard Rules (MUST)
1. Keep `adas:*` message names backward-compatible unless all producers/consumers are updated in the same change.
2. Preserve tab dirty-state semantics and close-confirmation behavior.
3. Keep workflow save/load payload compatibility unless explicitly approved to break.
4. Keep router -> service -> config/schema layering; do not move business logic into routers.
5. Any behavior/logic/architecture change must update the corresponding MANUAL doc sections in the same change.

## Required Documentation Workflow
After relevant code changes:
1. Update contract docs (or explicitly state "no contract impact").
2. Run `python tools/docs_index_builder.py --write`.
3. Run `python tools/docs_index_builder.py --check`.
4. If `--check` fails, fix docs before finishing.

## Decision Priority
When code and docs conflict:
1. Explicit user request in current task.
2. This `AGENTS.md`.
3. Contract documents under `docs/contracts/` and `docs/architecture/`.
4. Generated inventories under `docs/generated/`.

## Change Safety
If a requested change appears to violate these contracts:
1. Stop and call out the exact contract rule.
2. Propose compliant alternatives.
3. Proceed only after explicit user confirmation for intentional exception.
