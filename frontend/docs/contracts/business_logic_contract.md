# Business Logic Contract

## Purpose
Define non-negotiable domain behaviors for app-server routes/services and frontend consumers.

## Scope
This contract applies to:
- `app_server/api/*.py`
- `app_server/services/*.py`
- `app_server/schemas/*.py`
- `app_server/config.py`
- `ui/workflow/workflow_main.js`
- `ui/project_settings/project_settings.js`
- `ui/dataset/dataset_main.js`
- `reserving_class_lazy_picker.js`

## BL-1 Workflow Persistence Compatibility
When:
- Changing workflow save/load schemas or payload fields.

MUST:
- Keep backward compatibility for existing `.arcwf` files unless explicitly approved.
- Keep save/load route semantics stable (`/workflow/save`, `/workflow/save_as`, `/workflow/load`).
- Validate load fallback behavior for missing optional fields.

MUST NOT:
- Break previous workflow files silently.
- Change workflow payload shape without documenting migration.

Verify:
1. Existing workflow files load successfully.
2. New saves can be loaded in same runtime without data loss.

## BL-2 Table Summary Cache Semantics
When:
- Editing table summary read/refresh logic.

MUST:
- Preserve meaning of cache validity and `from_cache` behavior.
- Keep refresh endpoint behavior explicit about cache clear/rebuild.
- Preserve optional reserving-class refresh chaining semantics.

MUST NOT:
- Return stale summary while claiming fresh data.
- Remove refresh side effects without updating consumers/docs.

Verify:
1. `GET /table_summary` returns cached result when valid.
2. `POST /table_summary/refresh` rebuilds summary and reports refresh outcomes.

## BL-3 Reserving Class Data and Preferences
When:
- Editing reserving class tree/values/types/hidden-path/filter-spec logic.

MUST:
- Keep project-name validation and error status semantics.
- Keep preference file locking behavior and retry-safe error responses.
- Preserve route-level shape for hidden paths and filter spec payloads.

MUST NOT:
- Break cache/pref file schema without migration handling.
- Return ambiguous errors for lock/contention cases.

Verify:
1. Hidden path and filter spec read/write routes remain symmetric.
2. Refresh routes still report missing columns and output paths when applicable.

## BL-4 Project Settings Folder Operations
When:
- Editing folder rename/duplicate/delete or settings save flows.

MUST:
- Keep operations atomic from client perspective (rollback on partial failures where supported).
- Keep source/folder mapping consistency between settings JSON and filesystem structure.
- Keep audit log writes for material configuration changes.

MUST NOT:
- Leave settings JSON and folder structure diverged silently.
- Skip rollback/error propagation on failed intermediate operations.

Verify:
1. Rename/duplicate/delete operations update both metadata and filesystem as expected.
2. Failed operations report clear error and do not silently corrupt state.

## BL-5 Runtime Config and Path Resolution
When:
- Editing `ui_config` behavior or `app_server/config.py` path logic.

MUST:
- Keep `ui_config.json` as source of truth for root and configured subpaths.
- Refresh runtime paths after accepted config update.
- Keep derived path helpers consistent across services.

MUST NOT:
- Hardcode divergent path logic in individual services.
- Accept invalid config updates without guarded defaults.

Verify:
1. `POST /ui_config` changes effective runtime paths.
2. Dependent routes resolve paths against updated config.

## BL-6 API Validation and Status Contract
When:
- Editing route handlers.

MUST:
- Keep explicit validation for required inputs.
- Preserve semantic HTTP status usage (`400` input issue, `404` not found, `423` lock/contention, `500` unexpected).
- Keep response shape stable for established high-usage routes unless documented.

MUST NOT:
- Hide validation errors behind generic `500`.
- Change status semantics without updating clients/docs.

Verify:
1. Invalid requests return expected status and actionable message.
2. Existing frontend flows handle responses without regression.

## BL-7 Auditability for Critical Changes
When:
- Changing business rules that impact persisted project settings/type definitions.

MUST:
- Keep audit log events for material user-triggered changes.
- Ensure audit writes are best-effort but never crash the core transaction path unexpectedly.

MUST NOT:
- Remove audit hooks from critical edit paths without replacement.

Verify:
1. Relevant actions append audit records.
2. Audit failures are handled with safe degradation where intended.

## Change Checklist
Before finishing a business-logic change:
1. State which BL-* rules were impacted.
2. Update MANUAL sections in affected `docs/app_server/*.md` and/or `docs/app_server/domains/*.md`.
3. Run `python tools/docs_index_builder.py --write`.
4. Run `python tools/docs_index_builder.py --check`.
