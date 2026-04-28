# App Server Domain: workspace_paths

## Purpose
<!-- MANUAL:BEGIN -->
Runtime workspace path read/update domain.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN app_server.workspace_paths.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/workspace_paths` | `get_workspace_paths` | - | - | - |
| `POST` | `/workspace_paths` | `update_workspace_paths` | `WorkspacePathsUpdateRequest` | [`app_server/schemas/workspace_paths.py`](../../../app_server/schemas/workspace_paths.py) | - |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN app_server.workspace_paths.key_files -->
- [`app_server/api/workspace_paths_router.py`](../../../app_server/api/workspace_paths_router.py) - Read/update workspace path config.
- [`app_server/config.py`](../../../app_server/config.py) - Config loader and runtime path refresh.
- [`app_server/schemas/workspace_paths.py`](../../../app_server/schemas/workspace_paths.py) - Workspace path request models.
- [`workspace_paths.json`](../../../workspace_paths.json) - Persistent workspace path configuration.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Used by shell root-path settings modal.
- Triggers `config.refresh_runtime_paths()` on updates.
- Keeps legacy `/ui_config` as a compatibility alias.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Persists config in `workspace_paths.json`.
- Reads legacy `ui_config.json` as a fallback for older installs.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add config field: update schema + router serialization + config readers.
2. Keep legacy aliases stable until an intentional compatibility removal.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Invalid path config writes can impact all path-dependent domains.
<!-- MANUAL:END -->
