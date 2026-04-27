# App Server Domain: ui_config

## Purpose
<!-- MANUAL:BEGIN -->
Runtime UI config read/update domain.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN app_server.ui_config.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/ui_config` | `get_ui_config` | - | - | - |
| `POST` | `/ui_config` | `update_ui_config` | `UIConfigUpdateRequest` | [`app_server/schemas/ui_config.py`](../../../app_server/schemas/ui_config.py) | - |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN app_server.ui_config.key_files -->
- [`app_server/api/ui_config_router.py`](../../../app_server/api/ui_config_router.py) - Read/update root path config.
- [`app_server/config.py`](../../../app_server/config.py) - Config loader and runtime path refresh.
- [`app_server/schemas/ui_config.py`](../../../app_server/schemas/ui_config.py) - UI config request model.
- [`ui_config.json`](../../../ui_config.json) - Persistent root/path configuration.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Used by shell root-path settings modal.
- Triggers `config.refresh_runtime_paths()` on updates.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Persists config in `ui_config.json`.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add config field: update schema + router serialization + config readers.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Invalid config writes can impact all path-dependent domains.
<!-- MANUAL:END -->
