# Runtime: Config and Path Resolution

## Purpose
<!-- MANUAL:BEGIN -->
Document path/config setup and runtime path refresh behavior.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN runtime.config_paths.entry_points -->
- Path/config helper functions in `app_server/config.py`:
  - `_clean_path_segment`
  - `_find_existing_project_dir`
  - `_get_project_map_dir`
  - `_get_requests_dir`
  - `_get_scripting_dir`
  - `_get_user_appdata_cache_dir`
  - `_get_workflow_dir`
  - `_infer_project_name_from_table_path`
  - `_sanitize_project_dir_name`
  - `get_audit_log_path`
  - `get_cache_path`
  - `get_dataset_types_path`
  - `get_field_mapping_path`
  - `get_general_settings_path`
  - `get_path`
  - `get_project_data_dir`
  - `get_project_settings_workbook_path`
  - `get_reserving_class_combinations_path`
  - `get_reserving_class_filter_spec_pref_path`
  - `get_reserving_class_hidden_paths_pref_path`
  - `get_reserving_class_path_tree_path`
  - `get_reserving_class_types_path`
  - `get_reserving_class_values_path`
  - `get_root_path`
  - `get_scripting_prefs_path`
  - `load_workspace_paths`
  - `refresh_runtime_paths`
  - `save_workspace_paths`
- Workspace path config routes:
  - `GET` `/workspace_paths` handled by `get_workspace_paths`
  - `POST` `/workspace_paths` handled by `update_workspace_paths`
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN runtime.config_paths.key_files -->
- [`app_server/config.py`](../../app_server/config.py) - Primary runtime path + config module.
- [`app_server/api/workspace_paths_router.py`](../../app_server/api/workspace_paths_router.py) - HTTP interface for workspace path updates.
- [`workspace_paths.json`](../../workspace_paths.json) - Persisted workspace root and subpath config.
- [`app_server/main.py`](../../app_server/main.py) - App bootstrap and static path mounting.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Frontend shell settings modal calls `/workspace_paths` routes.
- App-server modules import `app_server.config` for runtime path resolution.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- `workspace_paths.json` is persistent source-of-truth for workspace root/path mapping.
- Runtime globals in `app_server/config.py` are refreshed from config.
- User-local fixed paths are also refreshed in `app_server/config.py`, including workflow export path (`~/Documents/ArcRho/workflows`) and scripting notebook path (`~/Documents/ArcRho/scripts`).
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add a new configurable path: update `workspace_paths.json` contract + `app_server/config.py` getters.
2. Change path refresh behavior: validate all services that depend on runtime globals.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Path changes affect every filesystem-backed domain.
- Environment-specific path assumptions can break packaged deployments.
<!-- MANUAL:END -->
