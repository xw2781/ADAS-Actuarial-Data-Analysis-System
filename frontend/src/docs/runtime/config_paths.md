# Runtime: Config and Path Resolution

## Purpose
<!-- MANUAL:BEGIN -->
Document path/config setup and runtime path refresh behavior.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN runtime.config_paths.entry_points -->
- Path/config helper functions in `backend/config.py`:
  - `_find_existing_project_dir`
  - `_get_data_dir`
  - `_get_project_map_dir`
  - `_get_requests_dir`
  - `_get_scripting_dir`
  - `_get_user_appdata_cache_dir`
  - `_get_virtual_projects_dir`
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
  - `load_ui_config`
  - `refresh_runtime_paths`
- UI config routes:
  - `GET` `/ui_config` handled by `get_ui_config`
  - `POST` `/ui_config` handled by `update_ui_config`
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN runtime.config_paths.key_files -->
- [`backend/config.py`](../../backend/config.py) - Primary runtime path + config module.
- [`backend/api/ui_config_router.py`](../../backend/api/ui_config_router.py) - HTTP interface for root path updates.
- [`ui_config.json`](../../ui_config.json) - Persisted root path and subpath config.
- [`backend/main.py`](../../backend/main.py) - App bootstrap and static path mounting.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Frontend shell settings modal calls `/ui_config` routes.
- Backend imports `backend.config` for runtime path resolution.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- `ui_config.json` is persistent source-of-truth for root/path mapping.
- Runtime globals in `backend/config.py` are refreshed from config.
- User-local fixed paths are also refreshed in `backend/config.py`, including workflow export path (`~/Documents/ArcRho/workflows`) and scripting notebook path (`~/Documents/ArcRho/scripts`).
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add a new configurable path: update `ui_config.json` contract + `backend/config.py` getters.
2. Change path refresh behavior: validate all services that depend on runtime globals.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Path changes affect every filesystem-backed domain.
- Environment-specific path assumptions can break packaged deployments.
<!-- MANUAL:END -->
