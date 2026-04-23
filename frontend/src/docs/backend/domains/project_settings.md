# Backend Domain: project_settings

## Purpose
<!-- MANUAL:BEGIN -->
Project settings source and folder-structure management domain.
Also persists per-project Source Data date boundaries in `general_settings.json`.
Provides project-folder filesystem operations used by Project Settings tree actions (rename/duplicate/create/delete).
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.project_settings.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/general_settings` | `get_general_settings` | `str` | - | `project_settings_service.get_general_settings` |
| `POST` | `/general_settings` | `update_general_settings` | `GeneralSettingsUpdateRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.update_general_settings` |
| `GET` | `/project_settings/{source}` | `get_project_settings` | `str` | - | `project_settings_service.get_project_settings` |
| `POST` | `/project_settings/{source}` | `update_project_settings` | `ProjectSettingsUpdateRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.update_project_settings` |
| `POST` | `/project_settings/{source}/create_project_folder` | `create_project_folder` | `CreateProjectFolderRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.create_project_folder` |
| `POST` | `/project_settings/{source}/delete_project_folder` | `delete_project_folder` | `DeleteProjectFolderRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.delete_project_folder` |
| `POST` | `/project_settings/{source}/duplicate_project_folder` | `duplicate_project_folder` | `DuplicateProjectFolderRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.duplicate_project_folder` |
| `GET` | `/project_settings/{source}/folders` | `get_project_folders` | `str` | - | `project_settings_service.get_project_folders` |
| `POST` | `/project_settings/{source}/folders` | `update_project_folders` | `FolderStructureUpdateRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.update_project_folders` |
| `POST` | `/project_settings/{source}/open_project_folder` | `open_project_folder` | `OpenProjectFolderRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.open_project_folder` |
| `POST` | `/project_settings/{source}/rename_project_folder` | `rename_project_folder` | `RenameProjectFolderRequest` | [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) | `project_settings_service.rename_project_folder` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.project_settings.key_files -->
- [`backend/api/project_settings_router.py`](../../../backend/api/project_settings_router.py) - Project settings CRUD and folder ops routes.
- [`backend/services/project_settings_service.py`](../../../backend/services/project_settings_service.py) - Project settings persistence service.
- [`backend/schemas/project_settings.py`](../../../backend/schemas/project_settings.py) - Project settings request schemas.
- [`project_settings.js`](../../../project_settings.js) - Frontend caller for project settings endpoints.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Heavily used by `project_settings.js` UI flows.
- Provides `/general_settings` read/write for Source Data Origin/Development boundary values.
- Provides project-folder CRUD-style endpoints under `/project_settings/{source}/*_project_folder` (including empty-folder creation for new-project tree action).
- Provides `POST /project_settings/{source}/open_project_folder` for opening a selected project directory in the host OS file explorer.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Handles folder CRUD and settings JSON writes.
- Handles project folder open requests by resolving the project directory through config path helpers before launching OS file explorer.
- `create_project_folder` creates an empty project folder plus `data` subfolder; client coordinates rollback if later folder-structure/settings saves fail.
- Handles per-project `general_settings.json` persistence in each project folder.
- Normalizes stored Origin/Development boundary values to plain integer strings (no commas, no trailing `.0`/`.00`).
- Stores `auto_generated` in `general_settings.json`; backend writes `project_name` as current project folder name to detect stale duplicated files.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add source key support: update router path params + service source resolution.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Folder operation rollbacks can leave partial state when interrupted.
<!-- MANUAL:END -->
