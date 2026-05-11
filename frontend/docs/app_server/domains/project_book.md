# App Server Domain: project_book

## Purpose
<!-- MANUAL:BEGIN -->
Project workbook domain resolved by project name and source folders.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN app_server.project_book.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/project_book/meta` | `project_book_meta` | - | - | `book_service._load_project_map_data`, `book_service._project_map_sheet_names` |
| `POST` | `/project_book/patch` | `project_book_patch` | `XlsmPatchRequest` | [`app_server/schemas/book.py`](../../../app_server/schemas/book.py) | `book_service._load_project_map_data`, `book_service._project_map_sheet_names` |
| `GET` | `/project_book/sheet` | `project_book_sheet` | `str` | - | `book_service._read_project_map_sheet_matrix` |
| `GET` | `/project_settings` | `list_project_settings_sources` | - | - | `project_settings_service.list_project_settings_sources` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN app_server.project_book.key_files -->
- [`app_server/api/project_book_router.py`](../../../app_server/api/project_book_router.py) - Project workbook metadata/sheet/patch routes.
- [`app_server/services/book_service.py`](../../../app_server/services/book_service.py) - Workbook patching implementation.
- [`app_server/services/project_settings_service.py`](../../../app_server/services/project_settings_service.py) - Project-folder path resolution.
- [`app_server/schemas/book.py`](../../../app_server/schemas/book.py) - Project workbook patch schema.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Used by project settings/dataset flows for project-specific workbook operations.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Depends on project settings path resolution.
- Project map rows are expected to carry `Project Name` and `Table Path`; folder placement lives in `folder_structure.json`; legacy `Folder`, `Preload`, `Project Settings`, and `Settings Profile` columns are no longer consumed by Project Settings saves.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Change project-book lookup rules: update router checks and service path resolvers.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Mismatched source/folder mappings can route to wrong files.
<!-- MANUAL:END -->
