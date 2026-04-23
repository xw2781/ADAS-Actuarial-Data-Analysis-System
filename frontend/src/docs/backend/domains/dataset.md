# Backend Domain: dataset

## Purpose
<!-- MANUAL:BEGIN -->
Dataset retrieval/patch domain for in-memory dataset instances.
Also handles dataset Notes persistence files under each project `data` folder.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.dataset.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `POST` | `/dataset/notes/load` | `load_dataset_notes` | `DatasetNotesLoadRequest` | [`backend/schemas/dataset.py`](../../../backend/schemas/dataset.py) | `dataset_service.load_dataset_notes` |
| `POST` | `/dataset/notes/save` | `save_dataset_notes` | `DatasetNotesSaveRequest` | [`backend/schemas/dataset.py`](../../../backend/schemas/dataset.py) | `dataset_service.save_dataset_notes` |
| `GET` | `/dataset/{ds_id}` | `get_dataset` | `str` | - | `dataset_service.get_dataset` |
| `GET` | `/dataset/{ds_id}/diagonal` | `get_diagonal` | `str` | - | `dataset_service.get_diagonal` |
| `POST` | `/dataset/{ds_id}/patch` | `patch_dataset` | `PatchRequest` | [`backend/schemas/dataset.py`](../../../backend/schemas/dataset.py) | `dataset_service.patch_dataset` |
| `GET` | `/datasets` | `list_datasets` | - | - | `dataset_service.list_datasets` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.dataset.key_files -->
- [`backend/api/dataset_router.py`](../../../backend/api/dataset_router.py) - Dataset query/patch routes.
- [`backend/services/dataset_service.py`](../../../backend/services/dataset_service.py) - Dataset in-memory operations.
- [`backend/schemas/dataset.py`](../../../backend/schemas/dataset.py) - Dataset patch request model.
- [`api.js`](../../../api.js) - Frontend client wrapper for dataset API.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Called by dataset/DFM frontend flows via `api.js`.
- Exposes Notes load/save endpoints for project-scoped dataset notes persistence.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Uses in-memory dataset map and patch payloads.
- Persists dataset Notes as JSON files in `projects/<project>/data/ADASTriNotes@<reserving class path with "\" -> "^">@<dataset name>.json` (filesystem-unsafe filename characters are replaced with `^`).
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Change patch semantics: align schema, service patch rules, and frontend expectations.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Patch operations can introduce subtle data integrity issues.
<!-- MANUAL:END -->
