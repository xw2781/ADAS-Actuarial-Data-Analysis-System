# Backend Domain: dataset_types

## Purpose
<!-- MANUAL:BEGIN -->
Dataset types catalog domain.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.dataset_types.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/dataset_types` | `get_dataset_types` | `str` | - | `dataset_types_service.normalize_dataset_types_data` |
| `POST` | `/dataset_types` | `save_dataset_types` | `DatasetTypesSaveRequest` | [`backend/schemas/dataset_types.py`](../../../backend/schemas/dataset_types.py) | `audit_service.safe_append_project_audit_log`, `dataset_types_service._build_dataset_source_resolver`, `dataset_types_service._extract_formula_components`, `dataset_types_service._load_dataset_source_map`, `dataset_types_service.save_dataset_types_payload` |
| `POST` | `/dataset_types/import_local_file` | `import_local_dataset_types_file` | `DatasetTypesImportLocalFileRequest` | [`backend/schemas/dataset_types.py`](../../../backend/schemas/dataset_types.py) | `dataset_types_service.parse_local_dataset_types_file` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.dataset_types.key_files -->
- [`backend/api/dataset_types_router.py`](../../../backend/api/dataset_types_router.py) - Dataset type catalog read/save routes.
- [`backend/services/dataset_types_service.py`](../../../backend/services/dataset_types_service.py) - Dataset type storage and normalization.
- [`backend/schemas/dataset_types.py`](../../../backend/schemas/dataset_types.py) - Dataset type save schema.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Used by project settings dataset types panel and dependent flows.
- `GET /dataset_types` keeps normalized 5-column `data.columns/data.rows` for compatibility and also returns `data.source_by_name` (dataset name -> Source expression) derived from project `dataset_types.json`.
- `POST /dataset_types/import_local_file` parses local `.json`/`.xlsx` Dataset Types files for UI-side local import; for `.xlsx`, parser assumes one sheet and the header exactly matches JSON column layout (supports 5-column local format or 6-column persisted format with trailing `Source`).
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Persists dataset type definitions under project folders.
- `POST /dataset_types` saves both `dataset_types.json` and same-folder `dataset_types.xlsx` with matching columns/rows (`Name`, `Data Format`, `Category`, `Calculated`, `Formula`, `Source`); XLSX header row is bold and column widths are auto-sized from header + cell contents (bounded min/max).
- `GET /dataset_types` source metadata extraction is backward-compatible with legacy files: `source_by_name` reads from a `Source` column when present or falls back to row index 5 for older row layouts.
- Save validation treats calculated formulas as valid when referenced components resolve to Dataset Type `Name` values; it no longer requires field-mapping source resolution for those components.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add type metadata field: align schema, service normalization, and frontend editor.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Type schema drift can break downstream interpretation logic.
<!-- MANUAL:END -->
