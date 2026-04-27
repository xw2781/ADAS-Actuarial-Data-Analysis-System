# Backend Domain: field_mapping

## Purpose
<!-- MANUAL:BEGIN -->
Field mapping persistence domain for project settings.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.field_mapping.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/field_mapping` | `get_field_mapping` | `str` | - | - |
| `POST` | `/field_mapping` | `save_field_mapping` | `FieldMappingSaveRequest` | [`backend/schemas/field_mapping.py`](../../../backend/schemas/field_mapping.py) | `field_mapping_service.save_field_mapping` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.field_mapping.key_files -->
- [`backend/api/field_mapping_router.py`](../../../backend/api/field_mapping_router.py) - Field mapping read/save routes.
- [`backend/services/field_mapping_service.py`](../../../backend/services/field_mapping_service.py) - Field mapping persistence and validation.
- [`backend/schemas/field_mapping.py`](../../../backend/schemas/field_mapping.py) - Field mapping request schema.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Used by project settings field mapping feature.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Stores mapping files under project folders.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add mapping attributes: update schema, service validation, and UI module.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Invalid mappings propagate into reserving class/dataset processing.
<!-- MANUAL:END -->
