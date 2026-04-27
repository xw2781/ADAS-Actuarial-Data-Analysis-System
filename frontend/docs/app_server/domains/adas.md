# App Server Domain: adas

## Purpose
<!-- MANUAL:BEGIN -->
ADAS calculations/precheck domain.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN app_server.adas.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `POST` | `/adas/headers` | `adas_headers` | `AdaHeadersRequest` | [`app_server/schemas/adas.py`](../../../app_server/schemas/adas.py) | `adas_service.adas_headers` |
| `POST` | `/adas/headers/cache/clear` | `clear_adas_headers_cache` | `AdaHeadersCacheClearRequest` | [`app_server/schemas/adas.py`](../../../app_server/schemas/adas.py) | `adas_service.clear_adas_headers_cache` |
| `GET` | `/adas/projects` | `adas_projects` | - | - | `adas_service.adas_projects` |
| `POST` | `/adas/tri` | `adas_tri` | `AdaTriRequest` | [`app_server/schemas/adas.py`](../../../app_server/schemas/adas.py) | `adas_service.run_adas_tri` |
| `POST` | `/adas/tri/precheck` | `adas_tri_precheck` | `AdaTriRequest` | [`app_server/schemas/adas.py`](../../../app_server/schemas/adas.py) | - |
| `POST` | `/adas/tri/refresh` | `adas_tri_refresh` | `AdaTriRequest` | [`app_server/schemas/adas.py`](../../../app_server/schemas/adas.py) | `adas_service.run_adas_tri` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN app_server.adas.key_files -->
- [`app_server/api/adas_router.py`](../../../app_server/api/adas_router.py) - ADAS tri/precheck/header endpoints.
- [`app_server/services/adas_service.py`](../../../app_server/services/adas_service.py) - ADAS processing and project listing.
- [`app_server/schemas/adas.py`](../../../app_server/schemas/adas.py) - ADAS request schemas.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Called by dataset/workflow actions requiring ADAS processing.
- Includes a cache-maintenance endpoint used by Project Settings reload to clear project-scoped `ADASHeaders*.csv` files.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Integrates headers/project listing and tri execution endpoints.
- Manages ADAS request-result CSV caches under each project `data` folder; supports targeted ADASHeaders cache clearing.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add new ADAS operation: keep precheck/execute contracts explicit.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Long-running computations need robust error messaging.
<!-- MANUAL:END -->
