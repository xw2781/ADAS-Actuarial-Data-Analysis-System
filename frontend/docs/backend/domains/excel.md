# Backend Domain: excel

## Purpose
<!-- MANUAL:BEGIN -->
Excel automation domain (selection reads and workbook operations).
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.excel.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `POST` | `/excel/active_selection` | `excel_active_selection` | - | - | `excel_service.excel_active_selection` |
| `POST` | `/excel/open_workbook` | `excel_open_workbook` | `ExcelOpenRequest` | [`backend/schemas/excel.py`](../../../backend/schemas/excel.py) | `excel_service.excel_open_workbook` |
| `POST` | `/excel/read_cell` | `excel_read_cell` | `ExcelCellReadRequest` | [`backend/schemas/excel.py`](../../../backend/schemas/excel.py) | `excel_service.excel_read_cell` |
| `POST` | `/excel/read_cells_batch` | `excel_read_cells_batch` | `ExcelBatchReadRequest` | [`backend/schemas/excel.py`](../../../backend/schemas/excel.py) | `excel_service.excel_read_cells_batch` |
| `POST` | `/excel/wait_for_enter` | `excel_wait_for_enter` | - | - | `excel_service.excel_wait_for_enter` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.excel.key_files -->
- [`backend/api/excel_router.py`](../../../backend/api/excel_router.py) - Excel COM automation routes.
- [`backend/services/excel_service.py`](../../../backend/services/excel_service.py) - Excel process interaction logic.
- [`backend/schemas/excel.py`](../../../backend/schemas/excel.py) - Excel request payload schemas.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Called by interactive Excel-based workflows.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Runtime depends on local Excel automation availability.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add automation method: schema + router + service must stay aligned.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Excel COM timing and environment dependencies are fragile.
<!-- MANUAL:END -->
