# App Server Domain: book

## Purpose
<!-- MANUAL:BEGIN -->
Workbook metadata/sheet/patch domain.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN app_server.book.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `POST` | `/book/meta` | `book_meta` | `AnyBookSheetRequest` | [`app_server/schemas/book.py`](../../../app_server/schemas/book.py) | `book_service.resolve_allowed_book` |
| `POST` | `/book/patch` | `book_patch` | `AnyBookPatchRequest` | [`app_server/schemas/book.py`](../../../app_server/schemas/book.py) | `book_service.resolve_allowed_book` |
| `POST` | `/book/sheet` | `book_sheet` | `AnyBookSheetRequest` | [`app_server/schemas/book.py`](../../../app_server/schemas/book.py) | `book_service.read_sheet_matrix`, `book_service.resolve_allowed_book` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN app_server.book.key_files -->
- [`app_server/api/book_router.py`](../../../app_server/api/book_router.py) - Workbook sheet/meta/patch routes.
- [`app_server/services/book_service.py`](../../../app_server/services/book_service.py) - Workbook data read/write helpers.
- [`app_server/schemas/book.py`](../../../app_server/schemas/book.py) - Workbook request schemas.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Shared by dataset-related frontend flows.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Reads/writes workbook content via service helpers.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add sheet operation: update router contract and service implementation.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Workbook file locking and formula behavior can vary by environment.
<!-- MANUAL:END -->
