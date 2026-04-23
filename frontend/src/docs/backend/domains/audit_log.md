# Backend Domain: audit_log

## Purpose
<!-- MANUAL:BEGIN -->
Audit log read/write domain for project actions.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.audit_log.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/audit_log` | `get_audit_log` | `str` | - | `audit_service.read_audit_log` |
| `POST` | `/audit_log` | `write_audit_log` | `AuditLogWriteRequest` | [`backend/schemas/audit_log.py`](../../../backend/schemas/audit_log.py) | `audit_service.append_project_audit_log` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.audit_log.key_files -->
- [`backend/api/audit_log_router.py`](../../../backend/api/audit_log_router.py) - Audit read/write routes.
- [`backend/services/audit_service.py`](../../../backend/services/audit_service.py) - Audit persistence helpers and locking.
- [`backend/schemas/audit_log.py`](../../../backend/schemas/audit_log.py) - Audit write payload schema.
- [`backend/config.py`](../../../backend/config.py) - Audit file constants and lock objects.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Called from settings/type update flows.
- Service enforces safe append logic.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Stores rolling JSON audit records with lock protection.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add audit event fields: update schema and writer helper together.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Lock/file contention may surface under concurrent writes.
<!-- MANUAL:END -->
