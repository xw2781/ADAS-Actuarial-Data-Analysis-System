# Backend Domain: workflow

## Purpose
<!-- MANUAL:BEGIN -->
Workflow file save/load domain.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN backend.workflow.entry_points -->
| Method | Path | Handler | Request Model | Schema | Service Calls |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/template/default_dir` | `template_default_dir` | - | - | `workflow_service.get_template_default_dir` |
| `GET` | `/workflow/default_dir` | `workflow_default_dir` | - | - | `workflow_service.get_workflow_default_dir` |
| `POST` | `/workflow/load` | `workflow_load` | `WorkflowLoadRequest` | [`backend/schemas/workflow.py`](../../../backend/schemas/workflow.py) | `workflow_service.load_workflow` |
| `POST` | `/workflow/save` | `workflow_save` | `WorkflowSaveRequest` | [`backend/schemas/workflow.py`](../../../backend/schemas/workflow.py) | `workflow_service.save_workflow` |
| `POST` | `/workflow/save_as` | `workflow_save_as` | `WorkflowSaveAsRequest` | [`backend/schemas/workflow.py`](../../../backend/schemas/workflow.py) | `workflow_service.save_workflow_as` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN backend.workflow.key_files -->
- [`backend/api/workflow_router.py`](../../../backend/api/workflow_router.py) - HTTP routes for workflow save/load/default dirs.
- [`backend/services/workflow_service.py`](../../../backend/services/workflow_service.py) - Workflow file I/O operations.
- [`backend/schemas/workflow.py`](../../../backend/schemas/workflow.py) - Workflow request models.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Consumed primarily by `workflow_main.js`.
- Uses typed request models in `backend/schemas/workflow.py`.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Reads/writes workflow files under configured workflow directory.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add a workflow route: update router + schema + service.
2. Keep backward compatibility when changing saved payload shape.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- File I/O errors and path permissions are common failure modes.
<!-- MANUAL:END -->
