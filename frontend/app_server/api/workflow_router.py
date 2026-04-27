from __future__ import annotations

from typing import Any, Dict, Tuple

from fastapi import APIRouter

from app_server.schemas.workflow import WorkflowLoadRequest, WorkflowSaveAsRequest, WorkflowSaveRequest
from app_server.services import workflow_service


router = APIRouter()


def _workflow_context() -> Tuple[str, str]:
    from app_server import config

    return config.WORKFLOW_DIR, config.WORKFLOW_EXT


@router.post("/workflow/save")
def workflow_save(req: WorkflowSaveRequest) -> Dict[str, Any]:
    workflow_dir, workflow_ext = _workflow_context()
    return workflow_service.save_workflow(req, workflow_dir=workflow_dir, workflow_ext=workflow_ext)


@router.post("/workflow/save_as")
def workflow_save_as(req: WorkflowSaveAsRequest) -> Dict[str, Any]:
    workflow_dir, workflow_ext = _workflow_context()
    return workflow_service.save_workflow_as(req, workflow_dir=workflow_dir, workflow_ext=workflow_ext)


@router.get("/workflow/default_dir")
def workflow_default_dir() -> Dict[str, Any]:
    workflow_dir, _ = _workflow_context()
    return workflow_service.get_workflow_default_dir(workflow_dir)


@router.get("/template/default_dir")
def template_default_dir() -> Dict[str, Any]:
    return workflow_service.get_template_default_dir()


@router.post("/workflow/load")
def workflow_load(req: WorkflowLoadRequest) -> Dict[str, Any]:
    workflow_dir, workflow_ext = _workflow_context()
    return workflow_service.load_workflow(req, workflow_dir=workflow_dir, workflow_ext=workflow_ext)
