from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app_server import config
from app_server.schemas.audit_log import AuditLogWriteRequest
from app_server.services import audit_service

router = APIRouter()


@router.get("/audit_log")
def get_audit_log(project_name: str, limit: int = 500) -> Dict[str, Any]:
    project_name_clean = str(project_name or "").strip()
    if not project_name_clean:
        raise HTTPException(400, "Missing project_name parameter")

    try:
        out = audit_service.read_audit_log(project_name_clean, limit=limit)
        return {"ok": True, **out}
    except ValueError as e:
        raise HTTPException(404, str(e))


@router.post("/audit_log")
def write_audit_log(req: AuditLogWriteRequest) -> Dict[str, Any]:
    project_name = str(req.project_name or "").strip()
    action = str(req.action or "").strip()
    if not project_name:
        raise HTTPException(400, "project_name is required")
    if not action:
        raise HTTPException(400, "action is required")
    if len(action) > 2000:
        raise HTTPException(400, "action is too long")
    try:
        out = audit_service.append_project_audit_log(
            project_name=project_name,
            action=action,
            user_name=req.user_name,
        )
        return {"ok": True, **out}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except PermissionError:
        raise HTTPException(423, "Audit log file is locked. Another user may have it open.")
    except Exception as e:
        raise HTTPException(500, f"Failed to write audit log: {str(e)}")
