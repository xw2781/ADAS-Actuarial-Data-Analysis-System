from __future__ import annotations

import os
import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from backend import config
from backend.schemas.field_mapping import FieldMappingSaveRequest
from backend.services import field_mapping_service

router = APIRouter()


@router.get("/field_mapping")
def get_field_mapping(project_name: str) -> Dict[str, Any]:
    if not project_name or not project_name.strip():
        raise HTTPException(400, "Missing project_name parameter")

    try:
        filepath = config.get_field_mapping_path(project_name)
    except ValueError as e:
        raise HTTPException(404, str(e))

    if not os.path.exists(filepath):
        return {
            "ok": True,
            "exists": False,
            "path": filepath,
            "data": {"project_name": project_name, "rows": []},
        }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {"project_name": project_name, "rows": []}
        if not isinstance(data.get("rows"), list):
            data["rows"] = []
        return {"ok": True, "exists": True, "path": filepath, "data": data}
    except Exception as e:
        raise HTTPException(500, f"Failed to read field mapping: {str(e)}")


@router.post("/field_mapping")
def save_field_mapping(req: FieldMappingSaveRequest) -> Dict[str, Any]:
    return field_mapping_service.save_field_mapping(
        project_name=req.project_name,
        table_path=req.table_path,
        rows=req.rows,
    )
