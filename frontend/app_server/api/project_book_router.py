from __future__ import annotations

import os
import json
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app_server import config
from app_server.schemas.book import XlsmPatchRequest
from app_server.services import book_service, project_settings_service

router = APIRouter()


@router.get("/project_settings")
def list_project_settings_sources() -> Dict[str, Any]:
    return project_settings_service.list_project_settings_sources()


@router.get("/project_book/meta")
def project_book_meta() -> Dict[str, Any]:
    if not os.path.exists(config.PROJECT_BOOK):
        raise HTTPException(404, f"Project map file not found: {config.PROJECT_BOOK}")

    data = book_service._load_project_map_data(config.PROJECT_BOOK)
    sheets = book_service._project_map_sheet_names(data)
    st = os.stat(config.PROJECT_BOOK)
    return {
        "path": config.PROJECT_BOOK,
        "mtime": st.st_mtime,
        "sheets": sheets,
    }


@router.get("/project_book/sheet")
def project_book_sheet(sheet: str) -> Dict[str, Any]:
    if not os.path.exists(config.PROJECT_BOOK):
        raise HTTPException(404, f"Project map file not found: {config.PROJECT_BOOK}")

    st = os.stat(config.PROJECT_BOOK)
    values = book_service._read_project_map_sheet_matrix(config.PROJECT_BOOK, sheet_name=sheet)
    return {"sheet": sheet, "values": values, "mtime": st.st_mtime}


@router.post("/project_book/patch")
def project_book_patch(req: XlsmPatchRequest) -> Dict[str, Any]:
    if not os.path.exists(config.PROJECT_BOOK):
        raise HTTPException(404, f"Project map file not found: {config.PROJECT_BOOK}")

    st = os.stat(config.PROJECT_BOOK)
    if req.file_mtime is not None and abs(st.st_mtime - req.file_mtime) > 1e-6:
        raise HTTPException(409, "Project map file changed on disk. Reload and retry.")

    try:
        data = book_service._load_project_map_data(config.PROJECT_BOOK)
        sheet_names = book_service._project_map_sheet_names(data)
        if req.sheet not in sheet_names:
            raise HTTPException(404, f"Sheet not found: {req.sheet}")

        sheet_obj = data.get(req.sheet) or {}
        headers = list(sheet_obj.get("headers") or [])
        rows = [list(r) if isinstance(r, list) else [] for r in (sheet_obj.get("rows") or [])]
        matrix: List[List[Any]] = [headers] + rows

        applied = 0
        rejected = []

        for it in req.items:
            rr = it.r
            cc = it.c
            if rr < 0 or cc < 0:
                rejected.append({"r": it.r, "c": it.c, "reason": "out_of_range"})
                continue

            while len(matrix) <= rr:
                matrix.append([])
            while len(matrix[rr]) <= cc:
                matrix[rr].append(None)

            matrix[rr][cc] = it.value
            applied += 1

        if matrix:
            sheet_obj["headers"] = list(matrix[0])
            sheet_obj["rows"] = [list(r) for r in matrix[1:]]
        else:
            sheet_obj["headers"] = []
            sheet_obj["rows"] = []
        data[req.sheet] = sheet_obj

        tmp_path = config.PROJECT_BOOK + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, config.PROJECT_BOOK)

        st2 = os.stat(config.PROJECT_BOOK)
        return {"ok": True, "applied": applied, "rejected": rejected, "mtime": st2.st_mtime}

    except PermissionError:
        raise HTTPException(423, "Project map file is locked. Close it and retry.")
