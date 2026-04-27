from __future__ import annotations

import hashlib
import os
from typing import Any, Dict

from fastapi import APIRouter

from app_server.schemas.adas import AdaTriRequest, AdaHeadersRequest, AdaHeadersCacheClearRequest
from app_server.helpers import set_data_path_like_vba
from app_server.services import adas_service

router = APIRouter()


@router.post("/adas/headers")
def adas_headers(req: AdaHeadersRequest) -> Dict[str, Any]:
    pairs = [
        ("Function", "ADASHeaders"),
        ("periodType", str(req.periodType)),
        ("Transposed", str(req.Transposed)),
        ("PeriodLength", str(req.PeriodLength)),
        ("ProjectName", req.ProjectName),
        ("StoredPeriodLength", str(req.StoredPeriodLength)),
    ]
    return adas_service.adas_headers(pairs, timeout_sec=max(0.1, float(req.timeout_sec)))


@router.post("/adas/headers/cache/clear")
def clear_adas_headers_cache(req: AdaHeadersCacheClearRequest) -> Dict[str, Any]:
    return adas_service.clear_adas_headers_cache(req.ProjectName)


@router.get("/adas/projects")
def adas_projects() -> Dict[str, Any]:
    return adas_service.adas_projects()


@router.post("/adas/tri/precheck")
def adas_tri_precheck(req: AdaTriRequest) -> Dict[str, Any]:
    pairs = [
        ("Function", "ADASTri"),
        ("Path", req.Path),
        ("DatasetName", req.TriangleName),
        ("Cumulative", str(req.Cumulative)),
        ("Transposed", str(False)),
        ("Calendar", str(False)),
        ("ProjectName", req.ProjectName),
        ("OriginLength", str(req.OriginLength)),
        ("DevelopmentLength", str(req.DevelopmentLength)),
    ]
    data_path = set_data_path_like_vba(pairs)
    need_request = not os.path.exists(data_path)
    ds_id = "adastri_" + hashlib.sha1(data_path.encode("utf-8")).hexdigest()[:16]
    return {
        "ok": True,
        "need_request": need_request,
        "cache_exists": (not need_request),
        "data_path": data_path,
        "ds_id": ds_id,
    }


@router.post("/adas/tri")
def adas_tri(req: AdaTriRequest) -> Dict[str, Any]:
    pairs = [
        ("Function", "ADASTri"),
        ("Path", req.Path),
        ("DatasetName", req.TriangleName),
        ("Cumulative", str(req.Cumulative)),
        ("Transposed", str(False)),
        ("Calendar", str(False)),
        ("ProjectName", req.ProjectName),
        ("OriginLength", str(req.OriginLength)),
        ("DevelopmentLength", str(req.DevelopmentLength)),
    ]
    data_path = set_data_path_like_vba(pairs)
    return adas_service.run_adas_tri(pairs, data_path, timeout_sec=max(0.1, float(req.timeout_sec)), force_refresh=False)


@router.post("/adas/tri/refresh")
def adas_tri_refresh(req: AdaTriRequest) -> Dict[str, Any]:
    pairs = [
        ("Function", "ADASTri"),
        ("Path", req.Path),
        ("DatasetName", req.TriangleName),
        ("Cumulative", str(req.Cumulative)),
        ("Transposed", str(False)),
        ("Calendar", str(False)),
        ("ProjectName", req.ProjectName),
        ("OriginLength", str(req.OriginLength)),
        ("DevelopmentLength", str(req.DevelopmentLength)),
    ]
    data_path = set_data_path_like_vba(pairs)
    return adas_service.run_adas_tri(pairs, data_path, timeout_sec=max(0.1, float(req.timeout_sec)), force_refresh=True)
