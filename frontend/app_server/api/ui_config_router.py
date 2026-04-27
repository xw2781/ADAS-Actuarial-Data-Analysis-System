from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app_server import config
from app_server.schemas.ui_config import UIConfigUpdateRequest

router = APIRouter()


@router.get("/ui_config")
def get_ui_config() -> Dict[str, Any]:
    cfg = config.load_ui_config()
    return {"ok": True, "config": cfg}


@router.post("/ui_config")
def update_ui_config(req: UIConfigUpdateRequest) -> Dict[str, Any]:
    cfg = config.load_ui_config()
    cfg["root_path"] = req.root_path
    try:
        with open(config.UI_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        config.refresh_runtime_paths()
        return {"ok": True, "config": cfg}
    except Exception as e:
        raise HTTPException(500, f"Failed to save config: {str(e)}")
