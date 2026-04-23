from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class WorkflowSaveRequest(BaseModel):
    name: str = ""
    prev_path: Optional[str] = None
    data: Dict[str, Any]


class WorkflowSaveAsRequest(BaseModel):
    path: str
    data: Dict[str, Any]


class WorkflowLoadRequest(BaseModel):
    path: str
