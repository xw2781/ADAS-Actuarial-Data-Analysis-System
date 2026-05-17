from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel


class ScriptRunRequest(BaseModel):
    code: str


class ScriptDeleteVarRequest(BaseModel):
    name: str


class ScriptNotebookSaveRequest(BaseModel):
    filename: str
    cells: List[Dict[str, Any]]


class ScriptNotebookLoadRequest(BaseModel):
    filename: str


class ScriptInspectRequest(BaseModel):
    code: str
    cursor_pos: int
