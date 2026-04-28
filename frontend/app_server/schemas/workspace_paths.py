from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class WorkspacePaths(BaseModel):
    projects_dir: Optional[str] = None
    requests_dir: Optional[str] = None


class WorkspacePathsUpdateRequest(BaseModel):
    workspace_root: str
    paths: Optional[WorkspacePaths] = None

