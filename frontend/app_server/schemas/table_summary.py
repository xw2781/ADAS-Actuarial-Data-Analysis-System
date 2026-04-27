from typing import Optional

from pydantic import BaseModel


class TableSummaryRefreshRequest(BaseModel):
    path: str
    project_name: Optional[str] = None
    refresh_reserving: bool = True
