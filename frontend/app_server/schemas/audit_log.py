from typing import Optional

from pydantic import BaseModel


class AuditLogWriteRequest(BaseModel):
    project_name: str
    action: str
    user_name: Optional[str] = None
