from pydantic import BaseModel


class UIConfigUpdateRequest(BaseModel):
    root_path: str
