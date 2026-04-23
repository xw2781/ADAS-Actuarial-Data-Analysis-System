from typing import List, Optional

from pydantic import BaseModel, Field


class PatchItem(BaseModel):
    r: int = Field(..., ge=0)
    c: int = Field(..., ge=0)
    value: Optional[float] = None


class PatchRequest(BaseModel):
    items: List[PatchItem]
    file_mtime: Optional[float] = None


class DatasetNotesLoadRequest(BaseModel):
    project_name: str
    reserving_class: str
    dataset_name: str


class DatasetNotesSaveRequest(BaseModel):
    project_name: str
    reserving_class: str
    dataset_name: str
    notes: str = ""
