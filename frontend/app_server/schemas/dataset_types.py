from typing import Any, List

from pydantic import BaseModel, Field


class DatasetTypesSaveRequest(BaseModel):
    project_name: str
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)


class DatasetTypesImportLocalFileRequest(BaseModel):
    file_path: str
