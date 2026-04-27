from typing import List, Optional

from pydantic import BaseModel, Field


class FieldMappingRow(BaseModel):
    field_name: str
    significance: Optional[str] = None
    dataset_type: Optional[str] = None
    level: Optional[int] = Field(default=None, ge=1)


class FieldMappingSaveRequest(BaseModel):
    project_name: str
    table_path: Optional[str] = ""
    rows: List[FieldMappingRow] = Field(default_factory=list)
