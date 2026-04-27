from typing import Any, List, Optional

from pydantic import BaseModel, Field


class XlsmCellPatch(BaseModel):
    r: int = Field(..., ge=0)   # 0-based
    c: int = Field(..., ge=0)   # 0-based
    value: Any = None


class XlsmPatchRequest(BaseModel):
    sheet: str
    items: List[XlsmCellPatch]
    file_mtime: Optional[float] = None


class AnyBookSheetRequest(BaseModel):
    book_path: str
    sheet: str


class AnyBookPatchRequest(BaseModel):
    book_path: str
    sheet: str
    items: List[XlsmCellPatch]
    file_mtime: Optional[float] = None
