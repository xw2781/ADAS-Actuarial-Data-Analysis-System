from pydantic import BaseModel


class AdaTriRequest(BaseModel):
    Path: str
    TriangleName: str
    ProjectName: str
    Cumulative: bool = True
    OriginLength: int = 12
    DevelopmentLength: int = 12
    timeout_sec: float = 6.0


class AdaHeadersRequest(BaseModel):
    periodType: int = 0
    Transposed: bool = False
    PeriodLength: int = 12
    ProjectName: str
    StoredPeriodLength: int = -1
    timeout_sec: float = 6.0


class AdaHeadersCacheClearRequest(BaseModel):
    ProjectName: str
