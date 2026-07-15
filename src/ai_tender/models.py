from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Block(BaseModel):
    id: str
    file: str
    location: str
    kind: str
    text: str


class Requirement(BaseModel):
    id: str = ""
    text: str
    category: str = "прочее"
    parameter: str | None = None
    operator: str | None = None
    value: str | float | int | None = None
    unit: str | None = None
    mandatory: bool = True
    source_block_id: str
    source_file: str
    source_location: str


class Status(StrEnum):
    compliant = "compliant"
    non_compliant = "non_compliant"
    partial = "partial"
    insufficient_evidence = "insufficient_evidence"
    not_applicable = "not_applicable"


class Evidence(BaseModel):
    file: str
    location: str
    quote: str
    block_id: str


class Comparison(BaseModel):
    requirement: Requirement
    status: Status
    explanation: str
    reference_value: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)


class AnalysisReport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tender_path: str
    assets_path: str
    model: str
    comparisons: list[Comparison]
    warnings: list[str] = Field(default_factory=list)
    report_dir: Path | None = None
