"""Structured outputs for the inspection agent and the few-shot listener."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    laser_idx: int = Field(description="index of the laser point (0-99, row-major on the 10x10 grid)")
    band_hz: str = Field(description="frequency band like '200-400'")
    delta_db: float = Field(description="energy change vs the empty-box reference in dB (negative = damped)")
    note: str = ""


class InspectionReport(BaseModel):
    verdict: str = Field(description="'object present' | 'empty' | 'inconclusive'")
    location_cell: str = Field(description="estimated position like 'x=2.3, y=7.8' on the position grid, or 'n/a'")
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]
    recommended_action: str
    caveats: list[str]


class ListenerGuess(BaseModel):
    presence: bool
    x: float = Field(description="estimated object column position on the nx-wide position grid, -1 if empty")
    y: float = Field(description="estimated object row position on the ny-tall position grid, -1 if empty")
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class SelfAudit(BaseModel):
    agree: bool
    note: str
