from __future__ import annotations

from dataclasses import dataclass

from ...core.models import ChatMessage


@dataclass(frozen=True)
class PersonalPeriod:
    messages: tuple[ChatMessage, ...]
    context: tuple[ChatMessage, ...]
    session_count: int = 1

    @property
    def start(self) -> int:
        return self.messages[0].created_at

    @property
    def end(self) -> int:
        return self.messages[-1].created_at


@dataclass(frozen=True)
class PersonalReport:
    target_id: str
    target_name: str
    messages: tuple[ChatMessage, ...]
    periods: tuple[PersonalPeriod, ...]
    mode: str
    detailed: bool = False
    transcript: tuple[ChatMessage, ...] = ()
    page: int = 1
    page_count: int = 1
    period_number: int = 0


@dataclass(frozen=True)
class PeriodAnalysis:
    title: str
    summary: str
    context: str
    quote: str = ""


@dataclass(frozen=True)
class PersonalAnalysis:
    overview: str
    periods: tuple[PeriodAnalysis, ...]
    source: str = "model"
