from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    sender_id: str
    sender_name: str
    text: str
    created_at: int
    interactions: int = 0
    message_id: str = ""


@dataclass(frozen=True)
class ReportStats:
    target_date: dt.date
    message_count: int
    active_member_count: int
    interaction_count: int
    character_count: int
    busiest_hour: int | None
    hourly_counts: tuple[int, ...]
    member_counts: tuple[tuple[str, str, int], ...]
