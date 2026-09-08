from __future__ import annotations

import datetime as dt
from collections import Counter
from typing import Iterable

from .models import ChatMessage, ReportStats
from .text import SPACE_RE


def compute_stats(
    messages: Iterable[ChatMessage],
    target_date: dt.date,
    timezone: dt.tzinfo | None = None,
) -> ReportStats:
    timezone = timezone or dt.datetime.now().astimezone().tzinfo
    records = list(messages)
    hourly = [0] * 24
    member_counter: Counter[str] = Counter()
    member_names: dict[str, str] = {}
    interactions = 0
    characters = 0

    for message in records:
        hour = dt.datetime.fromtimestamp(message.created_at, tz=timezone).hour
        hourly[hour] += 1
        member_counter[message.sender_id] += 1
        member_names[message.sender_id] = message.sender_name
        interactions += max(0, message.interactions)
        characters += len(SPACE_RE.sub("", message.text))

    busiest_hour = None
    if records:
        busiest_hour = max(range(24), key=lambda hour: (hourly[hour], -hour))

    ranked_members = tuple(
        (sender_id, member_names.get(sender_id, "未知成员"), count)
        for sender_id, count in member_counter.most_common()
    )
    return ReportStats(
        target_date=target_date,
        message_count=len(records),
        active_member_count=len(member_counter),
        interaction_count=interactions,
        character_count=characters,
        busiest_hour=busiest_hour,
        hourly_counts=tuple(hourly),
        member_counts=ranked_members,
    )
