from __future__ import annotations

import datetime as dt
import re
from typing import Any

_DATE_PATTERNS = (
    re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$"),
    re.compile(r"^(\d{4})年(\d{1,2})月(\d{1,2})日?$"),
    re.compile(r"^(\d{4})(\d{2})(\d{2})$"),
)


_SHORT_DATE_PATTERN = re.compile(r"^(\d{1,2})月(\d{1,2})日?$")


def parse_target_date(value: str | None, today: dt.date | None = None) -> dt.date:
    today = today or dt.datetime.now().astimezone().date()
    raw = (value or "").strip()
    if not raw or raw == "今天":
        return today
    if raw == "昨天":
        return today - dt.timedelta(days=1)
    if raw == "前天":
        return today - dt.timedelta(days=2)

    for pattern in _DATE_PATTERNS:
        match = pattern.fullmatch(raw)
        if match:
            return dt.date(*(int(part) for part in match.groups()))

    match = _SHORT_DATE_PATTERN.fullmatch(raw)
    if match:
        return dt.date(today.year, int(match.group(1)), int(match.group(2)))

    raise ValueError("日期格式无效")


def local_day_bounds(
    target_date: dt.date, timezone: dt.tzinfo | None = None
) -> tuple[int, int]:
    timezone = timezone or dt.datetime.now().astimezone().tzinfo
    start = dt.datetime.combine(target_date, dt.time.min, tzinfo=timezone)
    end = start + dt.timedelta(days=1)
    return int(start.timestamp()), int(end.timestamp())


def normalize_timestamp(value: Any, fallback: int = 0) -> int:
    if isinstance(value, dt.datetime):
        timestamp = value.timestamp()
    else:
        try:
            timestamp = float(value)
        except (TypeError, ValueError, OverflowError):
            return fallback

    if timestamp > 10_000_000_000:
        timestamp /= 1000
    if timestamp <= 0:
        return fallback
    return int(timestamp)
