from __future__ import annotations

import time

from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import At, Plain

from .config import MAX_MESSAGE_LENGTH
from .dates import normalize_timestamp
from .text import normalize_text


def plain_command_text(event: AstrMessageEvent) -> str:
    parts = [
        str(component.text)
        for component in event.get_messages()
        if isinstance(component, Plain)
    ]
    return " ".join(parts).strip() if parts else event.get_message_str().strip()


def history_session_key(event: AstrMessageEvent) -> str:
    return f"group_summary:group:{event.get_group_id()}"


def extract_message_text(event: AstrMessageEvent) -> str:
    outline = normalize_text(event.get_message_outline())
    if outline:
        return outline[:MAX_MESSAGE_LENGTH]
    return normalize_text(event.get_message_str())[:MAX_MESSAGE_LENGTH]


def interaction_count(event: AstrMessageEvent) -> int:
    count = 0
    for component in event.get_messages():
        component_name = component.__class__.__name__.lower()
        if component_name in {"at", "atall", "reply"}:
            count += 1
    return count


def event_timestamp(event: AstrMessageEvent) -> int:
    raw = getattr(event.message_obj, "timestamp", None)
    return normalize_timestamp(
        raw, fallback=int(getattr(event, "created_at", 0) or time.time())
    )


def mentioned_members(event: AstrMessageEvent) -> set[str]:
    return {
        str(component.qq).strip()
        for component in event.get_messages()
        if isinstance(component, At)
        and str(component.qq).strip() not in {"", "all", str(event.get_self_id())}
    }
