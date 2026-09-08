from __future__ import annotations

import datetime as dt
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from .config import (
    GROUP_COMMAND_PATTERN,
    MAX_PAGE_SIZE,
    MAX_SCAN_PAGES,
    MAX_STORED_MESSAGES,
    PERSONAL_COMMAND_PATTERN,
    PLUGIN_MARKER,
)
from .dates import local_day_bounds, normalize_timestamp
from .event import (
    event_timestamp,
    extract_message_text,
    history_session_key,
    interaction_count,
    plain_command_text,
)
from .models import ChatMessage
from .text import normalize_text


def history_content(event: AstrMessageEvent, text: str) -> dict[str, Any]:
    return {
        "plugin": PLUGIN_MARKER,
        "schema_version": 1,
        "group_id": str(event.get_group_id()),
        "message_id": str(getattr(event.message_obj, "message_id", "") or ""),
        "sender_id": str(event.get_sender_id() or ""),
        "sender_name": normalize_text(event.get_sender_name()) or "未知成员",
        "message_text": text,
        "interactions": interaction_count(event),
        "created_at": event_timestamp(event),
    }


def record_from_row(
    row: Any, start: int = 0, end: float = float("inf")
) -> ChatMessage | None:
    content = getattr(row, "content", None) or {}
    if not isinstance(content, dict) or content.get("plugin") != PLUGIN_MARKER:
        return None
    row_created_at = getattr(row, "created_at", None)
    timestamp = normalize_timestamp(content.get("created_at"))
    if not timestamp and row_created_at is not None:
        timestamp = normalize_timestamp(row_created_at)
    if not timestamp or not start <= timestamp < end:
        return None
    text = normalize_text(content.get("message_text"))
    if not text:
        return None
    try:
        interactions = max(0, int(content.get("interactions", 0)))
    except (TypeError, ValueError):
        interactions = 0
    return ChatMessage(
        sender_id=str(
            content.get("sender_id") or getattr(row, "sender_id", "") or "未知"
        ),
        sender_name=normalize_text(
            content.get("sender_name") or getattr(row, "sender_name", "") or "未知成员"
        ),
        text=text,
        created_at=timestamp,
        interactions=interactions,
        message_id=str(content.get("message_id") or ""),
    )


class MessageHistoryStore:
    def __init__(self, manager: Any):
        self.manager = manager

    async def store(self, event: AstrMessageEvent) -> None:
        if not event.get_group_id():
            return
        sender_id = str(event.get_sender_id() or "")
        if sender_id and sender_id == str(event.get_self_id() or ""):
            return

        command_text = normalize_text(event.get_message_str())
        if GROUP_COMMAND_PATTERN.fullmatch(
            command_text
        ) or PERSONAL_COMMAND_PATTERN.fullmatch(plain_command_text(event)):
            return
        text = extract_message_text(event)
        if not text or GROUP_COMMAND_PATTERN.fullmatch(text):
            return

        content = history_content(event, text)
        insert_kwargs = {
            "platform_id": event.get_platform_id(),
            "user_id": history_session_key(event),
            "content": content,
            "sender_id": content["sender_id"],
            "sender_name": content["sender_name"],
            "max_messages": MAX_STORED_MESSAGES,
        }
        try:
            await self.manager.insert(**insert_kwargs)
        except TypeError as exc:
            if "max_messages" not in str(exc):
                raise
            insert_kwargs.pop("max_messages")
            await self.manager.insert(**insert_kwargs)

    async def load(
        self,
        event: AstrMessageEvent,
        target_date: dt.date | None = None,
        bounds: tuple[int, int] | None = None,
    ) -> list[ChatMessage]:
        start, end = bounds or (
            local_day_bounds(target_date)
            if target_date is not None
            else (0, float("inf"))
        )
        records: list[ChatMessage] = []
        seen: set[tuple[str, str, int, str]] = set()
        for page in range(1, MAX_SCAN_PAGES + 1):
            rows = await self.manager.get(
                platform_id=event.get_platform_id(),
                user_id=history_session_key(event),
                page=page,
                page_size=MAX_PAGE_SIZE,
            )
            if not rows:
                break
            for row in rows:
                record = record_from_row(row, start, end)
                if record is None:
                    continue
                identity = (
                    record.message_id,
                    record.sender_id,
                    record.created_at,
                    record.text,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(record)
            if len(rows) < MAX_PAGE_SIZE:
                break
        records.sort(key=lambda record: (record.created_at, record.message_id))
        return records

    async def track(self, event: AstrMessageEvent) -> None:
        try:
            await self.store(event)
        except Exception as exc:
            logger.warning(f"[group_summary] store message failed: {exc}")
