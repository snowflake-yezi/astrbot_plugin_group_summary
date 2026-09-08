from __future__ import annotations

import hashlib
import json
from typing import Any

from astrbot.api.event import AstrMessageEvent

from ...core.config import PLUGIN_MARKER
from .models import PersonalReport


class PersonalIndexStore:
    def __init__(self, manager: Any):
        self.manager = manager

    @staticmethod
    def key(event: AstrMessageEvent, target_id: str) -> str:
        identity = [str(event.get_group_id()), str(event.get_sender_id()), target_id]
        digest = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return f"group_summary:personal_index:{digest}"

    async def save(
        self, event: AstrMessageEvent, key: str, report: PersonalReport
    ) -> None:
        kwargs = {
            "platform_id": event.get_platform_id(),
            "user_id": key,
            "sender_id": str(event.get_sender_id()),
            "sender_name": event.get_sender_name(),
            "max_messages": 1,
            "content": {
                "plugin": PLUGIN_MARKER,
                "kind": "personal_index",
                "schema_version": 1,
                "target_id": report.target_id,
                "periods": [
                    {
                        "start": period.start,
                        "end": period.end,
                        "count": len(period.messages),
                    }
                    for period in report.periods
                ],
            },
        }
        try:
            await self.manager.insert(**kwargs)
        except TypeError as exc:
            if "max_messages" not in str(exc):
                raise
            kwargs.pop("max_messages")
            await self.manager.insert(**kwargs)

    async def load_period(
        self, event: AstrMessageEvent, key: str, target_id: str, number: int
    ):
        rows = await self.manager.get(
            platform_id=event.get_platform_id(),
            user_id=key,
            page=1,
            page_size=1,
        )
        if not rows:
            raise ValueError("还没有这位群友的时段索引，请先发送：个人聊天记录 @群友。")
        content = getattr(rows[0], "content", None)
        if (
            not isinstance(content, dict)
            or content.get("kind") != "personal_index"
            or content.get("target_id") != target_id
        ):
            raise ValueError("时段索引不可用，请重新发送：个人聊天记录 @群友。")
        periods = content.get("periods", [])
        if not isinstance(periods, list) or not 1 <= number <= len(periods):
            raise ValueError(
                f"时段编号应在 1 到 {len(periods) if isinstance(periods, list) else 0} 之间，请参考最近一次个人总览。"
            )
        period = periods[number - 1]
        if not isinstance(period, dict) or any(
            type(period.get(field)) is not int for field in ("start", "end", "count")
        ):
            raise ValueError("时段索引不可用，请重新生成个人总览。")
        return period
