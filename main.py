from __future__ import annotations

import datetime as dt
import re
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .report import (
    SYSTEM_PROMPT,
    ChatMessage,
    build_analysis_prompt,
    build_fallback_analysis,
    compute_stats,
    format_text_report,
    local_day_bounds,
    merge_with_fallback,
    normalize_timestamp,
    parse_model_analysis,
    parse_target_date,
    render_report,
)


class GroupSummaryPlugin(Star):
    """Record group messages and generate a daily group-chat report."""

    PLUGIN_MARKER = "group_summary"
    MAX_PAGE_SIZE = 200
    MAX_SCAN_PAGES = 100
    MAX_STORED_MESSAGES = 20_000
    MAX_MESSAGE_LENGTH = 2_000
    COMMAND_PATTERN = re.compile(r"^/?群聊总结(?:\s+(.+?))?\s*$")

    def __init__(self, context: Context):
        super().__init__(context)
        self.context = context

    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        return f"group_summary:group:{event.get_group_id()}"

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[^\S\n]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_message_text(self, event: AstrMessageEvent) -> str:
        outline = self._normalize_text(event.get_message_outline())
        if outline:
            return outline[: self.MAX_MESSAGE_LENGTH]
        return self._normalize_text(event.get_message_str())[: self.MAX_MESSAGE_LENGTH]

    @staticmethod
    def _interaction_count(event: AstrMessageEvent) -> int:
        count = 0
        for component in event.get_messages():
            component_name = component.__class__.__name__.lower()
            if component_name in {"at", "atall", "reply"}:
                count += 1
        return count

    @staticmethod
    def _event_timestamp(event: AstrMessageEvent) -> int:
        raw = getattr(event.message_obj, "timestamp", None)
        return normalize_timestamp(raw, fallback=int(getattr(event, "created_at", 0) or time.time()))

    def _history_content(self, event: AstrMessageEvent, text: str) -> dict[str, Any]:
        return {
            "plugin": self.PLUGIN_MARKER,
            "schema_version": 1,
            "group_id": str(event.get_group_id()),
            "message_id": str(getattr(event.message_obj, "message_id", "") or ""),
            "sender_id": str(event.get_sender_id() or ""),
            "sender_name": self._normalize_text(event.get_sender_name()) or "未知成员",
            "message_text": text,
            "interactions": self._interaction_count(event),
            "created_at": self._event_timestamp(event),
        }

    async def _store_message(self, event: AstrMessageEvent) -> None:
        if not event.get_group_id():
            return
        sender_id = str(event.get_sender_id() or "")
        if sender_id and sender_id == str(event.get_self_id() or ""):
            return

        command_text = self._normalize_text(event.get_message_str())
        if self.COMMAND_PATTERN.fullmatch(command_text):
            return
        text = self._extract_message_text(event)
        if not text or self.COMMAND_PATTERN.fullmatch(text):
            return

        content = self._history_content(event, text)
        insert_kwargs = {
            "platform_id": event.get_platform_id(),
            "user_id": self._session_key(event),
            "content": content,
            "sender_id": content["sender_id"],
            "sender_name": content["sender_name"],
            "max_messages": self.MAX_STORED_MESSAGES,
        }
        try:
            await self.context.message_history_manager.insert(**insert_kwargs)
        except TypeError as exc:
            if "max_messages" not in str(exc):
                raise
            insert_kwargs.pop("max_messages")
            await self.context.message_history_manager.insert(**insert_kwargs)

    def _record_from_row(self, row: Any) -> ChatMessage | None:
        content = getattr(row, "content", None) or {}
        if not isinstance(content, dict) or content.get("plugin") != self.PLUGIN_MARKER:
            return None
        text = self._normalize_text(content.get("message_text"))
        if not text:
            return None
        row_created_at = getattr(row, "created_at", None)
        timestamp = normalize_timestamp(content.get("created_at"))
        if not timestamp and row_created_at is not None:
            timestamp = normalize_timestamp(row_created_at)
        if not timestamp:
            return None
        try:
            interactions = max(0, int(content.get("interactions", 0)))
        except (TypeError, ValueError):
            interactions = 0
        return ChatMessage(
            sender_id=str(content.get("sender_id") or getattr(row, "sender_id", "") or "未知"),
            sender_name=self._normalize_text(
                content.get("sender_name") or getattr(row, "sender_name", "") or "未知成员"
            ),
            text=text,
            created_at=timestamp,
            interactions=interactions,
            message_id=str(content.get("message_id") or ""),
        )

    async def _load_messages(self, event: AstrMessageEvent, target_date: dt.date) -> list[ChatMessage]:
        start, end = local_day_bounds(target_date)
        records: list[ChatMessage] = []
        seen: set[tuple[str, str, int, str]] = set()
        for page in range(1, self.MAX_SCAN_PAGES + 1):
            rows = await self.context.message_history_manager.get(
                platform_id=event.get_platform_id(),
                user_id=self._session_key(event),
                page=page,
                page_size=self.MAX_PAGE_SIZE,
            )
            if not rows:
                break
            for row in rows:
                record = self._record_from_row(row)
                if record is None or not start <= record.created_at < end:
                    continue
                identity = (record.message_id, record.sender_id, record.created_at, record.text)
                if identity in seen:
                    continue
                seen.add(identity)
                records.append(record)
            if len(rows) < self.MAX_PAGE_SIZE:
                break
        records.sort(key=lambda record: (record.created_at, record.message_id))
        return records

    async def _get_provider(self, event: AstrMessageEvent):
        get_async = getattr(self.context, "get_using_provider_async", None)
        if callable(get_async):
            return await get_async(umo=event.unified_msg_origin)
        get_sync = getattr(self.context, "get_using_provider", None)
        if callable(get_sync):
            return get_sync(umo=event.unified_msg_origin)
        return None

    async def _analyze(self, event: AstrMessageEvent, messages: list[ChatMessage], stats):
        fallback = build_fallback_analysis(messages, stats)
        provider = await self._get_provider(event)
        if provider is None:
            return build_fallback_analysis(messages, stats, "未配置聊天模型")

        response = await provider.text_chat(
            prompt=build_analysis_prompt(messages, stats),
            contexts=[],
            image_urls=[],
            system_prompt=SYSTEM_PROMPT,
        )
        completion = str(getattr(response, "completion_text", "") or "").strip()
        if not completion:
            raise ValueError("聊天模型返回了空内容")
        analysis = parse_model_analysis(completion, messages)
        return merge_with_fallback(analysis, fallback)

    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.regex(r".*")
    async def track_group_messages(self, event: AstrMessageEvent):
        try:
            await self._store_message(event)
        except Exception as exc:
            logger.warning(f"[group_summary] store message failed: {exc}")
        return
        yield

    @filter.regex(r"^/?群聊总结(?:\s+.+)?\s*$")
    async def group_summary(self, event: AstrMessageEvent):
        if not event.get_group_id():
            yield event.plain_result("群聊总结仅支持在群聊中使用。")
            return

        match = self.COMMAND_PATTERN.fullmatch(event.get_message_str().strip())
        date_text = match.group(1).strip() if match and match.group(1) else ""
        try:
            target_date = parse_target_date(date_text)
        except ValueError:
            yield event.plain_result(
                "日期格式无法识别。可用示例：群聊总结、群聊总结 昨天、群聊总结 2026-09-01、群聊总结 9月1日。"
            )
            return

        should_call_llm = getattr(event, "should_call_llm", None)
        if callable(should_call_llm):
            should_call_llm(False)

        try:
            messages = await self._load_messages(event, target_date)
        except Exception as exc:
            logger.warning(f"[group_summary] load messages failed: {exc}")
            yield event.plain_result("群聊记录读取失败，请稍后重试或查看 AstrBot 日志。")
            return

        if not messages:
            if target_date == dt.datetime.now().astimezone().date():
                detail = "今天还没有可总结的群聊记录。"
            else:
                detail = f"没有找到 {target_date.isoformat()} 的群聊记录。"
            yield event.plain_result(
                detail + "\n插件只能总结启用后记录到的消息；刚安装时请先等待群内产生新消息。"
            )
            return

        stats = compute_stats(messages, target_date)
        try:
            analysis = await self._analyze(event, messages, stats)
        except Exception as exc:
            logger.warning(f"[group_summary] model analysis failed, using statistics: {exc}")
            analysis = build_fallback_analysis(messages, stats, "模型分析暂不可用")

        try:
            image_path = render_report(stats, analysis)
            track_file = getattr(event, "track_temporary_local_file", None)
            if callable(track_file):
                track_file(str(image_path))
            yield event.image_result(str(image_path))
        except Exception as exc:
            logger.warning(f"[group_summary] report render failed, using text: {exc}")
            yield event.plain_result(format_text_report(stats, analysis))
