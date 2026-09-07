from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import re
import time
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.star import Context, Star

from .personal_report import (
    PERSONAL_SYSTEM_PROMPT,
    PersonalReport,
    build_detail_report,
    build_personal_fallback,
    build_personal_prompt,
    build_personal_report,
    format_personal_text,
    parse_personal_analysis,
    render_personal_report,
)
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
    PERSONAL_PATTERN = re.compile(r"^/?个人聊天(记录|详情)(?:\s+(.*))?\s*$")

    def __init__(self, context: Context):
        super().__init__(context)
        self.context = context
        self._personal_requests: set[str] = set()

    @staticmethod
    def _plain_command_text(event: AstrMessageEvent) -> str:
        parts = [str(component.text) for component in event.get_messages() if isinstance(component, Plain)]
        return " ".join(parts).strip() if parts else event.get_message_str().strip()

    @staticmethod
    def _personal_index_key(event: AstrMessageEvent, target_id: str) -> str:
        identity = [str(event.get_group_id()), str(event.get_sender_id()), target_id]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()
        return f"group_summary:personal_index:{digest}"

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
        if self.COMMAND_PATTERN.fullmatch(command_text) or self.PERSONAL_PATTERN.fullmatch(self._plain_command_text(event)):
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

    async def _load_messages(self, event: AstrMessageEvent, target_date: dt.date | None = None) -> list[ChatMessage]:
        start, end = local_day_bounds(target_date) if target_date is not None else (0, float("inf"))
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

    async def _analyze_personal(self, event: AstrMessageEvent, report: PersonalReport):
        try:
            provider = await self._get_provider(event)
            if provider is None:
                return build_personal_fallback(report, "未配置聊天模型")
            response = await provider.text_chat(
                prompt=build_personal_prompt(report), contexts=[], image_urls=[],
                system_prompt=PERSONAL_SYSTEM_PROMPT,
            )
            return parse_personal_analysis(str(getattr(response, "completion_text", "") or ""), report)
        except Exception as exc:
            logger.warning(f"[group_summary] personal analysis failed, using statistics: {exc}")
            return build_personal_fallback(report, "模型分析暂不可用")

    async def _save_personal_index(self, event: AstrMessageEvent, key: str, report: PersonalReport) -> None:
        kwargs = {
            "platform_id": event.get_platform_id(), "user_id": key,
            "sender_id": str(event.get_sender_id()), "sender_name": event.get_sender_name(),
            "max_messages": 1,
            "content": {
                "plugin": self.PLUGIN_MARKER, "kind": "personal_index", "schema_version": 1,
                "target_id": report.target_id,
                "periods": [
                    {"start": period.start, "end": period.end, "count": len(period.messages)}
                    for period in report.periods
                ],
            },
        }
        try:
            await self.context.message_history_manager.insert(**kwargs)
        except TypeError as exc:
            if "max_messages" not in str(exc):
                raise
            kwargs.pop("max_messages")
            await self.context.message_history_manager.insert(**kwargs)

    async def _load_personal_period(self, event: AstrMessageEvent, key: str, target_id: str, number: int):
        rows = await self.context.message_history_manager.get(
            platform_id=event.get_platform_id(), user_id=key, page=1, page_size=1,
        )
        if not rows:
            raise ValueError("还没有这位群友的时段索引，请先发送：个人聊天记录 @群友。")
        content = getattr(rows[0], "content", None)
        if not isinstance(content, dict) or content.get("kind") != "personal_index" or content.get("target_id") != target_id:
            raise ValueError("时段索引不可用，请重新发送：个人聊天记录 @群友。")
        periods = content.get("periods", [])
        if not isinstance(periods, list) or not 1 <= number <= len(periods):
            raise ValueError(f"时段编号应在 1 到 {len(periods) if isinstance(periods, list) else 0} 之间，请参考最近一次个人总览。")
        period = periods[number - 1]
        if not isinstance(period, dict) or any(type(period.get(field)) is not int for field in ("start", "end", "count")):
            raise ValueError("时段索引不可用，请重新生成个人总览。")
        return period

    @filter.regex(r"^/?个人聊天(?:记录|详情)(?:\s.*)?$")
    async def personal_chat(self, event: AstrMessageEvent):
        should_call_llm = getattr(event, "should_call_llm", None)
        if callable(should_call_llm):
            should_call_llm(False)
        if not event.get_group_id():
            yield event.plain_result("个人聊天记录和详情仅支持在群聊中使用。")
            return
        match = self.PERSONAL_PATTERN.fullmatch(self._plain_command_text(event))
        if match is None:
            yield event.plain_result("用法：个人聊天记录 @群友；个人聊天详情 @群友 时段编号 [页码]。")
            return
        targets = {
            str(component.qq).strip() for component in event.get_messages()
            if isinstance(component, At) and str(component.qq).strip() not in {"", "all", str(event.get_self_id())}
        }
        if len(targets) != 1:
            yield event.plain_result("请实际 @ 一位群友（不能是机器人或全体成员），例如：个人聊天记录 @群友。")
            return
        target_id = targets.pop()
        command, argument = match.group(1), (match.group(2) or "").strip()
        detailed = command == "详情"
        target_date = None
        number = page = 1
        if detailed:
            numbers = re.fullmatch(r"([1-9]\d{0,3})(?:\s+([1-9]\d{0,4}))?", argument)
            if numbers is None:
                yield event.plain_result("用法：个人聊天详情 @群友 时段编号 [页码]，例如：个人聊天详情 @群友 3。")
                return
            number, page = int(numbers.group(1)), int(numbers.group(2) or 1)
        elif argument and argument != "全部":
            try:
                target_date = parse_target_date(argument)
            except ValueError:
                yield event.plain_result("日期格式无法识别。示例：个人聊天记录 @群友 今天、个人聊天记录 @群友 2026-09-01。")
                return

        key = self._personal_index_key(event, target_id)
        request_key = f"{event.get_platform_id()}:{key}"
        if request_key in self._personal_requests:
            yield event.plain_result("你对这位群友的报告正在生成，请稍后再试。")
            return
        self._personal_requests.add(request_key)
        try:
            period = None
            try:
                if detailed:
                    period = await self._load_personal_period(event, key, target_id, number)
                messages = await self._load_messages(event)
            except ValueError as exc:
                yield event.plain_result(str(exc))
                return
            except Exception as exc:
                logger.warning(f"[group_summary] personal history read failed: {exc}")
                yield event.plain_result("个人聊天记录读取失败，请稍后重试或查看 AstrBot 日志。")
                return
            target_messages = [message for message in messages if message.sender_id == target_id]
            if period is not None:
                target_messages = [message for message in target_messages if period["start"] <= message.created_at <= period["end"]]
                if len(target_messages) != period["count"]:
                    yield event.plain_result("该时段记录已变化或部分已过期，请重新生成个人聊天记录，再按新的时段编号查看详情。")
                    return
            elif target_date is not None:
                start, end = local_day_bounds(target_date)
                target_messages = [message for message in target_messages if start <= message.created_at < end]
            if not target_messages:
                scope = target_date.isoformat() if target_date else "当前保留范围内"
                yield event.plain_result(f"没有找到这位群友在{scope}的发言。只能查询本群在插件启用后记录且尚未过期的消息。")
                return
            try:
                if detailed:
                    report = build_detail_report(messages, target_messages, number, page)
                else:
                    report = build_personal_report(messages, target_messages, target_date.isoformat() if target_date else "当前保留记录")
            except ValueError as exc:
                yield event.plain_result(str(exc))
                return
            analysis = await self._analyze_personal(event, report)
            try:
                image_path = await asyncio.to_thread(render_personal_report, report, analysis)
                track_file = getattr(event, "track_temporary_local_file", None)
                if callable(track_file):
                    track_file(str(image_path))
                results = [event.image_result(str(image_path))]
            except Exception as exc:
                logger.warning(f"[group_summary] personal render failed, using text: {exc}")
                text_report = format_personal_text(report, analysis)
                results = [event.plain_result(text_report[offset:offset + 3000]) for offset in range(0, len(text_report), 3000)]
            index_saved = True
            if not detailed:
                try:
                    await self._save_personal_index(event, key, report)
                except Exception as exc:
                    logger.warning(f"[group_summary] personal index save failed: {exc}")
                    index_saved = False
            for result in results:
                yield result
            if not index_saved:
                yield event.plain_result("本次时段索引保存失败，暂不能按本图编号查询详情，请稍后重新生成总览。")
            elif not detailed:
                yield event.plain_result(f"查看详情：个人聊天详情 @群友 时段编号（1-{len(report.periods)}）。编号对应你最近一次对该群友生成的总览。")
            elif report.page_count > 1:
                yield event.plain_result(
                    f"原始记录第 {report.page}/{report.page_count} 页；指定页码：个人聊天详情 @群友 {number} 页码。"
                )
        finally:
            self._personal_requests.discard(request_key)

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
