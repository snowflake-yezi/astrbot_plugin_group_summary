from __future__ import annotations

import asyncio
import datetime as dt

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ...core.config import GROUP_COMMAND_PATTERN
from ...core.dates import parse_target_date
from ...core.history import MessageHistoryStore
from ...core.statistics import compute_stats
from .domain import build_fallback_analysis, format_text_report
from .renderer import render_report
from .service import GroupSummaryService


class GroupSummaryHandler:
    def __init__(self, history: MessageHistoryStore, service: GroupSummaryService):
        self.history = history
        self.service = service

    async def handle(self, event: AstrMessageEvent):
        if not event.get_group_id():
            yield event.plain_result("群聊总结仅支持在群聊中使用。")
            return

        match = GROUP_COMMAND_PATTERN.fullmatch(event.get_message_str().strip())
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
            messages = await self.history.load(event, target_date)
        except Exception as exc:
            logger.warning(f"[group_summary] load messages failed: {exc}")
            yield event.plain_result(
                "群聊记录读取失败，请稍后重试或查看 AstrBot 日志。"
            )
            return

        if not messages:
            if target_date == dt.datetime.now().astimezone().date():
                detail = "今天还没有可总结的群聊记录。"
            else:
                detail = f"没有找到 {target_date.isoformat()} 的群聊记录。"
            yield event.plain_result(
                detail
                + "\n插件只能总结启用后记录到的消息；刚安装时请先等待群内产生新消息。"
            )
            return

        stats = compute_stats(messages, target_date)
        try:
            analysis = await self.service.analyze(event, messages, stats)
        except Exception as exc:
            logger.warning(
                f"[group_summary] model analysis failed, using statistics: {exc}"
            )
            analysis = build_fallback_analysis(messages, stats, "模型分析暂不可用")

        try:
            image_path = await asyncio.to_thread(render_report, stats, analysis)
            track_file = getattr(event, "track_temporary_local_file", None)
            if callable(track_file):
                track_file(str(image_path))
            yield event.image_result(str(image_path))
        except Exception as exc:
            logger.warning(f"[group_summary] report render failed, using text: {exc}")
            yield event.plain_result(format_text_report(stats, analysis))
