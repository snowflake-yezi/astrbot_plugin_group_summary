from __future__ import annotations

import asyncio
import re
import time

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ...core.config import PERSONAL_COMMAND_PATTERN
from ...core.dates import local_day_bounds, parse_target_date
from ...core.event import mentioned_members, plain_command_text
from ...core.history import MessageHistoryStore
from .domain import (
    CONTEXT_WINDOW,
    build_detail_report,
    build_personal_report,
    format_personal_text,
)
from .history import PersonalIndexStore
from .renderer import render_personal_report
from .service import PersonalSummaryService


class PersonalChatHandler:
    def __init__(
        self,
        history: MessageHistoryStore,
        service: PersonalSummaryService,
        index: PersonalIndexStore,
    ):
        self.history = history
        self.service = service
        self.index = index
        self._requests: set[str] = set()

    async def handle(self, event: AstrMessageEvent):
        should_call_llm = getattr(event, "should_call_llm", None)
        if callable(should_call_llm):
            should_call_llm(False)
        if not event.get_group_id():
            yield event.plain_result("个人聊天记录和详情仅支持在群聊中使用。")
            return
        match = PERSONAL_COMMAND_PATTERN.fullmatch(plain_command_text(event))
        if match is None:
            yield event.plain_result(
                "用法：个人聊天记录 @群友；个人聊天详情 @群友 时段编号 [页码]。"
            )
            return
        targets = mentioned_members(event)
        if len(targets) != 1:
            yield event.plain_result(
                "请实际 @ 一位群友（不能是机器人或全体成员），例如：个人聊天记录 @群友。"
            )
            return
        target_id = targets.pop()
        command, argument = match.group(1), (match.group(2) or "").strip()
        detailed = command == "详情"
        target_date = None
        number = page = 1
        if detailed:
            numbers = re.fullmatch(r"([1-9]\d{0,3})(?:\s+([1-9]\d{0,4}))?", argument)
            if numbers is None:
                yield event.plain_result(
                    "用法：个人聊天详情 @群友 时段编号 [页码]，例如：个人聊天详情 @群友 3。"
                )
                return
            number, page = int(numbers.group(1)), int(numbers.group(2) or 1)
        elif argument and argument != "全部":
            try:
                target_date = parse_target_date(argument)
            except ValueError:
                yield event.plain_result(
                    "日期格式无法识别。示例：个人聊天记录 @群友 今天、个人聊天记录 @群友 2026-09-01。"
                )
                return

        key = self.index.key(event, target_id)
        request_key = f"{event.get_platform_id()}:{key}"
        if request_key in self._requests:
            yield event.plain_result("你对这位群友的报告正在生成，请稍后再试。")
            return
        self._requests.add(request_key)
        try:
            started = time.perf_counter()
            period = None
            try:
                bounds = None
                if detailed:
                    period = await self.index.load_period(event, key, target_id, number)
                    bounds = (
                        period["start"] - CONTEXT_WINDOW,
                        period["end"] + CONTEXT_WINDOW + 1,
                    )
                elif target_date is not None:
                    start, end = local_day_bounds(target_date)
                    bounds = (start - CONTEXT_WINDOW, end + CONTEXT_WINDOW)
                messages = await self.history.load(event, bounds=bounds)
            except ValueError as exc:
                yield event.plain_result(str(exc))
                return
            except Exception as exc:
                logger.warning(f"[group_summary] personal history read failed: {exc}")
                yield event.plain_result(
                    "个人聊天记录读取失败，请稍后重试或查看 AstrBot 日志。"
                )
                return
            loaded_at = time.perf_counter()
            target_messages = [
                message for message in messages if message.sender_id == target_id
            ]
            if period is not None:
                target_messages = [
                    message
                    for message in target_messages
                    if period["start"] <= message.created_at <= period["end"]
                ]
                if len(target_messages) != period["count"]:
                    yield event.plain_result(
                        "该时段记录已变化或部分已过期，请重新生成个人聊天记录，再按新的时段编号查看详情。"
                    )
                    return
            elif target_date is not None:
                start, end = local_day_bounds(target_date)
                target_messages = [
                    message
                    for message in target_messages
                    if start <= message.created_at < end
                ]
            if not target_messages:
                scope = target_date.isoformat() if target_date else "当前保留范围内"
                yield event.plain_result(
                    f"没有找到这位群友在{scope}的发言。只能查询本群在插件启用后记录且尚未过期的消息。"
                )
                return
            try:
                if detailed:
                    report = build_detail_report(
                        messages, target_messages, number, page
                    )
                else:
                    report = build_personal_report(
                        messages,
                        target_messages,
                        target_date.isoformat() if target_date else "当前保留记录",
                    )
            except ValueError as exc:
                yield event.plain_result(str(exc))
                return
            built_at = time.perf_counter()
            analysis = await self.service.analyze(event, report)
            analyzed_at = time.perf_counter()
            try:
                image_path = await asyncio.to_thread(
                    render_personal_report, report, analysis
                )
                track_file = getattr(event, "track_temporary_local_file", None)
                if callable(track_file):
                    track_file(str(image_path))
                results = [event.image_result(str(image_path))]
            except Exception as exc:
                logger.warning(
                    f"[group_summary] personal render failed, using text: {exc}"
                )
                text_report = format_personal_text(report, analysis)
                results = [
                    event.plain_result(text_report[offset : offset + 3000])
                    for offset in range(0, len(text_report), 3000)
                ]
            logger.info(
                f"[group_summary] personal timing: records={len(messages)} "
                f"load={loaded_at - started:.3f}s build={built_at - loaded_at:.3f}s "
                f"analysis={analyzed_at - built_at:.3f}s render={time.perf_counter() - analyzed_at:.3f}s"
            )
            index_saved = True
            if not detailed:
                try:
                    await self.index.save(event, key, report)
                except Exception as exc:
                    logger.warning(f"[group_summary] personal index save failed: {exc}")
                    index_saved = False
            for result in results:
                yield result
            if not index_saved:
                yield event.plain_result(
                    "本次时段索引保存失败，暂不能按本图编号查询详情，请稍后重新生成总览。"
                )
            elif not detailed:
                yield event.plain_result(
                    f"查看详情：个人聊天详情 @群友 时段编号（1-{len(report.periods)}）。编号对应你最近一次对该群友生成的总览。"
                )
            elif report.page_count > 1:
                yield event.plain_result(
                    f"原始记录第 {report.page}/{report.page_count} 页；指定页码：个人聊天详情 @群友 {number} 页码。"
                )
        finally:
            self._requests.discard(request_key)
