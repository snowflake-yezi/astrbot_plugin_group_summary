from __future__ import annotations

from astrbot.api.event import AstrMessageEvent

from ...core.analysis import AnalysisClient
from ...core.models import ChatMessage, ReportStats
from .domain import (
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_fallback_analysis,
    merge_with_fallback,
    parse_model_analysis,
)
from .models import ReportAnalysis


class GroupSummaryService:
    def __init__(self, client: AnalysisClient):
        self.client = client

    async def analyze(
        self, event: AstrMessageEvent, messages: list[ChatMessage], stats: ReportStats
    ) -> ReportAnalysis:
        fallback = build_fallback_analysis(messages, stats)
        analysis = await self.client.analyze(
            event,
            prompt=build_analysis_prompt(messages, stats),
            system_prompt=SYSTEM_PROMPT,
            messages=messages,
            parse=lambda raw: merge_with_fallback(
                parse_model_analysis(raw, messages), fallback
            ),
        )
        return (
            analysis
            if analysis is not None
            else build_fallback_analysis(messages, stats, "未配置聊天模型")
        )
