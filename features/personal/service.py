from __future__ import annotations

import asyncio
from itertools import chain

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

from ...core.analysis import AnalysisClient
from .domain import (
    PERSONAL_SYSTEM_PROMPT,
    build_personal_fallback,
    build_personal_prompt,
    parse_personal_analysis,
)
from .models import PersonalAnalysis, PersonalReport


class PersonalSummaryService:
    def __init__(self, client: AnalysisClient):
        self.client = client

    async def analyze(
        self, event: AstrMessageEvent, report: PersonalReport
    ) -> PersonalAnalysis:
        try:
            analysis = await self.client.analyze(
                event,
                prompt=build_personal_prompt(report),
                system_prompt=PERSONAL_SYSTEM_PROMPT,
                messages=chain.from_iterable(
                    chain(period.messages, period.context) for period in report.periods
                ),
                parse=lambda raw: parse_personal_analysis(raw, report),
            )
            return (
                analysis
                if analysis is not None
                else build_personal_fallback(report, "未配置聊天模型")
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[group_summary] personal model analysis timed out, using statistics"
            )
            return build_personal_fallback(report, "模型分析超时")
        except Exception as exc:
            logger.warning(
                f"[group_summary] personal analysis failed, using statistics: {exc}"
            )
            return build_personal_fallback(report, "模型分析暂不可用")
