from __future__ import annotations

from contextlib import aclosing

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

from .core.analysis import AnalysisClient
from .core.history import MessageHistoryStore
from .features.group.handler import GroupSummaryHandler
from .features.group.service import GroupSummaryService
from .features.personal.handler import PersonalChatHandler
from .features.personal.history import PersonalIndexStore
from .features.personal.service import PersonalSummaryService


class GroupSummaryPlugin(Star):
    """组合群聊日报、个人聊天总览与时段详情。"""

    def __init__(self, context: Context):
        super().__init__(context)
        self.history = MessageHistoryStore(context.message_history_manager)
        self.analysis_client = AnalysisClient(context)
        self.group_handler = GroupSummaryHandler(
            self.history,
            GroupSummaryService(self.analysis_client),
        )
        self.personal_handler = PersonalChatHandler(
            self.history,
            PersonalSummaryService(self.analysis_client),
            PersonalIndexStore(context.message_history_manager),
        )

    # AstrBot 按装饰函数的模块路径绑定入口，装饰器必须保留在插件类上。
    @filter.event_message_type(filter.EventMessageType.ALL)
    @filter.regex(r".*")
    async def track_group_messages(self, event: AstrMessageEvent):
        await self.history.track(event)
        return
        yield

    @filter.regex(r"^/?群聊总结(?:\s+.+)?\s*$")
    async def group_summary(self, event: AstrMessageEvent):
        async with aclosing(self.group_handler.handle(event)) as results:
            async for result in results:
                yield result

    @filter.regex(r"^/?个人聊天(?:记录|详情)(?:\s.*)?$")
    async def personal_chat(self, event: AstrMessageEvent):
        async with aclosing(self.personal_handler.handle(event)) as results:
            async for result in results:
                yield result
