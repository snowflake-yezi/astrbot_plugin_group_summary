from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Callable, Iterable, TypeVar

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.star import Context

from .models import ChatMessage

Analysis = TypeVar("Analysis")


class AnalysisClient:
    ANALYSIS_CACHE_TTL = 600
    MAX_CACHED_ANALYSES = 64
    MODEL_TIMEOUT = 60

    def __init__(self, context: Context):
        self.context = context
        self._analysis_cache: OrderedDict[str, tuple[float, Any, Any]] = OrderedDict()

    async def _get_provider(self, event: AstrMessageEvent):
        get_async = getattr(self.context, "get_using_provider_async", None)
        if callable(get_async):
            return await get_async(umo=event.unified_msg_origin)
        get_sync = getattr(self.context, "get_using_provider", None)
        if callable(get_sync):
            return get_sync(umo=event.unified_msg_origin)
        return None

    def _analysis_key(
        self,
        event: AstrMessageEvent,
        provider,
        prompt: str,
        messages: Iterable[ChatMessage],
    ) -> str:
        get_model = getattr(provider, "get_model", None)
        model = get_model() if callable(get_model) else getattr(provider, "model", "")
        digest = hashlib.sha256(
            json.dumps(
                [
                    str(event.get_platform_id()),
                    str(event.get_group_id()),
                    str(event.get_sender_id()),
                    str(event.unified_msg_origin),
                    id(provider),
                    str(model),
                    prompt,
                ],
                ensure_ascii=False,
            ).encode("utf-8")
        )
        # 缓存签名覆盖未抽样记录，避免旧分析误命中。
        for message in messages:
            digest.update(
                json.dumps(
                    [
                        message.sender_id,
                        message.sender_name,
                        message.created_at,
                        message.message_id,
                        message.interactions,
                        message.text,
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        return digest.hexdigest()

    def _cached_analysis(self, key: str):
        now = time.monotonic()
        for expired in [
            key for key, entry in self._analysis_cache.items() if entry[0] <= now
        ]:
            del self._analysis_cache[expired]
        entry = self._analysis_cache.get(key)
        if entry is None:
            return None
        self._analysis_cache.move_to_end(key)
        logger.info("[group_summary] analysis cache hit")
        return entry[2]

    def _remember_analysis(self, key: str, provider, analysis) -> None:
        # 缓存同时持有模型实例，避免对象 ID 复用后误命中。
        self._analysis_cache[key] = (
            time.monotonic() + self.ANALYSIS_CACHE_TTL,
            provider,
            analysis,
        )
        self._analysis_cache.move_to_end(key)
        while len(self._analysis_cache) > self.MAX_CACHED_ANALYSES:
            self._analysis_cache.popitem(last=False)

    async def analyze(
        self,
        event: AstrMessageEvent,
        *,
        prompt: str,
        system_prompt: str,
        messages: Iterable[ChatMessage],
        parse: Callable[[str], Analysis],
    ) -> Analysis | None:
        provider = await self._get_provider(event)
        if provider is None:
            return None
        key = self._analysis_key(event, provider, prompt, messages)
        cached = self._cached_analysis(key)
        if cached is not None:
            return cached
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt, contexts=[], image_urls=[], system_prompt=system_prompt
            ),
            timeout=self.MODEL_TIMEOUT,
        )
        completion = str(getattr(response, "completion_text", "") or "").strip()
        if not completion:
            raise ValueError("聊天模型返回了空内容")
        analysis = parse(completion)
        self._remember_analysis(key, provider, analysis)
        return analysis
