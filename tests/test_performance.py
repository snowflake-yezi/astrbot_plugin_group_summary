from __future__ import annotations

import asyncio
import datetime as dt
import json
import types
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, patch

from test_personal import MentionEvent, message, rows_for
from test_plugin import FakeContext, GroupSummaryPlugin
from astrbot_plugin_group_summary.features.personal.domain import (
    build_detail_report,
    build_personal_prompt,
    build_personal_report,
)
from astrbot_plugin_group_summary.core.rendering import BaseReportRenderer
from astrbot_plugin_group_summary.core.statistics import compute_stats
from astrbot_plugin_group_summary.core import history as history_module


class AnalysisCacheTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.context = FakeContext()
        self.plugin = GroupSummaryPlugin(self.context)
        self.event = MentionEvent()
        self.messages = [
            message("2", index, f"发言 {index}", message_id=str(index))
            for index in range(200)
        ]
        self.report = build_personal_report(self.messages, self.messages, "全部")
        self.response = types.SimpleNamespace(
            completion_text=json.dumps(
                {
                    "overview": "发言概览",
                    "periods": [
                        {
                            "id": 1,
                            "title": "话题",
                            "summary": "发言分析",
                            "context": "讨论背景",
                        }
                    ],
                }
            )
        )
        self.provider = types.SimpleNamespace(
            model="model-a", text_chat=AsyncMock(return_value=self.response)
        )
        self.context.get_using_provider_async = AsyncMock(return_value=self.provider)

    async def test_detail_pages_share_analysis_but_keep_distinct_transcripts(self):
        first = build_detail_report(self.messages, self.messages, 1, 1)
        second = build_detail_report(self.messages, self.messages, 1, 2)
        analysis = await self.plugin.personal_handler.service.analyze(self.event, first)
        cached = await self.plugin.personal_handler.service.analyze(self.event, second)
        self.assertIs(analysis, cached)
        self.provider.text_chat.assert_awaited_once()
        self.assertNotEqual(first.transcript, second.transcript)

    async def test_unsampled_message_change_invalidates_analysis(self):
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        changed = self.messages.copy()
        changed[1] = replace(changed[1], text="这条未抽样的消息已变化")
        changed_report = build_personal_report(changed, changed, "全部")
        self.assertEqual(
            build_personal_prompt(changed_report), build_personal_prompt(self.report)
        )
        await self.plugin.personal_handler.service.analyze(self.event, changed_report)
        self.assertEqual(self.provider.text_chat.await_count, 2)

    async def test_new_context_invalidates_analysis(self):
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        nearby = message("3", 1, "新的群友回应")
        changed_report = build_personal_report(
            self.messages + [nearby], self.messages, "全部"
        )
        await self.plugin.personal_handler.service.analyze(self.event, changed_report)
        self.assertEqual(self.provider.text_chat.await_count, 2)

    async def test_cache_is_scoped_to_platform_group_requester_and_conversation(self):
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        for event in (MentionEvent(group_id="200"), MentionEvent(sender_id="8")):
            await self.plugin.personal_handler.service.analyze(event, self.report)
        platform = MentionEvent()
        platform.get_platform_id = lambda: "other-platform"
        await self.plugin.personal_handler.service.analyze(platform, self.report)
        conversation = MentionEvent()
        conversation.unified_msg_origin = "fake:other-conversation"
        await self.plugin.personal_handler.service.analyze(conversation, self.report)
        self.assertEqual(self.provider.text_chat.await_count, 5)

    async def test_switching_model_or_provider_invalidates_analysis(self):
        self.provider.get_model = lambda: self.provider.model
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        self.provider.model = "model-b"
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        self.assertEqual(self.provider.text_chat.await_count, 2)
        replacement = types.SimpleNamespace(
            model="model-b", text_chat=AsyncMock(return_value=self.response)
        )
        self.context.get_using_provider_async.return_value = replacement
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        replacement.text_chat.assert_awaited_once()

    async def test_cache_expires_and_evicts_least_recently_used_entries(self):
        self.plugin.analysis_client.ANALYSIS_CACHE_TTL = -1
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        self.plugin.analysis_client.ANALYSIS_CACHE_TTL = 600
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        self.assertEqual(self.provider.text_chat.await_count, 2)
        self.plugin.analysis_client.MAX_CACHED_ANALYSES = 2
        other = MentionEvent(sender_id="8")
        await self.plugin.personal_handler.service.analyze(other, self.report)
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        await self.plugin.personal_handler.service.analyze(
            MentionEvent(sender_id="9"), self.report
        )
        self.assertEqual(len(self.plugin.analysis_client._analysis_cache), 2)
        await self.plugin.personal_handler.service.analyze(self.event, self.report)
        self.assertEqual(self.provider.text_chat.await_count, 4)
        await self.plugin.personal_handler.service.analyze(other, self.report)
        self.assertEqual(self.provider.text_chat.await_count, 5)

    async def test_model_timeout_cancels_request_and_fallback_is_not_cached(self):
        cancelled = asyncio.Event()

        async def never_finishes(**kwargs):
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        self.plugin.analysis_client.MODEL_TIMEOUT = 0.01
        self.provider.text_chat.side_effect = never_finishes
        fallback = await self.plugin.personal_handler.service.analyze(
            self.event, self.report
        )
        self.assertEqual(fallback.source, "statistics")
        self.assertIn("超时", fallback.overview)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.plugin.analysis_client._analysis_cache)
        self.provider.text_chat.side_effect = None
        recovered = await self.plugin.personal_handler.service.analyze(
            self.event, self.report
        )
        self.assertEqual(recovered.source, "model")
        self.assertEqual(self.provider.text_chat.await_count, 2)

    async def test_invalid_model_response_is_not_cached(self):
        self.provider.text_chat.return_value = types.SimpleNamespace(
            completion_text="无效响应"
        )
        fallback = await self.plugin.personal_handler.service.analyze(
            self.event, self.report
        )
        self.assertEqual(fallback.source, "statistics")
        self.assertFalse(self.plugin.analysis_client._analysis_cache)
        self.provider.text_chat.return_value = self.response
        recovered = await self.plugin.personal_handler.service.analyze(
            self.event, self.report
        )
        self.assertEqual(recovered.source, "model")
        self.assertEqual(self.provider.text_chat.await_count, 2)

    async def test_group_summary_reuses_successful_analysis_and_limits_model_wait(self):
        self.provider.text_chat.return_value = types.SimpleNamespace(
            completion_text=json.dumps(
                {
                    "title": "聊天概览",
                    "topics": [{"title": "话题", "summary": "讨论话题"}],
                }
            )
        )
        stats = compute_stats(self.messages, dt.date(2026, 9, 1))
        first = await self.plugin.group_handler.service.analyze(
            self.event, self.messages, stats
        )
        second = await self.plugin.group_handler.service.analyze(
            self.event, self.messages, stats
        )
        self.assertIs(first, second)
        self.provider.text_chat.assert_awaited_once()
        self.plugin.analysis_client._analysis_cache.clear()
        self.plugin.analysis_client.MODEL_TIMEOUT = 0.01

        async def never_finishes(**kwargs):
            await asyncio.Future()

        self.provider.text_chat.side_effect = never_finishes
        with self.assertRaises(asyncio.TimeoutError):
            await self.plugin.group_handler.service.analyze(
                self.event, self.messages, stats
            )
        self.assertFalse(self.plugin.analysis_client._analysis_cache)


class HistoryBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_twenty_thousand_recent_records_use_twenty_queries(self):
        messages = [
            message("2", index, f"消息 {index}", message_id=str(index))
            for index in range(20_001)
        ]
        context = FakeContext(rows_for(messages))
        original_get = context.message_history_manager.get

        async def chronological_page(**kwargs):
            return list(reversed(await original_get(**kwargs)))

        context.message_history_manager.get = AsyncMock(side_effect=chronological_page)
        loaded = await GroupSummaryPlugin(context).history.load(MentionEvent())
        self.assertEqual(loaded, messages[1:])
        self.assertEqual(context.message_history_manager.get.await_count, 20)

    async def test_time_bounds_skip_text_processing_and_keep_exact_edges(self):
        messages = [
            message("2", index, f"消息 {index}", message_id=str(index))
            for index in range(100)
        ]
        plugin = GroupSummaryPlugin(FakeContext(rows_for(messages)))
        with patch.object(
            history_module, "normalize_text", wraps=history_module.normalize_text
        ) as normalize:
            loaded = await plugin.history.load(
                MentionEvent(),
                bounds=(messages[40].created_at, messages[45].created_at),
            )
        self.assertEqual(loaded, messages[40:45])
        processed = [call.args[0] for call in normalize.call_args_list]
        self.assertNotIn("消息 0", processed)
        self.assertNotIn("消息 45", processed)
        self.assertIn("消息 40", processed)


class TextLayoutTests(unittest.TestCase):
    def setUp(self):
        self.renderer = BaseReportRenderer()

    def test_wrapping_preserves_content_and_respects_font_width(self):
        samples = (
            "中文排版测试 English AVWA 09:30 123 " * 5,
            "cafe\u0301 e\u0301 hello 世界 " * 4,
        )
        for text in samples:
            for width in (60, 130, 1088):
                with self.subTest(width=width, text=text[:20]):
                    lines = self.renderer._wrap(text, self.renderer.font_body, width)
                    self.assertEqual("".join(lines), text)
                    self.assertTrue(
                        all(
                            self.renderer.draw.textlength(
                                line, font=self.renderer.font_body
                            )
                            <= width
                            for line in lines
                        )
                    )
                    for previous, following in zip(lines, lines[1:]):
                        self.assertGreater(
                            self.renderer.draw.textlength(
                                previous + following[0], font=self.renderer.font_body
                            ),
                            width,
                        )

    def test_newlines_and_truncation_keep_layout_bounded(self):
        self.assertEqual(
            self.renderer._wrap("一\n\n二\n", self.renderer.font_body, 200),
            ["一", "", "二", ""],
        )
        lines = self.renderer._wrap(
            "较长的中文句子" * 20, self.renderer.font_body, 130, max_lines=2
        )
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[-1].endswith("..."))
        self.assertTrue(
            all(
                self.renderer.draw.textlength(line, font=self.renderer.font_body) <= 130
                for line in lines
            )
        )

    def test_long_text_uses_fewer_font_measurements(self):
        text = "需要快速排版的长消息 English words 09:30。" * 50
        with patch.object(
            self.renderer.draw, "textlength", wraps=self.renderer.draw.textlength
        ) as measure:
            lines = self.renderer._wrap(text, self.renderer.font_body, 1088)
        self.assertEqual("".join(lines), text)
        self.assertLess(measure.call_count, len(text) // 3)


if __name__ == "__main__":
    unittest.main()
