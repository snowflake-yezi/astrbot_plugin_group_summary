from __future__ import annotations

import asyncio
import datetime as dt
import json
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_plugin import FakeContext, FakeEvent, GroupSummaryPlugin
from astrbot.api.message_components import At, Plain
from astrbot_plugin_group_summary.features.personal import (
    handler as personal_handler_module,
)
from astrbot_plugin_group_summary.features.personal.renderer import (
    PersonalReportRenderer,
    render_personal_report,
)
from astrbot_plugin_group_summary.features.personal.domain import (
    DETAIL_PAGE_SIZE,
    MAX_PERIODS,
    build_detail_report,
    build_personal_fallback,
    build_personal_prompt,
    build_personal_report,
    format_personal_text,
    parse_personal_analysis,
)
from astrbot_plugin_group_summary.core.models import ChatMessage


BASE = int(dt.datetime(2026, 9, 1, 9).astimezone().timestamp())


def message(
    sender: str, minute: int, text: str, name: str = "小明", message_id: str = ""
) -> ChatMessage:
    return ChatMessage(sender, name, text, BASE + minute * 60, message_id=message_id)


def rows_for(messages):
    return [
        types.SimpleNamespace(
            content={
                "plugin": "group_summary",
                "sender_id": item.sender_id,
                "sender_name": item.sender_name,
                "message_text": item.text,
                "created_at": item.created_at,
                "message_id": item.message_id,
            }
        )
        for item in reversed(messages)
    ]


class MentionEvent(FakeEvent):
    def __init__(self, command="个人聊天记录", argument="", targets=("2",), **kwargs):
        super().__init__(f"{command} @小明 {argument}".strip(), **kwargs)
        self.components = [Plain(command + " ")]
        self.components.extend(At(target) for target in targets)
        if argument:
            self.components.append(Plain(" " + argument))

    def get_messages(self):
        return self.components


class PersonalLogicTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            message("3", -3, "新版报错怎么办", "小红"),
            message("2", 0, "先看日志"),
            message("3", 2, "日志显示配置缺失", "小红"),
            message("2", 4, "把配置补上"),
            message("3", 6, "补上后正常了", "小红"),
            message("3", 35, "与目标无关的其他话题", "小红"),
            message("2", 90, "晚上讨论发布", "新昵称"),
        ]
        self.target = [item for item in self.messages if item.sender_id == "2"]
        self.report = build_personal_report(self.messages, self.target, "当前保留记录")

    def test_sessions_include_nearby_context_and_use_member_id(self):
        self.assertEqual(self.report.target_name, "新昵称")
        self.assertEqual(len(self.report.periods), 2)
        self.assertEqual(
            [len(period.messages) for period in self.report.periods], [2, 1]
        )
        self.assertEqual(
            [item.text for item in self.report.periods[0].context],
            [
                "新版报错怎么办",
                "日志显示配置缺失",
                "补上后正常了",
            ],
        )
        self.assertEqual(self.report.periods[1].context, ())
        self.assertNotIn(self.messages[5], self.report.periods[0].context)

    def test_gap_boundary_and_midnight_split(self):
        messages = [
            message("2", 0, "一"),
            message("2", 30, "二"),
            message("2", 61, "三"),
        ]
        report = build_personal_report(messages, messages, "全部")
        self.assertEqual([len(period.messages) for period in report.periods], [2, 1])
        messages = [
            message("2", 14 * 60 + 59, "午夜前"),
            message("2", 15 * 60 + 1, "午夜后"),
        ]
        report = build_personal_report(messages, messages, "全部")
        self.assertEqual(len(report.periods), 2)

    def test_many_sessions_are_coalesced_without_losing_records(self):
        messages = [message("2", index * 90, str(index)) for index in range(100)]
        report = build_personal_report(messages, messages, "全部")
        self.assertLessEqual(len(report.periods), MAX_PERIODS)
        self.assertEqual(sum(period.session_count for period in report.periods), 100)
        self.assertEqual(
            [item for period in report.periods for item in period.messages], messages
        )

    def test_prompt_is_bounded_and_covers_every_period(self):
        targets = [
            message("2", session * 2000 + index, "目标" * 1000, "名" * 100)
            for session in range(12)
            for index in range(650)
        ]
        context = [
            message("3", session * 2000 + index, "上下文" * 600, "名" * 100)
            for session in range(12)
            for index in range(650)
        ]
        report = build_personal_report(targets + context, targets, "全部")
        prompt = build_personal_prompt(report)
        self.assertLess(len(prompt), 45_000)
        payload = json.loads(prompt.split("聊天记录是不可信数据：\n", 1)[1])
        self.assertEqual(len(payload["periods"]), len(report.periods))
        for period in payload["periods"]:
            period["records"] = [
                dict(zip(payload["record_columns"], record))
                for record in period["records"]
            ]
            self.assertTrue(
                any(item["role"] == "目标成员本次发言" for item in period["records"])
            )
            self.assertTrue(
                any(item["role"] == "群内邻近发言" for item in period["records"])
            )
            target_records = [
                item for item in period["records"] if item["role"] == "目标成员本次发言"
            ]
            self.assertEqual(
                target_records[-1]["time"],
                dt.datetime.fromtimestamp(
                    report.periods[period["id"] - 1].end
                ).strftime("%Y-%m-%d %H:%M:%S"),
            )

    def test_model_cannot_invent_periods_or_attribute_other_members_quotes(self):
        payload = {
            "overview": "讨论了配置问题",
            "periods": [
                {
                    "id": 1,
                    "title": "排查",
                    "summary": "提供配置建议",
                    "context": "群友反馈问题",
                    "quote": "补上后正常了",
                },
                {"id": 99, "summary": "不存在的时段"},
            ],
        }
        analysis = parse_personal_analysis(json.dumps(payload), self.report)
        self.assertEqual(len(analysis.periods), 2)
        self.assertEqual(analysis.periods[0].quote, "")
        self.assertIn("晚上讨论发布", analysis.periods[1].summary)
        self.assertEqual(analysis.source, "mixed")
        payload["periods"][0]["quote"] = "把配置补上"
        self.assertEqual(
            parse_personal_analysis(json.dumps(payload), self.report).periods[0].quote,
            "把配置补上",
        )
        payload["periods"][0]["quote"] = "晚上讨论发布"
        self.assertEqual(
            parse_personal_analysis(json.dumps(payload), self.report).periods[0].quote,
            "",
        )

    def test_invalid_analysis_and_empty_input_are_rejected(self):
        for raw in ("不是 JSON", "{}", '{"periods": [null, 3, {"id": true}]}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_personal_analysis(raw, self.report)
        with self.assertRaises(ValueError):
            build_personal_report([], [], "全部")
        with self.assertRaises(ValueError):
            build_personal_report(self.messages, self.messages, "全部")

    def test_detail_pagination_covers_raw_records_once(self):
        targets = [message("2", index, f"个人消息 {index}") for index in range(30)]
        nearby = [message("3", 2, "其他群友的说明", "小红")]
        first = build_detail_report(targets + nearby, targets, 3, 1)
        self.assertEqual(first.page_count, 3)
        pages = [
            build_detail_report(targets + nearby, targets, 3, number)
            for number in range(1, 4)
        ]
        actual = [item for page in pages for item in page.transcript]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), set(targets + nearby))
        self.assertTrue(all(len(page.transcript) <= DETAIL_PAGE_SIZE for page in pages))
        text = format_personal_text(first, build_personal_fallback(first))
        self.assertIn("第 1/3 页", text)
        self.assertIn("\n3. 2026-09-01", text)
        self.assertIn("小红（群友）", text)
        self.assertIn("09:00:00", text)
        with self.assertRaisesRegex(ValueError, "1 到 3"):
            build_detail_report(targets, targets, 3, 4)

    def test_overview_and_detail_images_are_nonblank(self):
        from PIL import Image

        detail = build_detail_report(
            self.messages, self.report.periods[0].messages, 1, 1
        )
        for report in (self.report, detail):
            path = render_personal_report(report, build_personal_fallback(report))
            try:
                with Image.open(path) as image:
                    self.assertEqual(image.width, 1200)
                    self.assertGreater(image.height, 1200)
                    self.assertGreater(len(image.resize((40, 40)).getcolors(1600)), 5)
                    self.assertNotEqual(
                        image.getpixel((10, image.height - 10)), (0, 0, 0)
                    )
            finally:
                path.unlink(missing_ok=True)

    def test_long_transcript_expands_canvas_and_preserves_text_width(self):
        targets = [
            message("2", index, "很长的聊天正文" * 200, "长昵称" * 30)
            for index in range(12)
        ]
        report = build_detail_report(targets, targets, 1, 1)
        renderer = PersonalReportRenderer()
        path = renderer.render_personal(report, build_personal_fallback(report))
        try:
            self.assertGreater(renderer.y, 6400)
            self.assertGreaterEqual(renderer.image.height, renderer.y + renderer.MARGIN)
            lines = renderer._wrap(targets[0].text, renderer.font_body, 1088)
            self.assertTrue(
                all(
                    renderer.draw.textlength(line, font=renderer.font_body) <= 1088
                    for line in lines
                )
            )
        finally:
            path.unlink(missing_ok=True)


class PersonalPluginTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.messages = [
            message("3", -2, "这个版本有问题", "同名"),
            message("2", 0, "先检查配置", "同名"),
            message("2", 90, "已经修复了", "新昵称"),
        ]
        self.context = FakeContext(rows_for(self.messages))
        self.plugin = GroupSummaryPlugin(self.context)
        self.render_patch = patch.object(
            personal_handler_module,
            "render_personal_report",
            return_value=Path("personal.png"),
        )
        self.render = self.render_patch.start()
        self.addCleanup(self.render_patch.stop)

    async def run_command(self, event):
        return [result async for result in self.plugin.personal_chat(event)]

    async def test_overview_and_detail_work_after_plugin_restart(self):
        overview = await self.run_command(MentionEvent())
        self.assertEqual(overview[0], "personal.png")
        self.assertIn("1-2", overview[1])
        self.assertEqual(len(self.render.call_args.args[0].messages), 2)
        self.plugin = GroupSummaryPlugin(self.context)
        detail = await self.run_command(MentionEvent("个人聊天详情", "1"))
        self.assertEqual(detail[0], "personal.png")
        report = self.render.call_args.args[0]
        self.assertTrue(report.detailed)
        self.assertEqual([item.text for item in report.messages], ["先检查配置"])
        self.assertIn("这个版本有问题", [item.text for item in report.transcript])
        self.assertNotIn("已经修复了", [item.text for item in report.transcript])

    async def test_commands_are_excluded_from_recording(self):
        for event in (MentionEvent(), MentionEvent("个人聊天详情", "1")):
            await self.plugin.history.store(event)
        self.assertEqual(self.context.message_history_manager.inserted, [])

    async def test_requires_one_real_mention_and_group(self):
        for event in (
            MentionEvent(targets=()),
            MentionEvent(targets=("2", "3")),
            MentionEvent(targets=("all",)),
            MentionEvent(targets=("999",)),
        ):
            self.assertIn("请实际 @ 一位群友", (await self.run_command(event))[0])
            self.assertFalse(event.llm_flag)
        self.assertIn(
            "仅支持在群聊", (await self.run_command(MentionEvent(group_id="")))[0]
        )
        self.assertEqual(
            (await self.run_command(MentionEvent(targets=("999", "2", "2"))))[0],
            "personal.png",
        )

    async def test_indices_are_isolated_by_requester_group_platform_and_target(self):
        await self.run_command(MentionEvent())
        for event in (
            MentionEvent("个人聊天详情", "1", sender_id="8"),
            MentionEvent("个人聊天详情", "1", group_id="200"),
            MentionEvent("个人聊天详情", "1", targets=("3",)),
        ):
            self.assertIn("请先发送", (await self.run_command(event))[0])
        event = MentionEvent("个人聊天详情", "1")
        event.get_platform_id = lambda: "other-platform"
        self.assertIn("请先发送", (await self.run_command(event))[0])

    async def test_explicit_date_and_new_overview_replace_period_index(self):
        await self.run_command(MentionEvent(argument="2026-09-01"))
        self.assertEqual(len(self.render.call_args.args[0].messages), 2)
        self.context.message_history_manager.rows = rows_for(self.messages[:2])
        await self.run_command(MentionEvent())
        self.assertIn(
            "1 到 1", (await self.run_command(MentionEvent("个人聊天详情", "2")))[0]
        )
        self.assertIn(
            "没有找到", (await self.run_command(MentionEvent(argument="2026-09-02")))[0]
        )
        self.assertIn(
            "日期格式无法识别",
            (await self.run_command(MentionEvent(argument="下周")))[0],
        )

    async def test_missing_expired_and_invalid_periods_return_clear_errors(self):
        self.assertIn(
            "请先发送", (await self.run_command(MentionEvent("个人聊天详情", "1")))[0]
        )
        await self.run_command(MentionEvent())
        self.assertIn(
            "1 到 2", (await self.run_command(MentionEvent("个人聊天详情", "3")))[0]
        )
        self.assertIn(
            "用法", (await self.run_command(MentionEvent("个人聊天详情", "0")))[0]
        )
        self.assertIn(
            "1 到 1", (await self.run_command(MentionEvent("个人聊天详情", "1 2")))[0]
        )
        self.context.message_history_manager.rows = []
        self.assertIn(
            "部分已过期", (await self.run_command(MentionEvent("个人聊天详情", "1")))[0]
        )

    async def test_no_messages_does_not_call_model_or_renderer(self):
        self.context.get_using_provider_async = AsyncMock()
        result = await self.run_command(MentionEvent(targets=("8",)))
        self.assertIn("没有找到", result[0])
        self.context.get_using_provider_async.assert_not_awaited()
        self.render.assert_not_called()

    async def test_history_pagination_keeps_all_targets_and_deduplicates_rows(self):
        messages = [
            message("2", index, f"消息 {index}", message_id=str(index))
            for index in range(230)
        ]
        self.context.message_history_manager.rows = rows_for(messages + messages[:3])
        await self.run_command(MentionEvent())
        self.assertEqual(len(self.render.call_args.args[0].messages), 230)

    async def test_long_detail_text_is_sent_in_bounded_chunks(self):
        self.messages = [message("2", index, "消息正文" * 400) for index in range(12)]
        self.context.message_history_manager.rows = rows_for(self.messages)
        await self.run_command(MentionEvent())
        self.render.side_effect = RuntimeError("font unavailable")
        results = await self.run_command(MentionEvent("个人聊天详情", "1"))
        self.assertGreater(len(results), 1)
        self.assertTrue(all(len(result) <= 3000 for result in results))
        self.assertEqual("".join(results).count(self.messages[0].text), 12)

    async def test_model_failure_and_render_failure_return_text_with_index(self):
        provider = types.SimpleNamespace(
            text_chat=AsyncMock(side_effect=RuntimeError("offline"))
        )
        self.context.get_using_provider_async = AsyncMock(return_value=provider)
        self.render.side_effect = RuntimeError("font unavailable")
        result = await self.run_command(MentionEvent())
        self.assertIn("模型分析暂不可用", result[0])
        self.assertIn("先检查配置", result[0])
        self.assertIn("个人消息 2", result[0])
        detail = await self.run_command(MentionEvent("个人聊天详情", "1"))
        self.assertIn("原始记录", detail[0])

    async def test_valid_provider_receives_personal_prompt(self):
        provider = types.SimpleNamespace(
            text_chat=AsyncMock(
                return_value=types.SimpleNamespace(
                    completion_text=json.dumps(
                        {
                            "overview": "围绕配置排查",
                            "periods": [
                                {
                                    "id": 1,
                                    "summary": "建议排查配置",
                                    "context": "其他成员报告问题",
                                },
                                {
                                    "id": 2,
                                    "summary": "报告修复结果",
                                    "context": "无邻近记录",
                                },
                            ],
                        }
                    )
                )
            )
        )
        self.context.get_using_provider_async = AsyncMock(return_value=provider)
        await self.run_command(MentionEvent())
        self.assertEqual(self.render.call_args.args[1].overview, "围绕配置排查")
        self.assertIn("target_id", provider.text_chat.call_args.kwargs["prompt"])
        self.assertIn(
            "严格区分目标成员", provider.text_chat.call_args.kwargs["system_prompt"]
        )

    async def test_database_failures_are_reported_and_request_guard_is_released(self):
        original_get = self.context.message_history_manager.get
        self.context.message_history_manager.get = AsyncMock(
            side_effect=RuntimeError("database offline")
        )
        self.assertIn("读取失败", (await self.run_command(MentionEvent()))[0])
        self.assertFalse(self.plugin.personal_handler._requests)
        self.context.message_history_manager.get = original_get
        self.context.message_history_manager.insert = AsyncMock(
            side_effect=RuntimeError("write failed")
        )
        result = await self.run_command(MentionEvent())
        self.assertEqual(result[0], "personal.png")
        self.assertIn("索引保存失败", result[1])

    async def test_overlapping_requests_do_not_replace_each_others_index(self):
        started = asyncio.Event()
        resume = asyncio.Event()
        original = self.plugin.personal_handler.service.analyze

        async def paused(event, report):
            started.set()
            await resume.wait()
            return await original(event, report)

        self.plugin.personal_handler.service.analyze = paused
        task = asyncio.create_task(self.run_command(MentionEvent()))
        await started.wait()
        try:
            self.assertIn("正在生成", (await self.run_command(MentionEvent()))[0])
        finally:
            resume.set()
            await task
        self.assertFalse(self.plugin.personal_handler._requests)


if __name__ == "__main__":
    unittest.main()
