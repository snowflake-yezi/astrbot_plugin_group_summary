from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from test_plugin import FakeContext, FakeEvent, GroupSummaryPlugin, plugin_module
from test_personal import MentionEvent, message, rows_for
from astrbot_plugin_group_summary.features.personal import (
    handler as personal_handler_module,
)


class EntrypointTests(unittest.IsolatedAsyncioTestCase):
    def test_handlers_remain_async_generators_owned_by_plugin_module(self):
        for name in ("track_group_messages", "group_summary", "personal_chat"):
            handler = GroupSummaryPlugin.__dict__[name]
            self.assertTrue(inspect.isasyncgenfunction(handler))
            self.assertEqual(handler.__module__, plugin_module.__name__)
            self.assertEqual(handler.__qualname__, f"GroupSummaryPlugin.{name}")

    async def test_summary_entries_delegate_every_result_and_close_handlers(self):
        plugin = GroupSummaryPlugin(FakeContext())
        event = MentionEvent()
        for entry, owner in (
            (plugin.group_summary, plugin.group_handler),
            (plugin.personal_chat, plugin.personal_handler),
        ):
            closed = []

            async def results():
                try:
                    yield "第一条结果"
                    yield "第二条结果"
                finally:
                    closed.append(True)

            with patch.object(owner, "handle", return_value=results()) as handler:
                self.assertEqual(
                    [result async for result in entry(event)],
                    ["第一条结果", "第二条结果"],
                )
                handler.assert_called_once_with(event)
            self.assertEqual(closed, [True])

    async def test_closing_personal_entry_releases_request_guard(self):
        plugin = GroupSummaryPlugin(FakeContext(rows_for([message("2", 0, "发言")])))
        with patch.object(
            personal_handler_module,
            "render_personal_report",
            return_value=Path("personal.png"),
        ):
            results = plugin.personal_chat(MentionEvent())
            self.assertEqual(await anext(results), "personal.png")
            self.assertTrue(plugin.personal_handler._requests)
            await results.aclose()
            self.assertFalse(plugin.personal_handler._requests)

    async def test_recording_entry_persists_existing_schema_and_session_key(self):
        context = FakeContext()
        plugin = GroupSummaryPlugin(context)
        results = [
            result
            async for result in plugin.track_group_messages(FakeEvent("普通消息"))
        ]
        self.assertEqual(results, [])
        inserted = context.message_history_manager.inserted[0]
        self.assertEqual(inserted["user_id"], "group_summary:group:100")
        self.assertEqual(inserted["content"]["plugin"], "group_summary")
        self.assertEqual(inserted["content"]["schema_version"], 1)
        self.assertEqual(inserted["max_messages"], 20_000)

    async def test_recording_failure_remains_isolated(self):
        context = FakeContext()
        context.message_history_manager.insert = AsyncMock(
            side_effect=RuntimeError("database offline")
        )
        plugin = GroupSummaryPlugin(context)
        self.assertEqual(
            [
                result
                async for result in plugin.track_group_messages(FakeEvent("普通消息"))
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
