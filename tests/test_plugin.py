from __future__ import annotations

import datetime as dt
import importlib
import sys
import types
import unittest
from pathlib import Path


def install_astrbot_stubs() -> None:
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event_module = types.ModuleType("astrbot.api.event")
    star_module = types.ModuleType("astrbot.api.star")
    components_module = types.ModuleType("astrbot.api.message_components")

    class At:
        def __init__(self, qq, name=""):
            self.qq = qq
            self.name = name

    class Plain:
        def __init__(self, text):
            self.text = text

    class Logger:
        def warning(self, *args, **kwargs) -> None:
            pass

        def info(self, *args, **kwargs) -> None:
            pass

    class Filters:
        class EventMessageType:
            ALL = "all"

        @staticmethod
        def regex(*args, **kwargs):
            return lambda function: function

        @staticmethod
        def event_message_type(*args, **kwargs):
            return lambda function: function

    class AstrMessageEvent:
        pass

    class Context:
        pass

    class Star:
        def __init__(self, context) -> None:
            self.context = context

    api.logger = Logger()
    event_module.AstrMessageEvent = AstrMessageEvent
    event_module.filter = Filters
    star_module.Context = Context
    star_module.Star = Star
    components_module.At = At
    components_module.Plain = Plain
    astrbot.api = api
    sys.modules.setdefault("astrbot", astrbot)
    sys.modules.setdefault("astrbot.api", api)
    sys.modules.setdefault("astrbot.api.event", event_module)
    sys.modules.setdefault("astrbot.api.star", star_module)
    sys.modules.setdefault("astrbot.api.message_components", components_module)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
install_astrbot_stubs()
plugin_module = importlib.import_module("astrbot_plugin_group_summary.main")
GroupSummaryPlugin = plugin_module.GroupSummaryPlugin


class MessageObject:
    timestamp = int(dt.datetime.now().astimezone().timestamp())
    message_id = "m-1"


class FakeEvent:
    def __init__(self, text: str, group_id: str = "100", sender_id: str = "1") -> None:
        self.text = text
        self.group_id = group_id
        self.sender_id = sender_id
        self.message_obj = MessageObject()
        self.created_at = self.message_obj.timestamp
        self.unified_msg_origin = "fake:group:100"
        self.llm_flag = None

    def get_group_id(self):
        return self.group_id

    def get_sender_id(self):
        return self.sender_id

    def get_self_id(self):
        return "999"

    def get_sender_name(self):
        return "测试成员"

    def get_message_str(self):
        return self.text

    def get_message_outline(self):
        return "[At:999] " + self.text

    def get_messages(self):
        return []

    def get_platform_id(self):
        return "fake-platform"

    def should_call_llm(self, value):
        self.llm_flag = value

    def plain_result(self, value):
        return value

    def image_result(self, value):
        return value


class FakeHistoryManager:
    def __init__(self, rows=None) -> None:
        self.rows = rows or []
        self.inserted = []
        self.scoped_rows = {}

    async def insert(self, **kwargs):
        self.inserted.append(kwargs)
        key = (kwargs["platform_id"], kwargs["user_id"])
        if "personal_index" in kwargs["user_id"]:
            self.scoped_rows[key] = [types.SimpleNamespace(content=kwargs["content"])]

    async def get(self, **kwargs):
        key = (kwargs["platform_id"], kwargs["user_id"])
        rows = self.scoped_rows.get(
            key, [] if "personal_index" in kwargs["user_id"] else self.rows
        )
        offset = (kwargs["page"] - 1) * kwargs["page_size"]
        return rows[offset : offset + kwargs["page_size"]]


class FakeContext:
    def __init__(self, rows=None) -> None:
        self.message_history_manager = FakeHistoryManager(rows)

    async def get_using_provider_async(self, umo=None):
        return None


class PluginIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_command_is_not_recorded_when_outline_contains_at(
        self,
    ) -> None:
        context = FakeContext()
        plugin = GroupSummaryPlugin(context)
        await plugin.history.store(FakeEvent("群聊总结"))
        self.assertEqual(context.message_history_manager.inserted, [])

    async def test_regular_group_message_is_recorded_with_retention_limit(self) -> None:
        context = FakeContext()
        plugin = GroupSummaryPlugin(context)
        await plugin.history.store(FakeEvent("普通消息"))
        self.assertEqual(len(context.message_history_manager.inserted), 1)
        inserted = context.message_history_manager.inserted[0]
        self.assertEqual(inserted["max_messages"], 20_000)
        self.assertEqual(inserted["content"]["plugin"], "group_summary")

    async def test_private_message_is_not_recorded(self) -> None:
        context = FakeContext()
        plugin = GroupSummaryPlugin(context)
        await plugin.history.store(FakeEvent("普通消息", group_id=""))
        self.assertEqual(context.message_history_manager.inserted, [])

    async def test_no_history_returns_explanatory_fallback_without_model(self) -> None:
        context = FakeContext()
        plugin = GroupSummaryPlugin(context)
        event = FakeEvent("群聊总结")
        results = [result async for result in plugin.group_summary(event)]
        self.assertEqual(len(results), 1)
        self.assertIn("还没有可总结的群聊记录", results[0])
        self.assertIn("启用后", results[0])
        self.assertFalse(event.llm_flag)

    async def test_invalid_date_returns_usage(self) -> None:
        context = FakeContext()
        plugin = GroupSummaryPlugin(context)
        event = FakeEvent("群聊总结 下周一")
        results = [result async for result in plugin.group_summary(event)]
        self.assertIn("日期格式无法识别", results[0])


if __name__ == "__main__":
    unittest.main()
