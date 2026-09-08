"""Run with python -B tests/benchmark_personal.py; service delays are simulated."""

from __future__ import annotations

import asyncio
import cProfile
import json
import pstats
import time
import types
from unittest.mock import AsyncMock

from test_personal import MentionEvent, message, rows_for
from test_plugin import FakeContext, GroupSummaryPlugin
from astrbot_plugin_group_summary.personal_report import (
    build_detail_report,
    build_personal_fallback,
    build_personal_prompt,
    build_personal_report,
    render_personal_report,
)


async def benchmark() -> None:
    messages = [
        message("2" if index % 5 == 0 else "3", (index // 1600) * 120 + (index % 1600) // 60,
                "讨论版本配置和排查结果。" * 12, message_id=str(index))
        for index in range(20_000)
    ]
    context = FakeContext(rows_for(messages))
    original_get = context.message_history_manager.get
    queries = 0

    async def delayed_get(**kwargs):
        nonlocal queries
        queries += 1
        await asyncio.sleep(0.01)
        return await original_get(**kwargs)

    context.message_history_manager.get = delayed_get
    plugin = GroupSummaryPlugin(context)
    event = MentionEvent()
    started = time.perf_counter()
    loaded = await plugin._load_messages(event)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    target = [item for item in loaded if item.sender_id == "2"]
    report = build_personal_report(loaded, target, "性能验证模拟记录")
    build_seconds = time.perf_counter() - started
    prompt_chars = len(build_personal_prompt(report))
    started = time.perf_counter()
    path = render_personal_report(report, build_personal_fallback(report))
    render_seconds = time.perf_counter() - started
    image_bytes = path.stat().st_size
    path.unlink(missing_ok=True)

    response = json.dumps({"overview": "模拟分析", "periods": [
        {"id": 1, "title": "讨论配置", "summary": "讨论配置和排查结果", "context": "群友参与讨论"},
    ]})

    async def simulated_model(**kwargs):
        await asyncio.sleep(0.05)
        return types.SimpleNamespace(completion_text=response)

    provider = types.SimpleNamespace(text_chat=AsyncMock(side_effect=simulated_model))
    context.get_using_provider_async = AsyncMock(return_value=provider)
    started = time.perf_counter()
    for page in (1, 2):
        detail = build_detail_report(loaded, list(report.periods[0].messages), 1, page)
        await plugin._analyze_personal(event, detail)
    page_analysis_seconds = time.perf_counter() - started

    long_messages = [message("2", index, "中文长消息排版测试，含时间 09:00 和 English words。" * 35,
                             message_id=str(index)) for index in range(12)]
    long_report = build_detail_report(long_messages, long_messages, 1, 1)
    profiler = cProfile.Profile()
    profiler.enable()
    started = time.perf_counter()
    path = render_personal_report(long_report, build_personal_fallback(long_report))
    long_render_seconds = time.perf_counter() - started
    profiler.disable()
    path.unlink(missing_ok=True)
    print(json.dumps({
        "messages": len(loaded), "database_queries": queries, "simulated_query_delay_ms": 10,
        "load_seconds": round(load_seconds, 3), "build_seconds": round(build_seconds, 3),
        "overview_prompt_chars": prompt_chars, "overview_render_seconds": round(render_seconds, 3),
        "overview_image_bytes": image_bytes, "two_detail_pages_model_calls": provider.text_chat.await_count,
        "two_detail_pages_analysis_seconds": round(page_analysis_seconds, 3),
        "long_detail_render_seconds_profiled": round(long_render_seconds, 3),
    }, indent=2))
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(10)


if __name__ == "__main__":
    asyncio.run(benchmark())
