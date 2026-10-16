"""异步工具执行器（AsyncToolExecutor）的行为测试，全部离线。"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import pytest

from myagent.core.exceptions import ToolException
from myagent.tools.async_executor import AsyncToolExecutor
from myagent.tools.registry import ToolRegistry


class _ConcurrencyTracker:
    """记录同时执行中的任务数与峰值并发数。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        with self._lock:
            self.current -= 1


def _delayed_echo(text: str) -> str:
    """耗时与输入数字成反比，用于打乱完成顺序。"""
    time.sleep((3 - int(text)) * 0.05)
    return f"echo:{text}"


def _make_slow(tracker: _ConcurrencyTracker, delay: float) -> Callable[[str], str]:
    def slow(text: str) -> str:
        tracker.enter()
        try:
            time.sleep(delay)
        finally:
            tracker.exit()
        return text

    return slow


def _boom(text: str) -> str:
    raise RuntimeError("炸了")


def _registry(
    delay: float = 0.1, *, with_tracker: _ConcurrencyTracker | None = None
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_function("echo", "回显", lambda text: f"echo:{text}")
    registry.register_function("delayed", "延迟回显", _delayed_echo)
    registry.register_function("boom", "总是失败", _boom)
    if with_tracker is not None:
        registry.register_function("slow", "慢工具", _make_slow(with_tracker, delay))
    return registry


def _tasks(count: int, tool: str = "echo") -> list[dict[str, str]]:
    return [{"tool_name": tool, "input_data": str(index)} for index in range(count)]


async def test_parallel_results_follow_input_order() -> None:
    executor = AsyncToolExecutor(_registry(), max_workers=4)

    results = await executor.execute_tools_parallel(_tasks(3, "delayed"))

    assert results == ["echo:0", "echo:1", "echo:2"]


async def test_parallel_runs_concurrently() -> None:
    tracker = _ConcurrencyTracker()
    executor = AsyncToolExecutor(_registry(0.1, with_tracker=tracker), max_workers=4)

    start = time.perf_counter()
    results = await executor.execute_tools_parallel(_tasks(4, "slow"))
    elapsed = time.perf_counter() - start

    assert results == ["0", "1", "2", "3"]
    assert elapsed < 0.3  # 串行需要约 0.4s


async def test_max_workers_caps_concurrency() -> None:
    tracker = _ConcurrencyTracker()
    executor = AsyncToolExecutor(_registry(0.05, with_tracker=tracker), max_workers=2)

    await executor.execute_tools_parallel(_tasks(6, "slow"))

    assert tracker.peak <= 2
    assert tracker.peak == 2  # 至少有两项同时在跑，说明确实并发


async def test_missing_key_raises_with_index_and_missing() -> None:
    executor = AsyncToolExecutor(_registry())

    with pytest.raises(ToolException) as excinfo:
        await executor.execute_tools_parallel([{"input_data": "x"}])

    assert excinfo.value.context["index"] == 0
    assert excinfo.value.context["missing"] == ["tool_name"]


async def test_return_exceptions_true_keeps_exception_objects() -> None:
    executor = AsyncToolExecutor(_registry())
    tasks = [
        {"tool_name": "echo", "input_data": "a"},
        {"input_data": "b"},
        {"tool_name": "boom", "input_data": "c"},
    ]

    results = await executor.execute_tools_parallel(tasks, return_exceptions=True)

    assert results[0] == "echo:a"
    assert isinstance(results[1], ToolException)
    assert results[1].context["index"] == 1
    assert isinstance(results[2], ToolException)


async def test_return_exceptions_false_propagates_tool_failure() -> None:
    executor = AsyncToolExecutor(_registry())

    with pytest.raises(ToolException) as excinfo:
        await executor.execute_tools_parallel([{"tool_name": "boom", "input_data": "c"}])

    assert excinfo.value.context["tool"] == "boom"


async def test_empty_tasks_returns_empty_list() -> None:
    executor = AsyncToolExecutor(_registry())

    assert await executor.execute_tools_parallel([]) == []


async def test_aclose_is_idempotent_and_blocks_further_use() -> None:
    executor = AsyncToolExecutor(_registry())

    await executor.aclose()
    await executor.aclose()  # 幂等

    with pytest.raises(ToolException) as excinfo:
        await executor.execute_tool_async("echo", "a")
    assert "executor" in excinfo.value.context

    with pytest.raises(ToolException):
        await executor.execute_tools_parallel(_tasks(1))


async def test_async_context_manager_closes_executor() -> None:
    async with AsyncToolExecutor(_registry()) as executor:
        assert await executor.execute_tool_async("echo", "a") == "echo:a"

    with pytest.raises(ToolException):
        await executor.execute_tool_async("echo", "a")


def test_semaphore_is_reusable_across_event_loops() -> None:
    tracker = _ConcurrencyTracker()
    executor = AsyncToolExecutor(_registry(0.02, with_tracker=tracker), max_workers=2)
    tasks = _tasks(4, "slow")

    first = asyncio.run(executor.execute_tools_parallel(tasks))
    second = asyncio.run(executor.execute_tools_parallel(tasks))

    assert first == ["0", "1", "2", "3"]
    assert second == first


def test_non_positive_max_workers_is_rejected() -> None:
    with pytest.raises(ToolException) as excinfo:
        AsyncToolExecutor(_registry(), max_workers=0)

    assert excinfo.value.context["max_workers"] == 0
