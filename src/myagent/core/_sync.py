"""跨事件循环的同步包装。

教程中没有对应文件：教程全部 API 是同步的。本项目以 async 为唯一真实实现，
同步入口统一经由此模块包装（见 docs/deviations.md D4）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

__all__ = ["ensure_no_running_loop", "run_sync"]

_RUNNING_LOOP_MESSAGE = (
    "检测到正在运行的事件循环：同步接口（run / think / invoke）不能在协程内调用，"
    "请改用异步接口（arun / athink / ainvoke）。"
)


def ensure_no_running_loop() -> None:
    """若当前线程已有运行中的事件循环则抛错。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(_RUNNING_LOOP_MESSAGE)


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """在同步上下文中执行协程。

    Args:
        coro: 待执行的协程对象。

    Returns:
        协程的返回值。

    Raises:
        RuntimeError: 当前线程已有运行中的事件循环（此时无法安全地阻塞等待）。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()  # 避免 "coroutine was never awaited" 警告
    raise RuntimeError(_RUNNING_LOOP_MESSAGE)
