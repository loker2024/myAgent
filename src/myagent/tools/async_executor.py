"""异步工具执行器：在并发上限内并行执行多个工具调用。

教程定位：第七章 7.5.4 的 ``AsyncToolExecutor``。教程的 ``code/chapter7/`` 无对应脚本
（见 docs/deviations.md G1），以正文代码块为准，并按本项目约定修正以下偏离
（见 docs/deviations.md D24）：

- 教程用 ``concurrent.futures.ThreadPoolExecutor`` + ``__del__`` 清理；``__del__`` 的触发时机
  不可靠，这里改为纯 ``asyncio`` 实现，资源回收交给显式的 :meth:`AsyncToolExecutor.aclose`
  （同时支持作异步上下文管理器使用）；
- 并发上限由 ``asyncio.Semaphore`` 提供；``asyncio`` 原语会绑定到首次使用它的事件循环，
  因此信号量按「当前运行的事件循环」惰性创建，使同一实例跨多次 ``asyncio.run`` 复用仍安全；
- 教程的并行执行不校验任务缺键、也不透传 ``return_exceptions``；这里补上缺键校验
  （抛 :class:`ToolException`）与可选的异常捕获语义。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Final, Literal, overload

from ..core.exceptions import ToolException
from .registry import ToolInput, ToolRegistry

__all__ = ["AsyncToolExecutor"]

logger = logging.getLogger(__name__)

_REQUIRED_TASK_KEYS: Final[tuple[str, ...]] = ("tool_name", "input_data")


class AsyncToolExecutor:
    """并发执行工具调用的异步执行器。

    Args:
        registry: 提供工具的工具注册表。
        max_workers: 并发上限，通过信号量限制同时执行的任务数。

    Raises:
        ToolException: ``max_workers`` 小于 1 时。
    """

    def __init__(self, registry: ToolRegistry, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ToolException("并发上限必须为正整数", context={"max_workers": max_workers})
        self.registry = registry
        self.max_workers = max_workers
        self._closed = False
        self._semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    def __repr__(self) -> str:
        return f"AsyncToolExecutor(max_workers={self.max_workers}, closed={self._closed})"

    # ------------------------------------------------------------------ 执行

    async def execute_tool_async(self, tool_name: str, input_data: ToolInput) -> str:
        """在并发上限内异步执行单个工具。

        Raises:
            ToolException: 执行器已关闭，或工具执行失败时。
        """
        self._ensure_open()
        async with self._limiter():
            return await self.registry.aexecute_tool(tool_name, input_data)

    @overload
    async def execute_tools_parallel(
        self, tasks: Sequence[Mapping[str, str]], *, return_exceptions: Literal[False] = False
    ) -> list[str]: ...

    @overload
    async def execute_tools_parallel(
        self, tasks: Sequence[Mapping[str, str]], *, return_exceptions: Literal[True]
    ) -> list[Any]: ...

    async def execute_tools_parallel(
        self, tasks: Sequence[Mapping[str, str]], *, return_exceptions: bool = False
    ) -> list[Any]:
        """并发执行多个任务，结果**保持输入顺序**。

        Args:
            tasks: 每项形如 ``{"tool_name": ..., "input_data": ...}``。
            return_exceptions: 为 ``True`` 时把异常对象原样放进结果列表；为 ``False``（默认）
                时异常向上抛出。

        Returns:
            与 ``tasks`` 一一对应的结果列表；``tasks`` 为空时返回 ``[]``。

        Raises:
            ToolException: 执行器已关闭，或任务缺少必需键时（``return_exceptions=True`` 时
                后者会作为结果列表中的元素返回）。
        """
        self._ensure_open()
        if not tasks:
            return []
        pending = [self._run_task(index, task) for index, task in enumerate(tasks)]
        if return_exceptions:
            return list(await asyncio.gather(*pending, return_exceptions=True))
        return list(await asyncio.gather(*pending))

    # ------------------------------------------------------------------ 资源

    async def aclose(self) -> None:
        """关闭执行器；幂等，可重复调用。"""
        if self._closed:
            return
        self._closed = True
        self._semaphore = None
        self._semaphore_loop = None
        logger.debug("异步工具执行器已关闭")

    async def __aenter__(self) -> AsyncToolExecutor:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ 内部

    async def _run_task(self, index: int, task: Mapping[str, str]) -> str:
        missing = [key for key in _REQUIRED_TASK_KEYS if key not in task]
        if missing:
            raise ToolException(
                "并行任务缺少必需的键",
                context={"index": index, "missing": missing},
            )
        return await self.execute_tool_async(task["tool_name"], task["input_data"])

    def _limiter(self) -> asyncio.Semaphore:
        """取当前事件循环对应的信号量；循环变化时重建，保证跨 ``asyncio.run`` 复用安全。"""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._semaphore_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_workers)
            self._semaphore_loop = loop
        return self._semaphore

    def _ensure_open(self) -> None:
        if self._closed:
            raise ToolException("执行器已关闭", context={"executor": repr(self)})
