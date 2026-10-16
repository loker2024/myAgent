# 启动命令（在项目根目录执行）：
#     uv run python -X utf8 examples/ch7/07_tool_chain.py
"""教程 7.5.4 演示：ToolChain / ToolChainManager / AsyncToolExecutor（不依赖 LLM）。

本脚本演示工具系统的高级特性：

1. **工具链**：构造一条 2 步 ``ToolChain`` —— 先算表达式（输出键 ``value``），
   再对这个值求平方（输出键 ``squared``），经 ``ToolChainManager.execute_chain`` 执行；
2. **异步并行**：用 ``AsyncToolExecutor`` 并发执行 4 个计算任务，并与串行耗时对比；
3. **交互**：输入表达式走这条链，打印中间结果 ``value`` 与 ``squared``。

关于并行演示的一个必要说明：``CalculatorTool`` 的求值在 GIL 下是纯 CPU 工作，多线程
并行并不会变快，无法体现 ``AsyncToolExecutor`` 的价值。因此这里额外注册了一个演示用
的 ``slow_calc``（等待 0.25 秒模拟远程计算延迟后再调用 calculator），让「并发 < 串行」
可以被真实观测到。``slow_calc`` 内部同样只经 ``ToolRegistry.execute_tool`` 这一条路径。

交互方式：直接输入表达式；``/search <关键词>``（仅在配置了搜索密钥时）走一条含 search
的链；``exit`` / ``quit`` 或 Ctrl-D 退出。非法表达式会被 ``ToolException`` 接住并打印提示。
非交互场景（管道 / 重定向）下读到 EOF 后正常退出，退出码 0。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any

from myagent import (
    AsyncToolExecutor,
    CalculatorTool,
    SearchTool,
    ToolChain,
    ToolChainManager,
    ToolException,
    ToolRegistry,
)

QUIT_WORDS = frozenset({"exit", "quit"})
SEARCH_PREFIX = "/search"
SLOW_SECONDS = 0.25
SLOW_CALC_DESCRIPTION = "计算数学表达式，并模拟一次耗时 0.25 秒的远程调用（用于演示并发收益）"


def _make_slow_calc(registry: ToolRegistry) -> Callable[[str], str]:
    """构造 ``slow_calc`` 的函数体：先等待固定时长，再交给 calculator 求值。"""

    def slow_calc(expression: str) -> str:
        time.sleep(SLOW_SECONDS)
        return registry.execute_tool("calculator", expression)

    return slow_calc


def _build_registry() -> ToolRegistry:
    """注册 calculator 与演示用的 slow_calc。"""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_function("slow_calc", SLOW_CALC_DESCRIPTION, _make_slow_calc(registry))
    return registry


def _build_square_chain() -> ToolChain:
    """构造演示用的 2 步链：表达式 → value，value 的平方 → squared。"""
    chain = ToolChain("square", "先计算表达式，再对结果求平方")
    chain.add_step("calculator", "{input}", output_key="value")
    chain.add_step("calculator", "{value} * {value}", output_key="squared")
    return chain


def _show_chain_definition(chain: ToolChain) -> None:
    """打印链的步骤定义，便于对照后面看到的中间结果。"""
    print("== 工具链演示（不依赖 LLM）==")
    print(f"链 {chain.name}：{chain.description}")
    for index, step in enumerate(chain.steps, start=1):
        print(
            f"  第 {index} 步：{step.tool_name} "
            f"输入模板={step.input_template!r} 输出键={step.output_key}"
        )


def _run_chain_once(registry: ToolRegistry, manager: ToolChainManager, expression: str) -> None:
    """执行一次示例链，打印中间结果与最终结果。

    链的第 1 步输出（``value``）由 ``registry.execute_tool`` 单独取一份用于展示；
    整条链经 ``ToolChainManager.execute_chain`` 执行（``value`` 与 ``squared`` 均由此产生）。
    """
    value = registry.execute_tool("calculator", expression)
    squared = manager.execute_chain("square", expression)
    print(f"\n输入: {expression}")
    print(f"  第 1 步中间结果 value   = {value}")
    print(f"  第 2 步链最终结果 squared = {squared}\n")


async def _run_parallel_demo(registry: ToolRegistry) -> None:
    """用 AsyncToolExecutor 对比同一批任务的串行与并行耗时。"""
    expressions = ["1 + 1", "2 * 3", "sqrt(81)", "2 ** 10"]
    tasks = [{"tool_name": "slow_calc", "input_data": item} for item in expressions]

    async with AsyncToolExecutor(registry, max_workers=4) as executor:
        start = time.perf_counter()
        for expression in expressions:
            await executor.execute_tool_async("slow_calc", expression)
        serial_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        parallel = await executor.execute_tools_parallel(tasks)
        parallel_elapsed = time.perf_counter() - start

    print("== AsyncToolExecutor 并行演示 ==")
    print(f"4 个任务（每个 slow_calc 模拟耗时 {SLOW_SECONDS:.2f}s）：")
    for expression, value in zip(expressions, parallel, strict=True):
        print(f"  {expression} = {value}")
    verdict = "并发快于串行" if parallel_elapsed < serial_elapsed else "并发未快于串行"
    print(f"串行耗时 {serial_elapsed:.3f}s，并行耗时 {parallel_elapsed:.3f}s（{verdict}）\n")


async def _read_line(prompt: str) -> str | None:
    """读取一行用户输入。

    Returns:
        文本；空字符串表示空行（跳过）；``None`` 表示退出。
    """
    try:
        raw = await asyncio.to_thread(input, prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    text = raw.strip()
    if not text:
        return ""
    return None if text.lower() in QUIT_WORDS else text


@contextlib.contextmanager
def _muted_stdin() -> Iterator[None]:
    """执行期间静音终端输入（避免等待期的误敲被下一轮 ``input()`` 吃掉）。"""
    try:
        import termios
    except ImportError:
        yield
        return

    fd: int | None = None
    saved: list[Any] | None = None
    if sys.stdin.isatty():
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            muted = termios.tcgetattr(fd)
            muted[3] &= ~(termios.ECHO | termios.ICANON)
            termios.tcsetattr(fd, termios.TCSANOW, muted)
        except (OSError, ValueError):
            fd = saved = None

    try:
        yield
    finally:
        if fd is not None and saved is not None:
            try:
                termios.tcflush(fd, termios.TCIFLUSH)
                termios.tcsetattr(fd, termios.TCSANOW, saved)
            except (OSError, ValueError):
                pass


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 屏蔽逐请求 INFO 噪音

    registry = _build_registry()
    chain = _build_square_chain()
    manager = ToolChainManager(registry)
    manager.register_chain(chain)

    _show_chain_definition(chain)
    _run_chain_once(registry, manager, "(3 + 4) * 2")
    await _run_parallel_demo(registry)

    search_chain: ToolChain | None = None
    search = SearchTool()
    if search.available_backends:
        registry.register_tool(search)
        search_chain = ToolChain("search_first", "把关键词交给搜索工具并返回结果")
        search_chain.add_step("search", "{input}", output_key="found")
        manager.register_chain(search_chain)
        print("已配置搜索密钥：额外注册了 search_first 链（交互里用 /search 触发）。\n")
    else:
        print("提示：未配置搜索密钥，跳过含 search 的链（/search 会被拒绝）。\n")

    print("输入表达式：走这条链，打印 value 与 squared；exit / quit 或 Ctrl-D 退出。\n")

    while True:
        expression = await _read_line("> ")
        if expression is None:
            break
        if not expression:
            continue

        if expression.startswith(SEARCH_PREFIX):
            if search_chain is None:
                print("未配置搜索密钥，无法使用 /search。\n")
                continue
            query = expression[len(SEARCH_PREFIX) :].strip()
            if not query:
                print(f"用法：{SEARCH_PREFIX} <关键词>\n")
                continue
            try:
                print(f"搜索结果:\n{manager.execute_chain('search_first', query)}\n")
            except ToolException as exc:
                print(f"搜索失败：{exc}\n")
            continue

        try:
            with _muted_stdin():
                value = registry.execute_tool("calculator", expression)
                squared = manager.execute_chain("square", expression)
        except ToolException as exc:
            print(f"无法计算：{exc}\n")
            continue
        print(f"value   = {value}")
        print(f"squared = {squared}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
