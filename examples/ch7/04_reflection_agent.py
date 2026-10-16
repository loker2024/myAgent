# 启动命令（在项目根目录执行）：
#     uv run python -X utf8 examples/ch7/04_reflection_agent.py
"""教程 7.4.3 演示：ReflectionAgent —— 先出初稿，再「自我审查 → 按反馈改进」。

``ReflectionAgent(max_iterations=2)`` 的每轮流程：

1. ``initial``：针对任务给出第一版回答；
2. ``reflect``：审查上一版回答，给出改进建议（若判定「无需改进」则提前结束）；
3. ``refine``：按建议产出改进版，作为下一轮的起点。

本脚本演示：

- 每轮打印最终回答；
- 打开 INFO 日志，展示反思轮次的诊断信息（判定无需改进时会打印日志）。

交互方式：直接输入任务回车；``exit`` / ``quit`` 或 Ctrl-D 结束。
单轮若发生 ``AgentException``（例如提示词渲染失败、任务为空），只打印错误并继续接受下一轮。
非交互场景（管道 / 重定向）下读到 EOF 后正常退出，退出码 0。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import Iterator
from typing import Any

from myagent import AgentException, Config, HelloAgentsLLM, ReflectionAgent

SYSTEM_PROMPT = "你是一名严谨的写作助手，回答用中文，条理清晰。"
MAX_ITERATIONS = 2
QUIT_WORDS = frozenset({"exit", "quit"})


async def _read_question() -> str | None:
    """读取一行用户输入。

    Returns:
        任务文本；空字符串表示空行（跳过）；``None`` 表示退出。
    """
    try:
        raw = await asyncio.to_thread(input, "任务: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    text = raw.strip()
    if not text:
        return ""
    return None if text.lower() in QUIT_WORDS else text


@contextlib.contextmanager
def _muted_stdin() -> Iterator[None]:
    """请求模型期间静音终端输入（避免等待期的误敲被下一轮 ``input()`` 吃掉）。"""
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
    config = Config.from_env()
    config.configure_logging()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # 屏蔽逐请求 INFO 噪音

    provider = None if config.default_provider == "auto" else config.default_provider
    llm = HelloAgentsLLM(
        model=config.default_model,
        provider=provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = ReflectionAgent(
        name="reflection",
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        config=config,
        max_iterations=MAX_ITERATIONS,
    )

    print("== ReflectionAgent 演示 ==")
    print(f"provider={llm.provider} model={llm.model} max_iterations={MAX_ITERATIONS}")
    print("\n输入任务开始；exit / quit 或 Ctrl-D 退出。\n")

    try:
        while True:
            task = await _read_question()
            if task is None:
                break
            if not task:
                continue

            try:
                with _muted_stdin():
                    answer = await agent.arun(task)
            except AgentException as exc:
                print(f"反思失败：{exc}\n")
                continue
            print(f"最终回答:\n{answer}\n")
    finally:
        await llm.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
