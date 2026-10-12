# 启动命令（在项目根目录执行）：
#     uv run python -X utf8 examples/ch7/05_plan_solve_agent.py
"""教程 7.4.4 演示：PlanAndSolveAgent —— 先规划，再逐步执行。

``PlanAndSolveAgent`` 把一次任务拆成两个阶段：

1. ``planner``：让模型给出编号步骤的计划；
2. ``executor``：按顺序执行每一步，并把「前面各步的结果」作为历史交给下一步。

本脚本演示：

- 每轮打印计划（编号步骤）、每一步的结果与最终答案（最后一步的输出）。

交互方式：直接输入问题回车；``exit`` / ``quit`` 或 Ctrl-D 结束。
单轮若发生 ``AgentException``（例如规划输出无法解析成步骤），只打印错误并继续接受下一轮。
非交互场景（管道 / 重定向）下读到 EOF 后正常退出，退出码 0。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import Iterator
from typing import Any

from myagent import AgentException, Config, HelloAgentsLLM, PlanAndSolveAgent

SYSTEM_PROMPT = "你是一名善于拆解问题的助手，回答用中文。"
QUIT_WORDS = frozenset({"exit", "quit"})


async def _read_question() -> str | None:
    """读取一行用户输入。

    Returns:
        问题文本；空字符串表示空行（跳过）；``None`` 表示退出。
    """
    try:
        raw = await asyncio.to_thread(input, "问题: ")
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
    agent = PlanAndSolveAgent(
        name="plan_solve",
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        config=config,
    )

    print("== PlanAndSolveAgent 演示 ==")
    print(f"provider={llm.provider} model={llm.model}")
    print("\n输入问题开始；exit / quit 或 Ctrl-D 退出。\n")

    try:
        while True:
            question = await _read_question()
            if question is None:
                break
            if not question:
                continue

            try:
                with _muted_stdin():
                    result = await agent.arun_detailed(question)
            except AgentException as exc:
                print(f"规划/执行失败：{exc}\n")
                continue

            print("\n计划：")
            if not result.plan:
                print("  （规划器没有给出任何步骤）")
            for index, step in enumerate(result.plan, start=1):
                print(f"  {index}. {step}")
            print("\n各步结果：")
            for index, output in enumerate(result.step_results, start=1):
                print(f"  [步骤 {index}] {output}")
            print(f"\n最终答案: {result.answer}\n")
    finally:
        await llm.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
