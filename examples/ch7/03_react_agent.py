# 启动命令（在项目根目录执行）：
#     uv run python -X utf8 examples/ch7/03_react_agent.py
"""教程 7.4.2 演示：ReActAgent —— 推理（Reason）与行动（Act）交替进行。

``ReActAgent`` 每轮让模型输出 ``Thought`` 与 ``Action``：

- ``Action`` 形如 ``工具名[参数]`` 时会真的执行工具，把结果作为 ``Observation`` 写回历史；
- ``Action`` 形如 ``Finish[最终答案]`` 时结束循环。

本脚本演示：

- ``ReActAgent(max_steps=5)`` + ``CalculatorTool``（有搜索密钥时再加 ``SearchTool``）；
- 每轮打印 ``stop_reason``、逐步的 ``thought`` / ``action`` / ``observation`` 与最终答案。

交互方式：直接输入问题回车；``exit`` / ``quit`` 或 Ctrl-D 结束。
非交互场景（管道 / 重定向）下读到 EOF 后正常退出，退出码 0。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import Iterator
from typing import Any

from myagent import (
    CalculatorTool,
    Config,
    HelloAgentsLLM,
    ReActAgent,
    SearchTool,
    ToolRegistry,
)

MAX_STEPS = 5
QUIT_WORDS = frozenset({"exit", "quit"})


def _register_tools() -> ToolRegistry:
    """注册演示用工具；搜索工具只在配置了密钥时注册。"""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    search = SearchTool()
    if search.available_backends:
        registry.register_tool(search)
    else:
        print("提示：未配置搜索密钥，本次只注册计算器工具。")
    return registry


async def _read_question() -> str | None:
    """读取一行用户输入。

    Returns:
        问题文本；空字符串表示空行（跳过）；``None`` 表示退出。
    """
    try:
        raw = await asyncio.to_thread(input, "你: ")
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

    registry = _register_tools()
    provider = None if config.default_provider == "auto" else config.default_provider
    llm = HelloAgentsLLM(
        model=config.default_model,
        provider=provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    agent = ReActAgent(
        name="react",
        llm=llm,
        tool_registry=registry,
        config=config,
        max_steps=MAX_STEPS,
    )

    print("== ReActAgent 演示 ==")
    print(f"provider={llm.provider} model={llm.model} max_steps={MAX_STEPS}")
    print("可用工具：")
    print(registry.get_tools_description())
    print("\n输入问题开始；exit / quit 或 Ctrl-D 退出。\n")

    try:
        while True:
            question = await _read_question()
            if question is None:
                break
            if not question:
                continue

            with _muted_stdin():
                result = await agent.arun_traced(question)

            print(f"\nstop_reason = {result.stop_reason}")
            if not result.steps:
                print("（没有产生任何步骤）")
            for step in result.steps:
                print(f"[第 {step.index} 步]")
                if step.thought is not None:
                    print(f"  thought: {step.thought}")
                if step.action is not None:
                    print(f"  action: {step.action}")
                if step.observation is not None:
                    print(f"  observation: {step.observation}")
            print(f"最终答案: {result.answer}\n")
    finally:
        await llm.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
