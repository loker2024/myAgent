# 启动命令（在项目根目录执行）：
#     uv run python -X utf8 examples/ch7/02_simple_agent.py
"""教程 7.4.1 演示：SimpleAgent —— 基础对话范式 + 工具调用。

``SimpleAgent`` 是最小可用的对话 Agent：把「系统提示词 + 历史 + 当前输入」交给 LLM，
并从模型输出里识别 ``[TOOL_CALL:工具名:参数]`` 标记、执行工具、把结果回灌后继续回答。

本脚本演示：

- 注册 ``CalculatorTool``；若配置了搜索密钥（``TAVILY_API_KEY`` / ``SERPAPI_API_KEY``），
  再注册 ``SearchTool``；
- 启动时打印可用工具清单；
- 每轮把用户输入交给 ``SimpleAgent.arun``，打印最终回答，多轮之间保留上下文。

交互方式：直接输入问题回车；``exit`` / ``quit`` 或 Ctrl-D 结束。
非交互场景（管道 / 重定向）下读到 EOF 后正常退出，退出码 0。

注意：Agent 是否真的调用工具取决于模型是否能按约定格式输出工具调用标记。
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
    SearchTool,
    SimpleAgent,
    ToolRegistry,
)

SYSTEM_PROMPT = (
    "你是一个善于使用工具的助手：需要精确计算时，先调用 calculator 工具再回答；"
    "回答保持简洁，用中文。"
)
QUIT_WORDS = frozenset({"exit", "quit"})


def _register_tools() -> ToolRegistry:
    """注册演示用工具；搜索工具只在配置了密钥时注册。"""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    search = SearchTool()
    if search.available_backends:
        registry.register_tool(search)
    else:
        print(
            "提示：未配置 TAVILY_API_KEY / SERPAPI_API_KEY，本次只启用计算器工具"
            "（可用 /search 的解析见 06_tools.py）。"
        )
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
    """请求模型期间静音终端输入。

    等待响应时没有代码在读 stdin：敲下的字符会被 tty 回显、穿插进输出里，并在下一轮
    ``input()`` 时被立刻吃掉。这里在请求期间关闭回显与行缓冲，结束后恢复终端状态。
    """
    try:
        import termios
    except ImportError:  # 非 POSIX 平台没有 termios，也不会出现该问题
        yield
        return

    fd: int | None = None
    saved: list[Any] | None = None
    if sys.stdin.isatty():
        try:
            fd = sys.stdin.fileno()
            saved = termios.tcgetattr(fd)
            muted = termios.tcgetattr(fd)
            # c_lflag：关 ECHO（不回显）与 ICANON（不按行缓冲误敲的字符）
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
    agent = SimpleAgent(
        name="simple",
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        config=config,
        tool_registry=registry,
    )

    print("== SimpleAgent 演示 ==")
    print(f"provider={llm.provider} model={llm.model} base_url={llm.base_url}")
    print("可用工具：")
    print(registry.get_tools_description())
    print("\n输入问题开始对话；exit / quit 或 Ctrl-D 退出。\n")

    try:
        while True:
            question = await _read_question()
            if question is None:
                break
            if not question:
                continue
            with _muted_stdin():
                answer = await agent.arun(question)
            print(f"助手: {answer}\n")
    finally:
        await llm.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
