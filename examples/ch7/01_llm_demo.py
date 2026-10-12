# 启动命令（在项目根目录执行）：
#     n
"""教程 7.2 演示：provider 自动检测 + 交互式对话（流式 / 非流式）。

在项目根目录的 ``.env`` 中配置 ``LLM_MODEL_ID`` / ``LLM_API_KEY`` / ``LLM_BASE_URL``
（字段与 ``reference/hello-agents/code/chapter7/.env.example`` 一致）；也可以直接把
``LLM_BASE_URL`` 指向本地 Ollama / vLLM，框架会自动识别 provider。

没有 API key 时，可把 ``LLM_BASE_URL`` 指向任意本地 OpenAI 兼容服务来验证脚本本身：
``LLM_BASE_URL=http://127.0.0.1:<port>/v1`` —— 此时跑的**不是**真实模型服务。

交互方式：

- 直接输入问题回车：流式输出（``think``），多轮保留上下文；
- ``/invoke <问题>``：非流式输出（``invoke``），用于对照两种调用路径；
- ``exit`` / ``quit`` 或 Ctrl-D：结束对话。
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from typing import Any

from myagent import Config, HelloAgentsLLM, Message

SYSTEM_PROMPT = "你是简洁的助手，回答不超过两句。"
QUIT_WORDS = frozenset({"exit", "quit"})
INVOKE_PREFIX = "/invoke"


def _stream(llm: HelloAgentsLLM, messages: list[Message]) -> str:
    """流式打印回答，返回完整文本（供多轮历史使用）。"""
    parts: list[str] = []
    for chunk in llm.think(messages):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        parts.append(chunk)
    print()
    return "".join(parts)


def _invoke(llm: HelloAgentsLLM, messages: list[Message]) -> str:
    """非流式打印回答，返回完整文本。"""
    answer = llm.invoke(messages)
    print(answer)
    return answer


def _read_question() -> str | None:
    """读取一行用户输入。

    Returns:
        问题文本；空字符串表示空行（跳过）；``None`` 表示退出。
    """
    try:
        raw = input("你: ")
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

    等待响应时没有任何代码在读 stdin：敲下的字符由 tty 直接回显、穿插进流式输出里
    （看起来像"删不掉"），并在下一轮 ``input()`` 时被立刻吃掉、当成新提问自动提交。
    这里在请求期间关闭回显与行缓冲，响应结束后恢复终端状态、丢弃等待期误敲的输入。
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


def main() -> int:
    config = Config.from_env()
    config.configure_logging()
    provider = None if config.default_provider == "auto" else config.default_provider

    llm = HelloAgentsLLM(
        model=config.default_model,
        provider=provider,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    print(f"检测到的 provider: {llm.provider}")
    print(f"model={llm.model} base_url={llm.base_url} timeout={llm.timeout}")
    print(f"输入问题开始对话；{INVOKE_PREFIX} <问题> 走非流式；exit 退出。\n")

    messages: list[Message] = [Message(SYSTEM_PROMPT, "system")]
    try:
        while True:
            question = _read_question()
            if question is None:
                break
            if not question:
                continue

            use_stream = True
            if question.startswith(INVOKE_PREFIX):
                question = question[len(INVOKE_PREFIX) :].strip()
                if not question:
                    print(f"用法：{INVOKE_PREFIX} <问题>")
                    continue
                use_stream = False

            messages.append(Message(question, "user"))
            print("助手: ", end="", flush=True)
            with _muted_stdin():
                answer = _stream(llm, messages) if use_stream else _invoke(llm, messages)
            messages.append(Message(answer, "assistant"))
    finally:
        llm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
