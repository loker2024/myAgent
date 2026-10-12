# 启动命令（在项目根目录执行）：
#     uv run python -X utf8 examples/ch7/06_tools.py
"""教程 7.5.1 ~ 7.5.3 演示：工具系统 —— 注册、发现、schema 与执行。

本脚本展示工具系统的完整链路：

- 用 ``register_tool`` 注册类工具 ``CalculatorTool``（自带参数声明与校验）；
- 用 ``register_function`` 把普通函数适配成 ``FunctionTool``（参数声明由函数签名推断）；
- 打印 ``get_tools_description()``（可直接放进提示词的工具清单）；
- 打印一个工具的 ``to_openai_schema()``（OpenAI function calling 格式）；
- 两种注册方式最终走同一条执行路径 ``ToolRegistry.execute_tool``。

进入交互后：

- 直接输入表达式 → 交给 ``calculator`` 求值并打印结果；
- ``/shout <文本>`` → 调用示例函数工具；
- ``/search <关键词>`` → 仅在配置了搜索密钥时可用；
- ``exit`` / ``quit`` 或 Ctrl-D 结束（非交互场景读到 EOF 后正常退出，退出码 0）。

工具抛出的 ``ToolException`` 会被接住并打印友好提示，不会让脚本崩溃。
"""

from __future__ import annotations

import json

from myagent import CalculatorTool, Config, SearchTool, ToolException, ToolRegistry

QUIT_WORDS = frozenset({"exit", "quit"})
SHOUT_PREFIX = "/shout"
SEARCH_PREFIX = "/search"


def shout(text: str) -> str:
    """示例函数工具：把文本转成大写并统计字符数。"""
    stripped = text.strip()
    return f"{stripped.upper()}（共 {len(stripped)} 个字符）"


def _read_line(prompt: str) -> str | None:
    """读取一行输入。

    Returns:
        去掉首尾空白的文本；空字符串表示空行（跳过）；``None`` 表示退出。
    """
    try:
        raw = input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    text = raw.strip()
    if not text:
        return ""
    return None if text.lower() in QUIT_WORDS else text


def _build_registry() -> tuple[ToolRegistry, bool]:
    """注册工具并返回 (注册表, 是否启用了搜索)。"""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_function(
        "shout",
        "把给定文本转成大写并统计字符数",
        shout,
    )
    search = SearchTool()
    if search.available_backends:
        registry.register_tool(search)
        return registry, True
    print("提示：未配置 TAVILY_API_KEY / SERPAPI_API_KEY，搜索工具不可用（/search 会被拒绝）。")
    return registry, False


def _print_toolset(registry: ToolRegistry) -> None:
    """打印工具清单与一个 OpenAI function calling schema。"""
    print("== 已注册工具 ==")
    print(registry.get_tools_description())

    print("\n== OpenAI function calling schema（示例：shout）==")
    tool = registry.get_tool("shout")
    if tool is not None:
        print(json.dumps(tool.to_openai_schema(), ensure_ascii=False, indent=2))

    print("\n== 直接执行一次函数工具（两种注册方式共用 execute_tool 路径）==")
    print(
        f"execute_tool('shout', 'hello myagent') -> {registry.execute_tool('shout', 'hello myagent')}"
    )


def _run_lookup(registry: ToolRegistry, search_available: bool, line: str) -> None:
    """处理 ``/search`` 与 ``/shout`` 前缀命令。"""
    if line.startswith(SEARCH_PREFIX):
        if not search_available:
            print("未配置搜索密钥，无法使用 /search。\n")
            return
        query = line[len(SEARCH_PREFIX) :].strip()
        if not query:
            print(f"用法：{SEARCH_PREFIX} <关键词>")
            return
        try:
            print(f"搜索结果:\n{registry.execute_tool('search', query)}\n")
        except ToolException as exc:
            print(f"搜索失败：{exc}\n")
        return

    text = line[len(SHOUT_PREFIX) :].strip()
    if not text:
        print(f"用法：{SHOUT_PREFIX} <文本>")
        return
    try:
        print(f"结果: {registry.execute_tool('shout', text)}\n")
    except ToolException as exc:
        print(f"调用失败：{exc}\n")


def main() -> int:
    Config.from_env().configure_logging()
    registry, search_available = _build_registry()
    _print_toolset(registry)

    print("\n输入表达式让 calculator 求值；/shout <文本> 调用函数工具；")
    print(f"{SEARCH_PREFIX} <关键词> 搜索（若已配置密钥）；exit / quit 或 Ctrl-D 退出。\n")

    while True:
        line = _read_line("> ")
        if line is None:
            break
        if not line:
            continue
        if line.startswith((SHOUT_PREFIX, SEARCH_PREFIX)):
            _run_lookup(registry, search_available, line)
            continue
        try:
            print(f"结果: {registry.execute_tool('calculator', line)}\n")
        except ToolException as exc:
            print(f"计算失败：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
