"""工具注册表：工具的唯一对外调用路径。

教程定位：第七章 7.5.1 的 ``ToolRegistry``（注册、发现、执行）。
与教程的差异（见 docs/deviations.md D6、D22）：

- 教程里同一工具存在两条调用路径（``registry.execute_tool`` 与 ``tool.run`` 混用），
  此处只保留 :meth:`ToolRegistry.execute_tool` / :meth:`ToolRegistry.aexecute_tool` 两条
  （同步、异步各一，内部都走 :meth:`Tool.run`），参数解析与异常包装只有一份；
- 输入统一为「字符串或参数字典」：单参数工具接受裸字符串，多参数工具要求字典，
  取代教程硬编码在 Agent 里的按工具名分支；
- 教程的 ``register_function`` 参数顺序是 ``(name, description, func)``，此处保持一致，
  并可选 ``parameter_name`` 精确指定暴露的参数。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from ..core.exceptions import ToolException
from .base import FunctionTool, Tool

__all__ = ["ToolRegistry"]

logger = logging.getLogger(__name__)

_EMPTY_DESCRIPTION = "暂无可用工具"

ToolInput = str | Mapping[str, Any]


class ToolRegistry:
    """工具的注册、发现与执行中枢。

    注册方式有两种，但执行路径完全一致（都由 :meth:`execute_tool` 调度）：

    - :meth:`register_tool`：注册 :class:`Tool` 子类实例，适合带状态、带参数声明的工具；
    - :meth:`register_function`：把一个普通函数包成 :class:`FunctionTool`，快速集成。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------ 注册

    def register_tool(self, tool: Tool) -> None:
        """注册工具对象；同名工具会被覆盖并记一条警告。"""
        if tool.name in self._tools:
            logger.warning("工具 %s 已存在，将被覆盖", tool.name)
        self._tools[tool.name] = tool
        logger.debug("工具 %s 已注册", tool.name)

    def register_function(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        *,
        parameter_name: str | None = None,
    ) -> Tool:
        """把普通函数注册为工具。

        Args:
            name: 工具名。
            description: 工具描述。
            func: 工具函数，参数声明由 ``inspect.signature`` 推断。
            parameter_name: 只暴露函数某一个参数时显式指定。

        Returns:
            适配后的 :class:`FunctionTool`，便于调用方继续使用。
        """
        tool = FunctionTool(name, description, func, parameter_name=parameter_name)
        self.register_tool(tool)
        return tool

    def unregister(self, name: str) -> bool:
        """注销工具；返回是否真的移除了工具。"""
        return self._tools.pop(name, None) is not None

    # ------------------------------------------------------------------ 发现

    def get_tool(self, name: str) -> Tool | None:
        """按名取工具；不存在返回 ``None``。"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有工具名（注册顺序）。"""
        return list(self._tools)

    def get_tools_description(self) -> str:
        """生成可直接放进提示词的工具清单；无工具时返回「暂无可用工具」。"""
        lines = [f"- {tool.name}: {tool.description}" for tool in self._tools.values()]
        return "\n".join(lines) if lines else _EMPTY_DESCRIPTION

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """导出全部工具的 OpenAI function calling schema。"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    # ------------------------------------------------------------------ 执行

    def execute_tool(self, name: str, input_data: ToolInput) -> str:
        """同步执行工具。

        Args:
            name: 工具名。
            input_data: 单参数工具的裸字符串，或多参数工具的参数字典。

        Returns:
            工具输出文本。

        Raises:
            ToolException: 工具不存在、参数不合法，或工具自身执行失败时。
        """
        tool = self._require(name)
        parameters = self._prepare(tool, input_data)
        return self._dispatch(tool, parameters, run=tool.run)

    async def aexecute_tool(self, name: str, input_data: ToolInput) -> str:
        """异步执行工具（工具自带异步实现，或在线程池里跑同步实现）。"""
        tool = self._require(name)
        parameters = self._prepare(tool, input_data)
        try:
            return await tool.arun(parameters)
        except ToolException as exc:
            self._enrich(exc, tool, parameters)
            raise
        except Exception as exc:
            raise self._wrap(exc, tool, parameters) from exc

    def _dispatch(
        self, tool: Tool, parameters: Mapping[str, Any], *, run: Callable[[Mapping[str, Any]], str]
    ) -> str:
        """执行并统一异常语义：工具自己的 ToolException 原样上抛，其余包装成 ToolException。"""
        try:
            return run(parameters)
        except ToolException as exc:
            self._enrich(exc, tool, parameters)
            raise
        except Exception as exc:
            raise self._wrap(exc, tool, parameters) from exc

    @staticmethod
    def _wrap(exc: Exception, tool: Tool, parameters: Mapping[str, Any]) -> ToolException:
        return ToolException(
            "工具执行失败",
            context={
                "tool": tool.name,
                "parameters": dict(parameters),
                "error": type(exc).__name__,
                "detail": str(exc),
            },
            retryable=False,
        )

    # ------------------------------------------------------------------ 内部

    def _require(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolException(
                "工具未注册",
                context={"tool": name, "available": self.list_tools()},
            )
        return tool

    def _prepare(self, tool: Tool, input_data: ToolInput) -> dict[str, Any]:
        """把字符串或字典输入折算成参数字典，并交由工具校验。"""
        if isinstance(input_data, Mapping):
            raw: dict[str, Any] = dict(input_data)
        elif isinstance(input_data, str):
            declared = tool.get_parameters()
            required = [item for item in declared if item.required]
            if len(required) == 1:
                raw = {required[0].name: input_data}
            elif not declared:
                raw = {}
            elif len(declared) == 1:
                raw = {declared[0].name: input_data}
            else:
                raise ToolException(
                    "多必填参数的工具需要传入参数字典",
                    context={"tool": tool.name, "declared": [item.name for item in required]},
                )
        else:
            raise ToolException(
                "不支持的输入类型：只接受字符串或参数字典",
                context={"tool": tool.name, "type": type(input_data).__name__},
            )
        return tool.validate(raw)

    def _enrich(self, exc: ToolException, tool: Tool, parameters: Mapping[str, Any]) -> None:
        """给工具抛出的异常补上「哪个工具、什么参数」的定位信息。"""
        exc.context.setdefault("tool", tool.name)
        exc.context.setdefault("parameters", dict(parameters))
        exc.args = (exc._render(),)

    # ------------------------------------------------------------------ 容器协议

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())
