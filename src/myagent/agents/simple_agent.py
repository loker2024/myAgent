"""SimpleAgent：基础对话范式，可选工具调用。

教程定位：第七章 7.4.1 SimpleAgent。
与教程的差异（见 docs/deviations.md D8、D21、D27）：

- 消息统一为 ``Message`` 对象，不再裸拼 ``{"role", "content"}`` 字典；
- 参数解析不再按工具名硬编码分支，而是先解析 ``key=value`` 形式，且只在该工具的
  参数声明能对上时才当字典用，否则把整串交给注册表按「单必填参数」映射；
- 工具异常只捕获 :class:`ToolException`（结构化错误）并作为观察回灌模型，
  其余异常照常上抛，不会被吞成字符串；
- 流式接口不打印进度（``print`` 归 ``examples/``），也不参与工具循环（与教程一致）。
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

from ..core._sync import ensure_no_running_loop
from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import ToolException
from ..core.llm import HelloAgentsLLM
from ..core.message import Message
from ..tools.base import Tool
from ..tools.registry import ToolRegistry

__all__ = ["DEFAULT_SYSTEM_PROMPT", "SimpleAgent", "ToolCall"]

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "你是一个有用的AI助手。"

_TOOL_CALL_PATTERN = re.compile(r"\[TOOL_CALL:([^:\]]+):([^\]]+)\]")


@dataclass(frozen=True)
class ToolCall:
    """从模型输出里解析出的一次工具调用。

    Args:
        tool_name: 工具名。
        parameters: 参数的原始文本。
        raw: 命中的完整标记，用于从回答里剔除。
    """

    tool_name: str
    parameters: str
    raw: str


class SimpleAgent(Agent):
    """最小可用的对话 Agent，可选挂载工具。

    Args:
        name: Agent 名称。
        llm: LLM 客户端。
        system_prompt: 系统提示词；缺省用 :data:`DEFAULT_SYSTEM_PROMPT`。
        config: 配置。
        tool_registry: 工具注册表；``None`` 表示无工具。
        enable_tool_calling: 是否启用工具调用；没有注册表时强制为 ``False``。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
        tool_registry: ToolRegistry | None = None,
        enable_tool_calling: bool = True,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None

    # ------------------------------------------------------------------ 执行

    async def arun(self, input_text: str, max_tool_iterations: int = 3, **kwargs: Any) -> str:
        """执行一轮对话。

        Args:
            input_text: 用户输入。
            max_tool_iterations: 最多允许多少轮「模型请求工具 → 回灌结果」。
            **kwargs: 透传给 LLM 的调用参数（temperature / max_tokens）。

        Returns:
            最终回答；无论是否调用过工具，都会写入对话历史。
        """
        messages = self._build_messages(input_text)
        if not self.enable_tool_calling:
            answer = await self.llm.ainvoke(messages, **kwargs)
            self._record(input_text, answer)
            return answer
        return await self._arun_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    async def astream(self, input_text: str, **kwargs: Any) -> AsyncIterator[str]:
        """流式对话，逐段产出增量；工具循环不参与流式路径（与教程一致）。"""
        messages = self._build_messages(input_text)
        chunks: list[str] = []
        async for chunk in self.llm.athink(messages, **kwargs):
            chunks.append(chunk)
            yield chunk
        self._record(input_text, "".join(chunks))

    def stream_run(self, input_text: str, **kwargs: Any) -> Iterator[str]:
        """``astream`` 的同步包装（教程同名接口）；不能在运行中的事件循环内调用。"""
        ensure_no_running_loop()
        messages = self._build_messages(input_text)
        chunks: list[str] = []
        for chunk in self.llm.think(messages, **kwargs):
            chunks.append(chunk)
            yield chunk
        self._record(input_text, "".join(chunks))

    # ------------------------------------------------------------------ 工具循环

    async def _arun_with_tools(
        self,
        messages: list[Message],
        input_text: str,
        max_tool_iterations: int,
        **kwargs: Any,
    ) -> str:
        """带工具的多轮循环；工具结果以用户消息回灌，模型据此给出最终回答。"""
        registry = self._require_registry()
        final = ""
        for _ in range(max(1, max_tool_iterations)):
            response = await self.llm.ainvoke(messages, **kwargs)
            calls = self._parse_tool_calls(response)
            if not calls:
                final = response
                break
            cleaned = response
            observations: list[str] = []
            for call in calls:
                cleaned = cleaned.replace(call.raw, "")
                observations.append(await self._aexecute_tool_call(registry, call))
            messages.append(Message(cleaned.strip() or "（正在调用工具）", "assistant"))
            messages.append(
                Message(
                    "工具执行结果：\n"
                    + "\n\n".join(observations)
                    + "\n\n请基于这些结果给出完整的回答。",
                    "user",
                )
            )
        else:
            # 工具轮次用尽仍未收敛：再问一次，避免把中间态当最终答案。
            final = await self.llm.ainvoke(messages, **kwargs)
        self._record(input_text, final)
        return final

    async def _aexecute_tool_call(self, registry: ToolRegistry, call: ToolCall) -> str:
        """执行一次工具调用；工具失败时把结构化错误当作观察交回模型。"""
        logger.info("agent=%s 调用工具 %s(%s)", self.name, call.tool_name, call.parameters)
        try:
            result = await registry.aexecute_tool(
                call.tool_name, self._parse_tool_parameters(registry, call)
            )
        except ToolException as exc:
            logger.warning("agent=%s 工具 %s 失败: %s", self.name, call.tool_name, exc)
            return f"工具 {call.tool_name} 调用失败：{exc}"
        logger.debug("agent=%s 工具 %s 返回 %d 字符", self.name, call.tool_name, len(result))
        return result

    # ------------------------------------------------------------------ 解析

    @staticmethod
    def _parse_tool_calls(text: str) -> list[ToolCall]:
        """解析 ``[TOOL_CALL:name:parameters]`` 标记。"""
        return [
            ToolCall(
                tool_name=match.group(1).strip(),
                parameters=match.group(2).strip(),
                raw=match.group(0),
            )
            for match in _TOOL_CALL_PATTERN.finditer(text)
        ]

    @staticmethod
    def _parse_tool_parameters(registry: ToolRegistry, call: ToolCall) -> str | dict[str, str]:
        """把参数文本折算成注册表能接受的输入。

        只有 ``key=value[,key=value]`` 且每个键都是该工具已声明的参数时，才当作参数字典；
        否则原样返回字符串，由注册表映射到工具的单个必填参数。取代教程按工具名硬编码的分支。
        """
        text = call.parameters
        if "=" not in text:
            return text
        pairs: dict[str, str] = {}
        for segment in text.split(","):
            key, separator, value = segment.partition("=")
            if not separator or not key.strip():
                return text
            pairs[key.strip()] = value.strip()
        tool = registry.get_tool(call.tool_name)
        declared = {spec.name for spec in tool.get_parameters()} if tool is not None else set()
        if pairs and set(pairs) <= declared:
            return pairs
        return text

    # ------------------------------------------------------------------ 提示词

    def _build_messages(self, input_text: str) -> list[Message]:
        """组装「系统提示词 + 历史 + 当前输入」的消息列表。"""
        messages = [Message(self._get_enhanced_system_prompt(), "system")]
        messages.extend(self.get_history())
        messages.append(Message(input_text, "user"))
        return messages

    def _get_enhanced_system_prompt(self) -> str:
        """在系统提示词后追加可用工具与调用格式说明。"""
        base_prompt = self.system_prompt or DEFAULT_SYSTEM_PROMPT
        if not self.enable_tool_calling or self.tool_registry is None:
            return base_prompt
        tools_description = self.tool_registry.get_tools_description()
        if not self.tool_registry.list_tools():
            return base_prompt
        return (
            f"{base_prompt}\n\n## 可用工具\n"
            f"你可以使用以下工具来帮助回答问题：\n{tools_description}\n"
            "\n## 工具调用格式\n"
            "当需要使用工具时，请使用以下格式：\n"
            "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
            "例如：`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:search:query=Python编程,limit=5]`\n\n"
            "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"
        )

    # ------------------------------------------------------------------ 便利方法

    def add_tool(self, tool: Tool) -> None:
        """挂载一个工具；首次调用会自动创建注册表并启用工具调用。"""
        if self.tool_registry is None:
            self.tool_registry = ToolRegistry()
        self.tool_registry.register_tool(tool)
        self.enable_tool_calling = True

    def remove_tool(self, tool_name: str) -> bool:
        """卸载工具；返回是否真的移除了。"""
        if self.tool_registry is None:
            return False
        return self.tool_registry.unregister(tool_name)

    def list_tools(self) -> list[str]:
        """列出可用工具名。"""
        return self.tool_registry.list_tools() if self.tool_registry is not None else []

    def has_tools(self) -> bool:
        """是否处于「有可用工具」状态。"""
        return self.enable_tool_calling and self.tool_registry is not None

    # ------------------------------------------------------------------ 内部

    def _require_registry(self) -> ToolRegistry:
        if self.tool_registry is None:
            raise ToolException("该 Agent 未配置工具注册表", context={"agent": self.name})
        return self.tool_registry

    def _record(self, input_text: str, answer: str) -> None:
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(answer, "assistant"))
