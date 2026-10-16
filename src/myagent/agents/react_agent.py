"""ReActAgent：推理—行动循环的框架化实现。

教程定位：第七章 7.4.2 ReActAgent。
与教程的差异（见 docs/deviations.md D8、D9、D25、D27）：

- 执行历史（``Action`` / ``Observation`` 行）随单次调用传递，不再存实例态
  ``self.current_history``，同一 Agent 可并发处理多个问题；
- 步数用尽不再是固定文案，而是返回 :class:`ReActResult`（含 ``stop_reason`` 与逐步轨迹），
  教程式 ``str`` 返回由 :meth:`ReActAgent.arun` 保留；
- 消息用 ``Message`` 对象；工具只捕获 :class:`ToolException` 并作为观察回灌模型。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import AgentException, ToolException
from ..core.llm import HelloAgentsLLM
from ..core.message import Message
from ..tools.registry import ToolRegistry

__all__ = ["DEFAULT_REACT_PROMPT", "ReActAgent", "ReActResult", "ReActStep", "StopReason"]

logger = logging.getLogger(__name__)

DEFAULT_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动：
"""

StopReason = Literal["finished", "max_steps", "no_action"]

_THOUGHT_PATTERN = re.compile(r"Thought:\s*(.*?)(?=\nAction:|$)", re.DOTALL)
_ACTION_PATTERN = re.compile(r"Action:\s*(.*?)$", re.DOTALL)
_ACTION_CALL_PATTERN = re.compile(r"(\w+)\[(.*)\]", re.DOTALL)
_ACTION_INPUT_PATTERN = re.compile(r"\w+\[(.*)\]", re.DOTALL)


@dataclass(frozen=True)
class ReActStep:
    """ReAct 循环中的一步记录。

    Args:
        index: 第几步（从 1 开始）。
        thought: 模型给出的思考；未解析到则为 ``None``。
        action: 模型给出的行动；未解析到则为 ``None``。
        observation: 工具返回或错误说明；``Finish`` 一步为 ``None``。
    """

    index: int
    thought: str | None
    action: str | None
    observation: str | None = None


@dataclass
class ReActResult:
    """一次 ReAct 运行的完整结果。

    Args:
        answer: 最终答案；未能收敛时是说明性的收尾文本。
        stop_reason: 结束原因 —— ``finished``（模型给出 Finish）、``max_steps``
            （步数用尽）、``no_action``（模型输出不符合格式）。
        steps: 逐步轨迹。
        history: 教程口径的执行历史行（``Action`` / ``Observation``）。
    """

    answer: str
    stop_reason: StopReason
    steps: list[ReActStep] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


class ReActAgent(Agent):
    """推理与行动交替进行的 Agent。

    Args:
        name: Agent 名称。
        llm: LLM 客户端。
        tool_registry: 工具注册表（必填，ReAct 没有工具就没有意义）。
        system_prompt: 系统提示词（保留给子类/展示，默认提示词走 ``prompt_template``）。
        config: 配置。
        max_steps: 最大循环步数。
        custom_prompt: 自定义提示词模板，需包含 ``{tools}`` / ``{question}`` / ``{history}``。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        tool_registry: ToolRegistry,
        system_prompt: str | None = None,
        config: Config | None = None,
        max_steps: int = 5,
        custom_prompt: str | None = None,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.prompt_template = custom_prompt or DEFAULT_REACT_PROMPT

    # ------------------------------------------------------------------ 执行

    async def arun(self, input_text: str, **kwargs: Any) -> str:
        """执行一次 ReAct 循环并返回最终答案（教程同签名）。"""
        return (await self.arun_traced(input_text, **kwargs)).answer

    async def arun_traced(self, input_text: str, **kwargs: Any) -> ReActResult:
        """执行一次 ReAct 循环，返回含 ``stop_reason`` 与逐步轨迹的结构化结果。"""
        tools_description = self.tool_registry.get_tools_description()
        history: list[str] = []
        steps: list[ReActStep] = []
        answer = ""
        stop_reason: StopReason = "max_steps"

        for index in range(1, self.max_steps + 1):
            prompt = self._render_prompt(tools_description, input_text, history)
            response = await self.llm.ainvoke([Message(prompt, "user")], **kwargs)
            thought, action = self._parse_output(response)
            logger.debug(
                "agent=%s 第 %d 步 thought=%r action=%r", self.name, index, thought, action
            )

            if action is None:
                steps.append(ReActStep(index, thought, None, "未能解析出 Action"))
                stop_reason = "no_action"
                answer = f"模型第 {index} 步的输出中没有可解析的 Action，无法继续。"
                break

            if action.startswith("Finish"):
                answer = self._parse_action_input(action)
                steps.append(ReActStep(index, thought, action))
                stop_reason = "finished"
                break

            tool_name, tool_input = self._parse_action(action)
            if not tool_name or tool_input is None:
                observation = "无效的 Action 格式，请使用 `工具名[参数]` 或 `Finish[答案]`。"
            else:
                observation = await self._aexecute_action(tool_name, tool_input)
            steps.append(ReActStep(index, thought, action, observation))
            history.append(f"Action: {action}")
            history.append(f"Observation: {observation}")

        if stop_reason == "max_steps":
            answer = f"已达最大步数 {self.max_steps}，仍未得到最终答案。"

        self._record(input_text, answer)
        return ReActResult(answer=answer, stop_reason=stop_reason, steps=steps, history=history)

    async def _aexecute_action(self, tool_name: str, tool_input: str) -> str:
        """执行一次工具调用；工具失败时把结构化错误当作观察交回模型。"""
        logger.info("agent=%s 行动 %s[%s]", self.name, tool_name, tool_input)
        try:
            return await self.tool_registry.aexecute_tool(tool_name, tool_input)
        except ToolException as exc:
            logger.warning("agent=%s 工具 %s 失败: %s", self.name, tool_name, exc)
            return f"工具 {tool_name} 调用失败：{exc}"

    # ------------------------------------------------------------------ 解析

    def _render_prompt(self, tools: str, question: str, history: Sequence[str]) -> str:
        try:
            return self.prompt_template.format(
                tools=tools,
                question=question,
                history="\n".join(history) or "无",
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise AgentException(
                "ReAct 提示词模板渲染失败：模板里出现了未提供的占位符（仅支持 {tools}/{question}/{history}）",
                context={"agent": self.name, "error": type(exc).__name__},
            ) from exc

    @staticmethod
    def _parse_output(text: str) -> tuple[str | None, str | None]:
        """从模型输出中拆出 ``Thought`` 与 ``Action``。"""
        thought_match = _THOUGHT_PATTERN.search(text)
        action_match = _ACTION_PATTERN.search(text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    @staticmethod
    def _parse_action(action: str) -> tuple[str | None, str | None]:
        """把 ``工具名[参数]`` 拆成二元组；格式不符返回 ``(None, None)``。"""
        match = _ACTION_CALL_PATTERN.match(action)
        return (match.group(1), match.group(2)) if match else (None, None)

    @staticmethod
    def _parse_action_input(action: str) -> str:
        """取出 ``Finish[答案]`` 里的答案。"""
        match = _ACTION_INPUT_PATTERN.match(action)
        return match.group(1) if match else ""

    # ------------------------------------------------------------------ 内部

    def _record(self, input_text: str, answer: str) -> None:
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(answer, "assistant"))

    def __repr__(self) -> str:
        return (
            f"ReActAgent(name={self.name!r}, provider={self.llm.provider!r}, "
            f"max_steps={self.max_steps}, tools={self.tool_registry.list_tools()})"
        )
