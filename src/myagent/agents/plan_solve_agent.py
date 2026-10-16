"""PlanAndSolveAgent：先规划、再逐步执行的框架化实现。

教程定位：第七章 7.4.4 PlanAndSolveAgent。
与教程的差异（见 docs/deviations.md D8、D26）：

- 规划器输出解析失败时抛 :class:`AgentException`（带原始响应），不再静默返回 ``None``
  让调用方拿到「没有答案」的空结果；
- 计划解析支持 `````python`` 代码块、裸列表字面量、以及正文夹杂列表三种形态，
  解析结果一律规整成 ``list[str]``；
- 结构化结果 :class:`PlanAndSolveResult` 暴露计划与各步输出，教程式 ``str`` 返回由
  :meth:`PlanAndSolveAgent.arun` 保留。
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import AgentException
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

__all__ = [
    "DEFAULT_EXECUTOR_PROMPT",
    "DEFAULT_PLANNER_PROMPT",
    "PlanAndSolveAgent",
    "PlanAndSolveResult",
]

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_PROMPT = """你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

DEFAULT_EXECUTOR_PROMPT = """你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

_FENCED_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_REQUIRED_PROMPTS: Final[tuple[str, ...]] = ("planner", "executor")


@dataclass
class PlanAndSolveResult:
    """一次规划—执行运行的完整结果。

    Args:
        question: 原始问题。
        plan: 规划器给出的步骤列表。
        step_results: 与 ``plan`` 一一对应的各步输出。
        answer: 最后一步的输出，即最终答案。
    """

    question: str
    plan: list[str]
    step_results: list[str] = field(default_factory=list)
    answer: str = ""


class PlanAndSolveAgent(Agent):
    """先让模型给出计划，再让模型逐步执行的 Agent。

    Args:
        name: Agent 名称。
        llm: LLM 客户端。
        system_prompt: 系统提示词；给出后会作为 system 消息参与每一次调用。
        config: 配置。
        custom_prompts: 覆盖默认提示词，键取 ``planner`` / ``executor``；未给出的沿用默认模板。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
        custom_prompts: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.prompts: dict[str, str] = {
            "planner": DEFAULT_PLANNER_PROMPT,
            "executor": DEFAULT_EXECUTOR_PROMPT,
            **(custom_prompts or {}),
        }
        missing = sorted(set(_REQUIRED_PROMPTS) - set(self.prompts))
        if missing:
            raise AgentException(
                "规划执行 Agent 缺少必需的提示词",
                context={"agent": name, "missing": missing},
            )

    # ------------------------------------------------------------------ 执行

    async def arun(self, input_text: str, **kwargs: Any) -> str:
        """执行一次「规划 → 逐步执行」并返回最终答案（``input_text`` 即教程中的 ``question``）。"""
        return (await self.arun_detailed(input_text, **kwargs)).answer

    async def arun_detailed(self, question: str, **kwargs: Any) -> PlanAndSolveResult:
        """执行一次「规划 → 逐步执行」，返回含计划与各步输出的结构化结果。"""
        if not question.strip():
            raise AgentException("待规划的问题不能为空", context={"agent": self.name})

        plan = self._parse_plan(
            await self._ask(self._render("planner", question=question), **kwargs)
        )
        logger.info("agent=%s 生成计划 %d 步: %s", self.name, len(plan), plan)

        history = ""
        step_results: list[str] = []
        for index, step in enumerate(plan, 1):
            prompt = self._render(
                "executor",
                question=question,
                plan=plan,
                history=history or "无",
                current_step=step,
            )
            output = await self._ask(prompt, **kwargs)
            step_results.append(output)
            history += f"步骤 {index}: {step}\n结果: {output}\n\n"
            logger.debug("agent=%s 第 %d/%d 步完成", self.name, index, len(plan))

        answer = step_results[-1]
        self.add_message(Message(question, "user"))
        self.add_message(Message(answer, "assistant"))
        return PlanAndSolveResult(
            question=question, plan=plan, step_results=step_results, answer=answer
        )

    # ------------------------------------------------------------------ 解析

    def _parse_plan(self, response: str) -> list[str]:
        """把规划器输出解析成步骤列表。

        Raises:
            AgentException: 三种形态都解析不出非空字符串列表时。
        """
        candidates: list[str] = [match.group(1) for match in _FENCED_BLOCK.finditer(response)]
        candidates.append(response)
        for candidate in candidates:
            steps = self._literal_list(candidate)
            if steps:
                return steps
        raise AgentException(
            "无法把规划器输出解析为步骤列表",
            context={"agent": self.name, "response": response},
        )

    @staticmethod
    def _literal_list(text: str) -> list[str]:
        """尽力把一段文本解析成 ``list[str]``；失败返回空列表。"""
        stripped = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        snippets = [stripped]
        if start != -1 and end > start:
            snippets.append(text[start : end + 1])
        for snippet in snippets:
            try:
                value = ast.literal_eval(snippet)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
            if isinstance(value, list) and value:
                return [item if isinstance(item, str) else str(item) for item in value]
        return []

    # ------------------------------------------------------------------ 内部

    async def _ask(self, prompt: str, **kwargs: Any) -> str:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(self.system_prompt, "system"))
        messages.append(Message(prompt, "user"))
        return await self.llm.ainvoke(messages, **kwargs)

    def _render(self, key: str, **values: Any) -> str:
        """渲染指定提示词；模板缺少所需占位符时抛 :class:`AgentException`。"""
        try:
            return self.prompts[key].format(**values)
        except (KeyError, IndexError, ValueError) as exc:
            raise AgentException(
                "提示词模板渲染失败：模板里的占位符没有对应取值",
                context={
                    "agent": self.name,
                    "prompt": key,
                    "error": type(exc).__name__,
                    "available": sorted(values),
                },
            ) from exc

    def __repr__(self) -> str:
        return f"PlanAndSolveAgent(name={self.name!r}, provider={self.llm.provider!r})"
