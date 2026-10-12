"""ReflectionAgent：生成—反思—改进循环的框架化实现。

教程定位：第七章 7.4.3 ReflectionAgent（只给出提示词，实现由练习者补全）。
与教程的差异（见 docs/deviations.md D8、D26、D27）：

- 提示词渲染统一走 :meth:`ReflectionAgent._render`：缺占位符时抛
  :class:`AgentException`（带 agent 名与缺失键），不再静默产出残缺提示词；
- 三个默认模板共用一套占位符（``task`` / ``content`` / ``last_attempt`` / ``feedback``），
  因此教程测试里的 ``{content}`` 风格与正文模板的 ``{last_attempt}`` 风格都能直接用；
- 消息用 ``Message`` 对象，中间产物不落历史（只有任务与最终回答进历史）。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from ..core.agent import Agent
from ..core.config import Config
from ..core.exceptions import AgentException
from ..core.llm import HelloAgentsLLM
from ..core.message import Message

__all__ = ["DEFAULT_PROMPTS", "ReflectionAgent"]

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS: Final[dict[str, str]] = {
    "initial": """
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
""",
    "reflect": """
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
""",
    "refine": """
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
""",
}

_NO_IMPROVEMENT_MARKERS: Final[tuple[str, ...]] = ("无需改进", "no need for improvement")


class ReflectionAgent(Agent):
    """先给出初稿，再反复「自我审查 → 按反馈改进」的 Agent。

    Args:
        name: Agent 名称。
        llm: LLM 客户端。
        system_prompt: 系统提示词；给出后会作为 system 消息参与每一轮调用。
        config: 配置。
        max_iterations: 最多迭代几轮反思—改进。
        custom_prompts: 覆盖默认提示词，键取 ``initial`` / ``reflect`` / ``refine``；
            未给出的键沿用默认模板。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
        max_iterations: int = 3,
        custom_prompts: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.prompts: dict[str, str] = {**DEFAULT_PROMPTS, **(custom_prompts or {})}
        missing = sorted({"initial", "reflect", "refine"} - set(self.prompts))
        if missing:
            raise AgentException(
                "反思 Agent 缺少必需的提示词",
                context={"agent": name, "missing": missing},
            )

    # ------------------------------------------------------------------ 执行

    async def arun(self, input_text: str, **kwargs: Any) -> str:
        """执行一次反思循环并返回最终回答（``input_text`` 即教程中的 ``task``）。"""
        task = input_text
        if not task.strip():
            raise AgentException("反思任务不能为空", context={"agent": self.name})

        answer = await self._ask(self._render("initial", task=task), **kwargs)
        for iteration in range(1, self.max_iterations + 1):
            feedback = await self._ask(
                self._render("reflect", task=task, content=answer, last_attempt=answer), **kwargs
            )
            if any(
                marker in feedback or marker in feedback.lower()
                for marker in _NO_IMPROVEMENT_MARKERS
            ):
                logger.info("agent=%s 第 %d 轮反思判定无需改进，提前结束", self.name, iteration)
                break
            answer = await self._ask(
                self._render(
                    "refine",
                    task=task,
                    content=answer,
                    last_attempt=answer,
                    feedback=feedback,
                ),
                **kwargs,
            )
            logger.debug(
                "agent=%s 第 %d 轮改进完成，回答 %d 字符", self.name, iteration, len(answer)
            )

        self.add_message(Message(task, "user"))
        self.add_message(Message(answer, "assistant"))
        return answer

    # ------------------------------------------------------------------ 内部

    async def _ask(self, prompt: str, **kwargs: Any) -> str:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(self.system_prompt, "system"))
        messages.append(Message(prompt, "user"))
        return await self.llm.ainvoke(messages, **kwargs)

    def _render(self, key: str, **values: str) -> str:
        """渲染指定提示词；模板缺少所需占位符时抛 :class:`AgentException`。"""
        template = self.prompts[key]
        try:
            return template.format(**values)
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
        return (
            f"ReflectionAgent(name={self.name!r}, provider={self.llm.provider!r}, "
            f"max_iterations={self.max_iterations})"
        )
