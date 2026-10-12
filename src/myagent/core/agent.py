"""Agent 抽象基类。

教程定位：第七章 7.3.3 Agent 抽象基类。
与教程的差异（见 docs/deviations.md D4/D11）：``arun`` 是子类必须实现的唯一真实入口，
``run`` 是其同步包装；``max_history_length`` 真正生效（超出后丢弃最早的消息）。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ._sync import run_sync
from .config import Config
from .llm import HelloAgentsLLM
from .message import Message

__all__ = ["Agent"]


class Agent(ABC):
    """所有智能体的抽象基类。

    Args:
        name: Agent 名称，用于日志与展示。
        llm: LLM 客户端实例。
        system_prompt: 系统提示词。
        config: 配置；缺省使用默认配置。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: str | None = None,
        config: Config | None = None,
    ) -> None:
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
        self._logger = logging.getLogger(f"{__name__}.{type(self).__name__}")

    @abstractmethod
    async def arun(self, input_text: str, **kwargs: Any) -> str:
        """异步执行一次任务并返回最终答案。

        Args:
            input_text: 用户输入。
            **kwargs: 子类自定义参数（如 ``max_steps``）。

        Returns:
            最终答案文本。
        """
        raise NotImplementedError

    def run(self, input_text: str, **kwargs: Any) -> str:
        """同步执行一次任务；``arun`` 的包装，禁止在运行中的事件循环内调用。"""
        return run_sync(self.arun(input_text, **kwargs))

    def add_message(self, message: Message) -> None:
        """追加一条消息到历史，并按 ``config.max_history_length`` 裁剪最早的溢出消息。"""
        self._history.append(message)
        limit = self.config.max_history_length
        overflow = len(self._history) - limit
        if overflow > 0:
            del self._history[:overflow]
            self._logger.debug(
                "历史长度超过 max_history_length=%d，丢弃最早 %d 条消息", limit, overflow
            )

    def clear_history(self) -> None:
        """清空历史消息。"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """返回历史消息的副本，修改副本不影响 Agent 内部状态。"""
        return list(self._history)

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, provider={self.llm.provider!r})"
