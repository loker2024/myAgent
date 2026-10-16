"""框架内统一的消息格式。

教程定位：第七章 7.3.1 Message 类。
与教程的差异（见 docs/deviations.md D2）：用 ``@dataclass`` 取代 ``pydantic.BaseModel``，
去掉 ``timestamp: datetime = None`` 的类型/默认值矛盾，时间戳带 UTC 时区。
构造形式与教程一致：``Message("你好", "user")``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

__all__ = ["Message", "MessageRole"]

MessageRole = Literal["user", "assistant", "system", "tool"]


@dataclass
class Message:
    """一条对话消息。

    Args:
        content: 消息正文。
        role: 角色，限定为 user / assistant / system / tool。
        timestamp: 创建时间，默认当前 UTC 时间。
        metadata: 附加信息（工具名、耗时、usage 等），默认空字典。
    """

    content: str
    role: MessageRole
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        """转换为 OpenAI 兼容的 ``{"role", "content"}`` 字典。"""
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
