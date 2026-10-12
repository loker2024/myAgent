"""统一的异常体系。

教程定位：第七章 7.3 未单列异常文件，参考实现见 ``hello_agents/core/exceptions.py``。
与教程的差异（见 docs/deviations.md D10）：教程里是 5 个空 ``pass`` 类体，此处每个异常都携带
定位上下文（provider / tool / model / status），并用 ``retryable`` 区分可重试与不可重试。
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "AgentException",
    "ConfigException",
    "HelloAgentsException",
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMException",
    "LLMRateLimitError",
    "LLMServerError",
    "ToolException",
]


class HelloAgentsException(Exception):
    """框架所有异常的基类。

    Args:
        message: 人类可读的错误描述。
        context: 定位用的键值对，例如 ``{"provider": "ollama", "model": "llama3"}``。
        retryable: 显式覆盖类级别的可重试标记。
    """

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, object] | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.message = message
        self.context: dict[str, object] = dict(context or {})
        if retryable is not None:
            self.retryable = retryable
        super().__init__(self._render())

    def _render(self) -> str:
        if not self.context:
            return self.message
        detail = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({detail})"


class ConfigException(HelloAgentsException):
    """配置错误：参数非法、缺少凭据、provider 未知。"""


class AgentException(HelloAgentsException):
    """Agent 运行期错误。"""


class ToolException(HelloAgentsException):
    """工具执行错误。"""


class LLMException(HelloAgentsException):
    """LLM 调用错误（默认不可重试，重试由子类声明）。"""


class LLMAuthenticationError(LLMException):
    """凭据被拒绝（HTTP 401/403），重试无意义。"""

    retryable = False


class LLMRateLimitError(LLMException):
    """触发限流（HTTP 429），退避后可重试。"""

    retryable = True


class LLMServerError(LLMException):
    """服务端错误（HTTP 5xx），可重试。"""

    retryable = True


class LLMConnectionError(LLMException):
    """网络层失败（连接、DNS、超时），可重试。"""

    retryable = True
