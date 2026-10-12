"""myagent：跟练 Hello-Agents 教程（第 7 章起）并以现代写法重构的智能体框架。

教程中 ``from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message, ToolRegistry``
对应本项目 ``from myagent import HelloAgentsLLM, Config, Message``；``SimpleAgent`` 等
agent 实现与 ``ToolRegistry`` 分别在 P3、P4 落地。
"""

from .core import (
    PROVIDERS,
    Agent,
    AgentException,
    Config,
    ConfigException,
    HelloAgentsException,
    HelloAgentsLLM,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMException,
    LLMRateLimitError,
    LLMServerError,
    Message,
    MessageRole,
    ProviderSpec,
    ToolException,
    default_model_for,
    detect_provider,
    resolve_credentials,
    spec_for,
)

__all__ = [
    "PROVIDERS",
    "Agent",
    "AgentException",
    "Config",
    "ConfigException",
    "HelloAgentsException",
    "HelloAgentsLLM",
    "LLMAuthenticationError",
    "LLMConnectionError",
    "LLMException",
    "LLMRateLimitError",
    "LLMServerError",
    "Message",
    "MessageRole",
    "ProviderSpec",
    "ToolException",
    "default_model_for",
    "detect_provider",
    "resolve_credentials",
    "spec_for",
]
