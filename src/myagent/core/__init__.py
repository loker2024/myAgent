"""框架核心组件：LLM 客户端、消息、配置、Agent 基类、异常体系。"""

from .agent import Agent
from .config import Config, env_bool, env_float, env_int
from .exceptions import (
    AgentException,
    ConfigException,
    HelloAgentsException,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMException,
    LLMRateLimitError,
    LLMServerError,
    ToolException,
)
from .llm import (
    PROVIDERS,
    HelloAgentsLLM,
    ProviderSpec,
    default_model_for,
    detect_provider,
    resolve_credentials,
    spec_for,
)
from .message import Message, MessageRole

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
    "env_bool",
    "env_float",
    "env_int",
    "resolve_credentials",
    "spec_for",
]
