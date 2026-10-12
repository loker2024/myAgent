"""工具系统：基类、注册表、内置工具与高级特性。

教程中 ``from hello_agents import ToolRegistry`` / ``from hello_agents.tools import CalculatorTool``
对应本项目 ``from myagent import ToolRegistry`` / ``from myagent.tools import CalculatorTool``。
"""

from .async_executor import AsyncToolExecutor
from .base import FunctionTool, Tool, ToolParameter, ToolType
from .builtin import CalculatorTool, SearchTool
from .chain import ToolChain, ToolChainManager, ToolStep
from .registry import ToolInput, ToolRegistry

__all__ = [
    "AsyncToolExecutor",
    "CalculatorTool",
    "FunctionTool",
    "SearchTool",
    "Tool",
    "ToolChain",
    "ToolChainManager",
    "ToolInput",
    "ToolParameter",
    "ToolRegistry",
    "ToolStep",
    "ToolType",
]
