"""工具系统：基类、注册表、内置工具与高级特性。

教程中 ``from hello_agents import ToolRegistry`` / ``from hello_agents.tools import CalculatorTool``
对应本项目 ``from myagent import ToolRegistry`` / ``from myagent.tools import CalculatorTool``。
"""

from .base import FunctionTool, Tool, ToolParameter, ToolType
from .builtin import CalculatorTool, SearchTool
from .registry import ToolInput, ToolRegistry

__all__ = [
    "CalculatorTool",
    "FunctionTool",
    "SearchTool",
    "Tool",
    "ToolInput",
    "ToolParameter",
    "ToolRegistry",
    "ToolType",
]
