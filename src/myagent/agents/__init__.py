"""四种 Agent 范式：Simple / ReAct / Reflection / Plan-and-Solve。

教程中 ``from hello_agents import SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent``
对应本项目 ``from myagent.agents import SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent``；
顶层 ``myagent`` 也已转出这些名字。
"""

from .plan_solve_agent import (
    DEFAULT_EXECUTOR_PROMPT,
    DEFAULT_PLANNER_PROMPT,
    PlanAndSolveAgent,
    PlanAndSolveResult,
)
from .react_agent import DEFAULT_REACT_PROMPT, ReActAgent, ReActResult, ReActStep, StopReason
from .reflection_agent import DEFAULT_PROMPTS, ReflectionAgent
from .simple_agent import DEFAULT_SYSTEM_PROMPT, SimpleAgent, ToolCall

__all__ = [
    "DEFAULT_EXECUTOR_PROMPT",
    "DEFAULT_PLANNER_PROMPT",
    "DEFAULT_PROMPTS",
    "DEFAULT_REACT_PROMPT",
    "DEFAULT_SYSTEM_PROMPT",
    "PlanAndSolveAgent",
    "PlanAndSolveResult",
    "ReActAgent",
    "ReActResult",
    "ReActStep",
    "ReflectionAgent",
    "SimpleAgent",
    "StopReason",
    "ToolCall",
]
