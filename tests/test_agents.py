"""四种 Agent 范式的行为测试（离线，走本地 stub LLM 服务）。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from myagent import (
    DEFAULT_SYSTEM_PROMPT,
    AgentException,
    Config,
    HelloAgentsLLM,
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
    Tool,
    ToolException,
    ToolParameter,
    ToolRegistry,
)
from tests.conftest import Scripted, StubLLM


def _client(stub: StubLLM) -> HelloAgentsLLM:
    return HelloAgentsLLM(
        provider="openai", base_url=stub.url, api_key="test-key", model="stub-model", env={}
    )


def _messages(stub: StubLLM, index: int) -> list[dict[str, str]]:
    return stub.requests[index]["payload"]["messages"]


class _Recorder(Tool):
    """记录每次收到的参数，返回可断言的文本。"""

    def __init__(
        self, name: str = "recorder", parameters: list[ToolParameter] | None = None
    ) -> None:
        super().__init__(name, "记录参数", parameters or [ToolParameter("text", "string", "文本")])
        self.seen: list[dict[str, Any]] = []

    def run(self, parameters: Mapping[str, Any]) -> str:
        self.seen.append(dict(parameters))
        return f"记录:{parameters[next(iter(parameters))]}"


class _Failing(Tool):
    def __init__(self) -> None:
        super().__init__("boom", "总是失败", [ToolParameter("text", "string", "文本")])

    def run(self, parameters: Mapping[str, Any]) -> str:
        raise ToolException("后端不可用", context={"reason": "offline"})


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register_tool(tool)
    return registry


# --------------------------------------------------------------------------- SimpleAgent


async def test_simple_agent_plain_chat_records_history(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="在的，你好"))
    agent = SimpleAgent("助手", _client(stub_llm), system_prompt="你是测试助手")

    assert await agent.arun("在吗") == "在的，你好"

    assert _messages(stub_llm, 0) == [
        {"role": "system", "content": "你是测试助手"},
        {"role": "user", "content": "在吗"},
    ]
    assert [(message.role, message.content) for message in agent.get_history()] == [
        ("user", "在吗"),
        ("assistant", "在的，你好"),
    ]


async def test_simple_agent_default_system_prompt_applies_without_tools(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))
    agent = SimpleAgent("助手", _client(stub_llm))

    assert agent.has_tools() is False
    await agent.arun("你好")

    assert _messages(stub_llm, 0)[0]["content"] == DEFAULT_SYSTEM_PROMPT


async def test_simple_agent_enhanced_prompt_lists_tools_and_format(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))
    recorder = _Recorder("search")
    agent = SimpleAgent("助手", _client(stub_llm), tool_registry=_registry(recorder))

    await agent.arun("你好")

    system_prompt = _messages(stub_llm, 0)[0]["content"]
    assert "- search: 记录参数" in system_prompt
    assert "[TOOL_CALL:{tool_name}:{parameters}]" in system_prompt


async def test_simple_agent_executes_tool_and_feeds_result_back(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="我来算一下\n[TOOL_CALL:calculator:2 + 3]"),
            Scripted(content="结果是 5"),
        ]
    )
    calculator = _Recorder("calculator", [ToolParameter("expression", "string", "表达式")])
    agent = SimpleAgent("助手", _client(stub_llm), tool_registry=_registry(calculator))

    assert await agent.arun("算一下 2+3") == "结果是 5"

    assert calculator.seen == [{"expression": "2 + 3"}]
    assert _messages(stub_llm, 1)[2] == {"role": "assistant", "content": "我来算一下"}
    assert _messages(stub_llm, 1)[3]["content"].startswith("工具执行结果：\n记录:2 + 3")


async def test_simple_agent_maps_parameter_pairs_only_when_declared(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="[TOOL_CALL:two:value=7]"),
            Scripted(content="[TOOL_CALL:two:other=7]"),
            Scripted(content="好了"),
        ]
    )
    tool = _Recorder(
        "two",
        [
            ToolParameter("value", "string", "值"),
            ToolParameter("times", "integer", "次数", required=False, default=1),
        ],
    )
    agent = SimpleAgent("助手", _client(stub_llm), tool_registry=_registry(tool))

    assert await agent.arun("试试") == "好了"

    assert tool.seen == [{"value": "7", "times": 1}, {"value": "other=7", "times": 1}]


async def test_simple_agent_reports_tool_failure_then_answers(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [Scripted(content="[TOOL_CALL:boom:查询]"), Scripted(content="工具不可用，我直接回答")]
    )
    agent = SimpleAgent("助手", _client(stub_llm), tool_registry=_registry(_Failing()))

    assert await agent.arun("查询") == "工具不可用，我直接回答"

    observation = _messages(stub_llm, 1)[3]["content"]
    assert "工具 boom 调用失败" in observation
    assert "offline" in observation


async def test_simple_agent_reports_unknown_tool(stub_llm: StubLLM) -> None:
    stub_llm.script.extend([Scripted(content="[TOOL_CALL:ghost:x]"), Scripted(content="换个说法")])
    agent = SimpleAgent("助手", _client(stub_llm), tool_registry=_registry(_Recorder()))

    await agent.arun("试试")

    assert "工具未注册" in _messages(stub_llm, 1)[3]["content"]


async def test_simple_agent_caps_tool_iterations_with_one_final_call(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="[TOOL_CALL:recorder:一]"),
            Scripted(content="[TOOL_CALL:recorder:二]"),
            Scripted(content="收尾回答"),
        ]
    )
    agent = SimpleAgent("助手", _client(stub_llm), tool_registry=_registry(_Recorder()))

    assert await agent.arun("问题", max_tool_iterations=2) == "收尾回答"
    assert len(stub_llm.requests) == 3


async def test_simple_agent_without_tool_calling_ignores_registry(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="原样回答 [TOOL_CALL:recorder:x]"))
    recorder = _Recorder()
    agent = SimpleAgent(
        "助手", _client(stub_llm), tool_registry=_registry(recorder), enable_tool_calling=False
    )

    assert await agent.arun("问题") == "原样回答 [TOOL_CALL:recorder:x]"
    assert recorder.seen == []
    assert len(stub_llm.requests) == 1


async def test_simple_agent_add_and_remove_tool(stub_llm: StubLLM) -> None:
    agent = SimpleAgent("助手", _client(stub_llm))
    recorder = _Recorder("later")

    assert agent.list_tools() == []
    agent.add_tool(recorder)
    assert agent.has_tools() is True
    assert agent.list_tools() == ["later"]
    assert agent.remove_tool("later") is True
    assert agent.list_tools() == []


async def test_simple_agent_astream_yields_chunks_and_records_history(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=True, blocks=["你好", "，世界"]))
    agent = SimpleAgent("助手", _client(stub_llm))

    chunks = [chunk async for chunk in agent.astream("打个招呼")]

    assert chunks == ["你好", "，世界"]
    assert [(message.role, message.content) for message in agent.get_history()] == [
        ("user", "打个招呼"),
        ("assistant", "你好，世界"),
    ]


def test_simple_agent_stream_run_is_incremental_and_sync(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=True, blocks=["A", "B"]))
    agent = SimpleAgent("助手", _client(stub_llm))

    assert list(agent.stream_run("你好")) == ["A", "B"]
    assert agent.get_history()[-1].content == "AB"


def test_simple_agent_run_is_sync_wrapper_around_arun(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="同步回答"))
    agent = SimpleAgent("助手", _client(stub_llm), config=Config(max_history_length=10))

    assert agent.run("问题") == "同步回答"
    assert len(agent.get_history()) == 2


# --------------------------------------------------------------------------- ReActAgent


async def test_react_agent_finishes_with_answer_and_trace(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="Thought: 先查一下\nAction: recorder[abc]"),
            Scripted(content="Thought: 信息够了\nAction: Finish[最终答案]"),
        ]
    )
    recorder = _Recorder()
    agent = ReActAgent("推理助手", _client(stub_llm), _registry(recorder))

    result = await agent.arun_traced("问题是什么")

    assert result.stop_reason == "finished"
    assert result.answer == "最终答案"
    assert recorder.seen == [{"text": "abc"}]
    assert [(step.index, step.thought, step.action, step.observation) for step in result.steps] == [
        (1, "先查一下", "recorder[abc]", "记录:abc"),
        (2, "信息够了", "Finish[最终答案]", None),
    ]
    assert result.history == ["Action: recorder[abc]", "Observation: 记录:abc"]
    prompt = _messages(stub_llm, 1)[0]["content"]
    assert "Observation: 记录:abc" in prompt
    assert "**Question:** 问题是什么" in prompt


async def test_react_agent_arun_returns_plain_answer_and_records_history(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="Thought: 直接答\nAction: Finish[42]"))
    agent = ReActAgent("推理助手", _client(stub_llm), _registry(_Recorder()))

    assert await agent.arun("问题") == "42"
    assert [(message.role, message.content) for message in agent.get_history()] == [
        ("user", "问题"),
        ("assistant", "42"),
    ]


async def test_react_agent_stops_at_max_steps(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="Thought: 继续\nAction: recorder[一]"),
            Scripted(content="Thought: 继续\nAction: recorder[二]"),
        ]
    )
    recorder = _Recorder()
    agent = ReActAgent("推理助手", _client(stub_llm), _registry(recorder), max_steps=2)

    result = await agent.arun_traced("问题")

    assert result.stop_reason == "max_steps"
    assert result.answer == "已达最大步数 2，仍未得到最终答案。"
    assert len(result.steps) == 2
    assert len(stub_llm.requests) == 2


async def test_react_agent_reports_unparsable_action(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="我直接思考了一下，没有按格式输出。"))
    recorder = _Recorder()
    agent = ReActAgent("推理助手", _client(stub_llm), _registry(recorder))

    result = await agent.arun_traced("问题")

    assert result.stop_reason == "no_action"
    assert "Action" in result.answer
    assert recorder.seen == []
    assert len(stub_llm.requests) == 1


async def test_react_agent_feeds_back_invalid_action_format(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="Thought: 随便\nAction: 这句不是合法格式"),
            Scripted(content="Thought: 好的\nAction: Finish[ok]"),
        ]
    )
    agent = ReActAgent("推理助手", _client(stub_llm), _registry(_Recorder()))

    result = await agent.arun_traced("问题")

    assert result.answer == "ok"
    assert result.steps[0].observation is not None
    assert "无效的 Action 格式" in result.steps[0].observation


async def test_react_agent_feeds_back_tool_failure(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="Thought: 查\nAction: boom[查询]"),
            Scripted(content="Thought: 算了\nAction: Finish[换个办法]"),
        ]
    )
    agent = ReActAgent("推理助手", _client(stub_llm), _registry(_Failing()))

    result = await agent.arun_traced("问题")

    assert result.answer == "换个办法"
    assert result.steps[0].observation is not None
    assert "工具 boom 调用失败" in result.steps[0].observation


async def test_react_agent_rejects_prompt_template_with_unknown_placeholder(
    stub_llm: StubLLM,
) -> None:
    agent = ReActAgent(
        "推理助手",
        _client(stub_llm),
        _registry(_Recorder()),
        custom_prompt="问题：{question}，工具：{tools}，历史：{history}，额外：{extra}",
    )

    with pytest.raises(AgentException) as excinfo:
        await agent.arun("问题")

    assert excinfo.value.context["error"] == "KeyError"
    assert excinfo.value.context["agent"] == "推理助手"
    assert stub_llm.requests == []


async def test_react_agent_does_not_leak_execution_history_between_runs(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="Thought: 查\nAction: recorder[一]"),
            Scripted(content="Thought: 结束\nAction: Finish[第一]"),
            Scripted(content="Thought: 结束\nAction: Finish[第二]"),
        ]
    )
    agent = ReActAgent("推理助手", _client(stub_llm), _registry(_Recorder()))

    await agent.arun_traced("第一个问题")
    await agent.arun_traced("第二个问题")

    assert "Observation: 记录:一" in _messages(stub_llm, 1)[0]["content"]
    assert "Observation" not in _messages(stub_llm, 2)[0]["content"]
    assert "无" in _messages(stub_llm, 2)[0]["content"]


# --------------------------------------------------------------------------- ReflectionAgent


async def test_reflection_agent_stops_when_no_improvement_needed(stub_llm: StubLLM) -> None:
    stub_llm.script.extend([Scripted(content="初稿"), Scripted(content="无需改进")])
    agent = ReflectionAgent("反思助手", _client(stub_llm))

    assert await agent.arun("写一段文字") == "初稿"
    assert len(stub_llm.requests) == 2
    assert agent.get_history()[-1].content == "初稿"


async def test_reflection_agent_refines_with_feedback_then_stops(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content="初稿"),
            Scripted(content="结尾太仓促"),
            Scripted(content="改进稿"),
            Scripted(content="无需改进"),
        ]
    )
    agent = ReflectionAgent("反思助手", _client(stub_llm))

    assert await agent.arun("写一段文字") == "改进稿"
    assert "结尾太仓促" in _messages(stub_llm, 2)[0]["content"]
    assert "初稿" in _messages(stub_llm, 2)[0]["content"]


async def test_reflection_agent_honours_max_iterations(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [Scripted(content=f"稿{i}") for i in range(1, 6)] + [Scripted(content="还可以更好")]
    )
    agent = ReflectionAgent("反思助手", _client(stub_llm), max_iterations=2)

    assert await agent.arun("任务") == "稿5"
    assert len(stub_llm.requests) == 5


async def test_reflection_agent_supports_tutorial_style_custom_prompts(stub_llm: StubLLM) -> None:
    stub_llm.script.extend([Scripted(content="代码初稿"), Scripted(content="无需改进")])
    agent = ReflectionAgent(
        "代码助手",
        _client(stub_llm),
        system_prompt="你是 Python 专家",
        custom_prompts={
            "initial": "你是Python专家，请编写函数：{task}",
            "reflect": "请审查代码的算法效率：\n任务：{task}\n代码：{content}",
            "refine": "请根据反馈优化代码：\n任务：{task}\n反馈：{feedback}",
        },
    )

    await agent.arun("找出素数")

    assert [message["role"] for message in _messages(stub_llm, 0)] == ["system", "user"]
    assert "请审查代码的算法效率" in _messages(stub_llm, 1)[1]["content"]
    assert "代码初稿" in _messages(stub_llm, 1)[1]["content"]


async def test_reflection_agent_rejects_empty_task(stub_llm: StubLLM) -> None:
    agent = ReflectionAgent("反思助手", _client(stub_llm))

    with pytest.raises(AgentException):
        await agent.arun("   ")

    assert stub_llm.requests == []


async def test_reflection_agent_reports_missing_prompt_placeholder(stub_llm: StubLLM) -> None:
    agent = ReflectionAgent(
        "反思助手", _client(stub_llm), custom_prompts={"initial": "任务：{task}，额外：{extra}"}
    )

    with pytest.raises(AgentException) as excinfo:
        await agent.arun("任务")

    assert excinfo.value.context["prompt"] == "initial"
    assert stub_llm.requests == []


# --------------------------------------------------------------------------- PlanAndSolveAgent


async def test_plan_solve_agent_parses_fenced_plan_and_runs_steps(stub_llm: StubLLM) -> None:
    stub_llm.script.extend(
        [
            Scripted(content='好的：\n```python\n["算周一的量", "求和"]\n```'),
            Scripted(content="周一 15"),
            Scripted(content="总共 15"),
        ]
    )
    agent = PlanAndSolveAgent("规划助手", _client(stub_llm))

    result = await agent.arun_detailed("苹果问题")

    assert result.plan == ["算周一的量", "求和"]
    assert result.step_results == ["周一 15", "总共 15"]
    assert result.answer == "总共 15"
    assert len(stub_llm.requests) == 3
    second_step_prompt = _messages(stub_llm, 2)[0]["content"]
    assert "当前步骤:\n求和" in second_step_prompt
    assert "步骤 1: 算周一的量\n结果: 周一 15" in second_step_prompt
    assert agent.get_history()[-1].content == "总共 15"


async def test_plan_solve_agent_parses_bare_list(stub_llm: StubLLM) -> None:
    stub_llm.script.extend([Scripted(content="['第一步']"), Scripted(content="结果")])
    agent = PlanAndSolveAgent("规划助手", _client(stub_llm))

    assert await agent.arun("问题") == "结果"


async def test_plan_solve_agent_raises_on_unparsable_plan(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="我先想一想，暂时没有计划。"))
    agent = PlanAndSolveAgent("规划助手", _client(stub_llm))

    with pytest.raises(AgentException) as excinfo:
        await agent.arun("问题")

    assert excinfo.value.context["response"] == "我先想一想，暂时没有计划。"
    assert len(stub_llm.requests) == 1


async def test_plan_solve_agent_rejects_empty_plan(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="```python\n[]\n```"))
    agent = PlanAndSolveAgent("规划助手", _client(stub_llm))

    with pytest.raises(AgentException):
        await agent.arun("问题")


async def test_plan_solve_agent_rejects_empty_question(stub_llm: StubLLM) -> None:
    agent = PlanAndSolveAgent("规划助手", _client(stub_llm))

    with pytest.raises(AgentException):
        await agent.arun("  ")

    assert stub_llm.requests == []


async def test_plan_solve_agent_supports_custom_prompts(stub_llm: StubLLM) -> None:
    stub_llm.script.extend([Scripted(content='["算一下"]'), Scripted(content="42")])
    agent = PlanAndSolveAgent(
        "数学助手",
        _client(stub_llm),
        custom_prompts={
            "planner": "分解数学问题：{question}",
            "executor": "问题：{question} 当前：{current_step} 历史：{history} 计划：{plan}",
        },
    )

    assert await agent.arun("1+1") == "42"
    assert "分解数学问题：1+1" in _messages(stub_llm, 0)[0]["content"]
