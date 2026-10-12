"""Agent 基类与同步包装的行为测试。"""

from __future__ import annotations

from typing import Any, cast

import pytest

from myagent import Agent, Config, HelloAgentsLLM, Message
from tests.conftest import Scripted, StubLLM


class _EchoAgent(Agent):
    """测试用最小实现：把整段历史交给 LLM 并记录回答。"""

    async def arun(self, input_text: str, **kwargs: Any) -> str:
        self.add_message(Message(input_text, "user"))
        answer = await self.llm.ainvoke([message.to_dict() for message in self.get_history()])
        self.add_message(Message(answer, "assistant"))
        return answer


def _client(stub: StubLLM) -> HelloAgentsLLM:
    return HelloAgentsLLM(
        provider="openai",
        base_url=stub.url,
        api_key="test-key",
        model="stub-model",
        env={},
    )


def test_run_executes_arun_and_records_history(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="回答一"))
    agent = _EchoAgent("echo", _client(stub_llm))

    assert agent.run("问题一") == "回答一"

    history = agent.get_history()
    assert [(message.role, message.content) for message in history] == [
        ("user", "问题一"),
        ("assistant", "回答一"),
    ]
    assert stub_llm.requests[0]["payload"]["messages"] == [{"role": "user", "content": "问题一"}]


def test_history_is_truncated_to_max_history_length(stub_llm: StubLLM) -> None:
    stub_llm.script.extend([Scripted(content="回答一"), Scripted(content="回答二")])
    agent = _EchoAgent("echo", _client(stub_llm), config=Config(max_history_length=3))

    agent.run("问题一")
    agent.run("问题二")

    history = agent.get_history()
    assert [(message.role, message.content) for message in history] == [
        ("assistant", "回答一"),
        ("user", "问题二"),
        ("assistant", "回答二"),
    ]


def test_get_history_returns_a_copy(stub_llm: StubLLM) -> None:
    agent = _EchoAgent("echo", _client(stub_llm))
    agent.add_message(Message("hi", "user"))

    agent.get_history().clear()

    assert len(agent.get_history()) == 1


def test_clear_history_empties_records(stub_llm: StubLLM) -> None:
    agent = _EchoAgent("echo", _client(stub_llm))
    agent.add_message(Message("hi", "user"))

    agent.clear_history()

    assert agent.get_history() == []


def test_str_and_repr_expose_provider(stub_llm: StubLLM) -> None:
    agent = _EchoAgent("echo", _client(stub_llm))

    assert "echo" in str(agent)
    assert "openai" in str(agent)
    assert "_EchoAgent" in repr(agent)


def test_base_agent_cannot_be_instantiated(stub_llm: StubLLM) -> None:
    with pytest.raises(TypeError):
        cast(Any, Agent)("base", _client(stub_llm))


async def test_run_refuses_to_execute_inside_event_loop(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))
    agent = _EchoAgent("echo", _client(stub_llm))

    with pytest.raises(RuntimeError):
        agent.run("问题")

    assert stub_llm.requests == []


async def test_arun_is_usable_directly_from_async_context(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="异步回答"))
    agent = _EchoAgent("echo", _client(stub_llm))

    assert await agent.arun("问题") == "异步回答"
    assert len(agent.get_history()) == 2
