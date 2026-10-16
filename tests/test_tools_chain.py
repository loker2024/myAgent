"""工具链（ToolChain / ToolChainManager）的行为测试，全部离线。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from myagent.core.exceptions import ToolException
from myagent.tools.chain import ToolChain, ToolChainManager, ToolStep
from myagent.tools.registry import ToolRegistry


def _registry() -> ToolRegistry:
    """两个单参数假工具：转大写与加括号。"""
    registry = ToolRegistry()
    registry.register_function("upper", "转大写", lambda text: text.upper())
    registry.register_function("wrap", "加括号", lambda text: f"[{text}]")
    return registry


def _two_step_chain() -> ToolChain:
    chain = ToolChain("demo", "两步演示")
    chain.add_step("upper", "{input}", "first")
    chain.add_step("wrap", "前一步={first}")
    return chain


def test_step_can_reference_previous_output_and_initial_input() -> None:
    registry = _registry()
    chain = ToolChain("demo", "拼接初始输入与上一步结果")
    chain.add_step("upper", "{input}")
    chain.add_step("wrap", "{input}-{step_0_result}")

    assert chain.execute(registry, "hello") == "[hello-HELLO]"


def test_default_output_key_follows_step_index() -> None:
    chain = ToolChain("demo", "")
    chain.add_step("upper", "{input}")
    chain.add_step("wrap", "{step_0_result}")

    assert chain.steps[0].output_key == "step_0_result"
    assert chain.steps[1].output_key == "step_1_result"


def test_explicit_output_key_is_preserved() -> None:
    chain = ToolChain("demo", "")
    step = chain.add_step("upper", "{input}", "custom")

    assert step.output_key == "custom"
    assert chain.steps[0] == ToolStep("upper", "{input}", "custom")


def test_steps_is_an_immutable_snapshot() -> None:
    chain = ToolChain("demo", "")
    chain.add_step("upper", "{input}")
    snapshot = chain.steps

    chain.add_step("wrap", "{input}")

    assert isinstance(snapshot, tuple)
    assert len(snapshot) == 1  # 快照不随后续 add_step 变化
    assert len(chain.steps) == 2


def test_missing_template_variable_raises_with_context() -> None:
    registry = _registry()
    chain = ToolChain("demo", "")
    chain.add_step("upper", "{nope}")

    with pytest.raises(ToolException) as excinfo:
        chain.execute(registry, "x")

    assert excinfo.value.context["chain"] == "demo"
    assert excinfo.value.context["step"] == 1
    assert excinfo.value.context["missing"] == "nope"


def test_unknown_tool_error_bubbles_up() -> None:
    registry = _registry()
    chain = ToolChain("demo", "")
    chain.add_step("ghost", "{input}")

    with pytest.raises(ToolException) as excinfo:
        chain.execute(registry, "x")

    assert excinfo.value.context["tool"] == "ghost"


def test_empty_chain_raises_with_context() -> None:
    chain = ToolChain("empty", "")

    with pytest.raises(ToolException) as excinfo:
        chain.execute(_registry(), "x")

    assert excinfo.value.context["chain"] == "empty"


def test_execute_does_not_mutate_caller_context() -> None:
    registry = _registry()
    chain = ToolChain("demo", "")
    chain.add_step("wrap", "hi {name}")
    context: dict[str, Any] = {"name": "world"}

    assert chain.execute(registry, "x", context) == "[hi world]"
    assert context == {"name": "world"}  # input 与各步结果都写在副本里


def test_sync_and_async_execution_agree() -> None:
    registry = _registry()
    chain = _two_step_chain()

    assert chain.execute(registry, "abc") == "[前一步=ABC]"
    assert chain.execute(registry, "abc") == "[前一步=ABC]"  # 同一链可重复执行


async def test_aexecute_matches_execute() -> None:
    registry = _registry()
    chain = _two_step_chain()
    context: Mapping[str, Any] = {}

    assert await chain.aexecute(registry, "abc", context) == chain.execute(registry, "abc")


async def test_aexecute_unknown_tool_error_bubbles_up() -> None:
    registry = _registry()
    chain = ToolChain("demo", "")
    chain.add_step("ghost", "{input}")

    with pytest.raises(ToolException) as excinfo:
        await chain.aexecute(registry, "x")

    assert excinfo.value.context["tool"] == "ghost"


def test_manager_register_list_get_unregister() -> None:
    manager = ToolChainManager(_registry())
    chain = _two_step_chain()

    assert manager.get_chain("demo") is None
    manager.register_chain(chain)

    assert manager.list_chains() == ["demo"]
    assert manager.get_chain("demo") is chain
    assert manager.unregister_chain("demo") is True
    assert manager.unregister_chain("demo") is False  # 再次注销无效果
    assert manager.list_chains() == []


def test_manager_execute_chain_missing_name_raises_with_context() -> None:
    manager = ToolChainManager(_registry())
    manager.register_chain(_two_step_chain())

    with pytest.raises(ToolException) as excinfo:
        manager.execute_chain("nope", "x")

    assert excinfo.value.context["chain"] == "nope"
    assert excinfo.value.context["available"] == ["demo"]


def test_manager_execute_chain_runs_the_chain() -> None:
    manager = ToolChainManager(_registry())
    manager.register_chain(_two_step_chain())

    assert manager.execute_chain("demo", "abc") == "[前一步=ABC]"


async def test_manager_aexecute_chain_runs_the_chain() -> None:
    manager = ToolChainManager(_registry())
    manager.register_chain(_two_step_chain())

    with pytest.raises(ToolException):
        await manager.aexecute_chain("nope", "x")

    assert await manager.aexecute_chain("demo", "abc") == "[前一步=ABC]"
