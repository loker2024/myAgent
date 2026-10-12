"""ToolRegistry / Tool / FunctionTool 的行为测试（离线）。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, cast

import pytest

from myagent.core.exceptions import ToolException
from myagent.tools.base import Tool, ToolParameter
from myagent.tools.registry import ToolRegistry


class _Recorder(Tool):
    """记录收到的参数，并回报一段可断言的文本。"""

    def __init__(
        self, name: str = "recorder", parameters: list[ToolParameter] | None = None
    ) -> None:
        super().__init__(name, "记录参数", parameters or [ToolParameter("text", "string", "文本")])
        self.seen: list[dict[str, Any]] = []

    def run(self, parameters: Mapping[str, Any]) -> str:
        self.seen.append(dict(parameters))
        return f"记录:{parameters[next(iter(parameters))]}"


class _AsyncOnly(Tool):
    """只实现异步路径的工具：``run`` 被调用即视为失败。"""

    def __init__(self) -> None:
        super().__init__("async_only", "只有异步实现", [ToolParameter("text", "string", "文本")])

    def run(self, parameters: Mapping[str, Any]) -> str:
        raise AssertionError("异步执行不应走到同步实现")

    async def arun(self, parameters: Mapping[str, Any]) -> str:
        await asyncio.sleep(0)
        return f"异步:{parameters['text']}"


def test_register_function_infers_parameters_from_signature() -> None:
    def greet(name: str, times: int = 1) -> str:
        return f"你好 {name} " * times

    registry = ToolRegistry()
    registry.register_function("greet", "打招呼", greet)

    tool = registry.get_tool("greet")
    assert tool is not None
    assert [(spec.name, spec.type, spec.required) for spec in tool.get_parameters()] == [
        ("name", "string", True),
        ("times", "integer", False),
    ]
    assert registry.execute_tool("greet", "小明") == "你好 小明 "
    assert registry.execute_tool("greet", {"name": "小明", "times": "2"}) == "你好 小明 你好 小明 "


def test_register_function_rejects_functions_without_positional_parameters() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolException) as excinfo:
        registry.register_function("now", "当前时间", lambda: "now")

    assert excinfo.value.context["function"] == "now"
    assert registry.list_tools() == []


def test_register_function_can_expose_a_single_parameter() -> None:
    registry = ToolRegistry()
    registry.register_function(
        "shout",
        "放大文本",
        lambda text, loud=True: text.upper() if loud else text,
        parameter_name="text",
    )

    tool = registry.get_tool("shout")
    assert tool is not None
    assert [spec.name for spec in tool.get_parameters()] == ["text"]
    assert registry.execute_tool("shout", "hi") == "HI"


def test_duplicate_registration_overrides_and_warning_is_logged() -> None:
    registry = ToolRegistry()
    registry.register_function("echo", "旧", lambda text: "旧")
    registry.register_function("echo", "新", lambda text: f"新:{text}")

    assert len(registry) == 1
    assert registry.execute_tool("echo", "x") == "新:x"


def test_unregister_reports_whether_tool_existed() -> None:
    registry = ToolRegistry()
    registry.register_function("echo", "回显", lambda text: text)

    assert registry.unregister("echo") is True
    assert registry.unregister("echo") is False
    assert "echo" not in registry


def test_get_tools_description_and_empty_placeholder() -> None:
    registry = ToolRegistry()
    assert registry.get_tools_description() == "暂无可用工具"

    registry.register_tool(_Recorder())
    assert registry.get_tools_description() == "- recorder: 记录参数"


def test_to_openai_schemas_marks_required_and_describes_defaults() -> None:
    registry = ToolRegistry()
    tool = _Recorder(
        "search",
        [
            ToolParameter("query", "string", "关键词"),
            ToolParameter("limit", "integer", "条数", required=False, default=3),
        ],
    )
    registry.register_tool(tool)

    schema = registry.to_openai_schemas()[0]["function"]
    assert schema["name"] == "search"
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["limit"]["description"] == "条数（默认: 3）"


def test_single_required_parameter_accepts_bare_string() -> None:
    registry = ToolRegistry()
    tool = _Recorder(
        "search",
        [
            ToolParameter("query", "string", "关键词"),
            ToolParameter("limit", "integer", "条数", required=False, default=3),
        ],
    )
    registry.register_tool(tool)

    assert registry.execute_tool("search", "python") == "记录:python"
    assert tool.seen == [{"query": "python", "limit": 3}]


def test_bare_string_is_rejected_for_multi_required_parameter_tool() -> None:
    registry = ToolRegistry()
    tool = _Recorder(
        "compare",
        [ToolParameter("left", "string", "左"), ToolParameter("right", "string", "右")],
    )
    registry.register_tool(tool)

    with pytest.raises(ToolException) as excinfo:
        registry.execute_tool("compare", "1,2")

    assert excinfo.value.context["declared"] == ["left", "right"]


def test_unknown_tool_lists_available_tools() -> None:
    registry = ToolRegistry()
    registry.register_tool(_Recorder())

    with pytest.raises(ToolException) as excinfo:
        registry.execute_tool("missing", "x")

    assert excinfo.value.context["available"] == ["recorder"]


def test_unknown_and_missing_parameters_are_rejected() -> None:
    registry = ToolRegistry()
    registry.register_tool(_Recorder())

    with pytest.raises(ToolException) as unknown:
        registry.execute_tool("recorder", {"other": "x"})
    assert unknown.value.context["unknown"] == ["other"]

    with pytest.raises(ToolException) as missing:
        registry.execute_tool("recorder", {"text": None})
    assert missing.value.context["missing"] == "text"


def test_unsupported_input_type_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register_tool(_Recorder())

    with pytest.raises(ToolException) as excinfo:
        registry.execute_tool("recorder", cast(Any, 5))

    assert excinfo.value.context["type"] == "int"


def test_parameters_are_coerced_to_declared_types() -> None:
    registry = ToolRegistry()
    tool = _Recorder(
        "typed",
        [
            ToolParameter("text", "string", "文本"),
            ToolParameter("count", "integer", "次数", required=False, default=1),
            ToolParameter("ratio", "number", "比例", required=False, default=0.5),
            ToolParameter("flag", "boolean", "开关", required=False, default=False),
            ToolParameter("tags", "array", "标签", required=False, default=()),
        ],
    )
    registry.register_tool(tool)

    registry.execute_tool(
        "typed", {"text": 7, "count": "3", "ratio": "1.5", "flag": "ON", "tags": "a,b"}
    )

    assert tool.seen == [{"text": "7", "count": 3, "ratio": 1.5, "flag": True, "tags": ["a", "b"]}]


def test_uncoercible_parameter_reports_the_original_value() -> None:
    registry = ToolRegistry()
    registry.register_tool(_Recorder("count_tool", [ToolParameter("count", "integer", "次数")]))

    with pytest.raises(ToolException) as excinfo:
        registry.execute_tool("count_tool", {"count": "很多"})

    assert excinfo.value.context["expected"] == "integer"
    assert excinfo.value.context["value"] == "很多"


def test_parameter_declaration_rejects_required_with_default() -> None:
    with pytest.raises(ToolException) as excinfo:
        ToolParameter("limit", "integer", "条数", required=True, default=3)

    assert excinfo.value.context["parameter"] == "limit"


def test_tool_exception_passes_through_with_position_added() -> None:
    class _Failing(Tool):
        def __init__(self) -> None:
            super().__init__("boom", "总是失败", [ToolParameter("text", "string", "文本")])

        def run(self, parameters: Mapping[str, Any]) -> str:
            raise ToolException("业务失败", context={"reason": "policy"})

    registry = ToolRegistry()
    registry.register_tool(_Failing())

    with pytest.raises(ToolException) as excinfo:
        registry.execute_tool("boom", "x")

    assert excinfo.value.context["reason"] == "policy"
    assert excinfo.value.context["tool"] == "boom"
    assert "policy" in str(excinfo.value)


def test_unexpected_tool_error_is_wrapped_with_details() -> None:
    class _Buggy(Tool):
        def __init__(self) -> None:
            super().__init__("buggy", "抛非工具异常", [ToolParameter("text", "string", "文本")])

        def run(self, parameters: Mapping[str, Any]) -> str:
            raise ValueError("内部错误")

    registry = ToolRegistry()
    registry.register_tool(_Buggy())

    with pytest.raises(ToolException) as excinfo:
        registry.execute_tool("buggy", "x")

    assert excinfo.value.context["error"] == "ValueError"
    assert excinfo.value.context["detail"] == "内部错误"
    assert isinstance(excinfo.value.__cause__, ValueError)


async def test_aexecute_tool_runs_sync_tool_off_the_event_loop() -> None:
    registry = ToolRegistry()
    recorder = _Recorder()
    registry.register_tool(recorder)

    assert await registry.aexecute_tool("recorder", "异步调用") == "记录:异步调用"
    assert recorder.seen == [{"text": "异步调用"}]


async def test_aexecute_tool_prefers_tool_async_implementation() -> None:
    registry = ToolRegistry()
    registry.register_tool(_AsyncOnly())

    assert await registry.aexecute_tool("async_only", "x") == "异步:x"


def test_registry_container_protocol() -> None:
    registry = ToolRegistry()
    registry.register_tool(_Recorder())

    assert len(registry) == 1
    assert "recorder" in registry
    assert [tool.name for tool in registry] == ["recorder"]
