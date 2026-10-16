"""内置工具（calculator / search）的行为测试。"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import httpx
import pytest

from myagent.core.exceptions import ToolException
from myagent.tools.builtin.calculator import CalculatorTool
from myagent.tools.builtin.search import SearchTool
from myagent.tools.registry import ToolRegistry

# ============================================================== calculator


def _calculate(expression: str) -> str:
    return CalculatorTool().run({"expression": expression})


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(15 + 3) * 2", "36"),
        ("2 ** 10", "1024"),
        ("17 % 5", "2"),
        ("17 // 5", "3"),
        ("-5 + 2", "-3"),
        ("+7", "7"),
        ("3 - -4", "7"),
    ],
)
def test_calculator_arithmetic(expression: str, expected: str) -> None:
    assert _calculate(expression) == expected


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(16)", "4"),
        ("math.sqrt(25)", "5"),
        ("sin(0)", "0"),
        ("cos(0)", "1"),
        ("floor(3.7)", "3"),
        ("ceil(3.2)", "4"),
        ("fabs(-3)", "3"),
        ("gcd(12, 18)", "6"),
        ("factorial(5)", "120"),
        ("pow(2, 3)", "8"),
        ("log(e)", "1"),
        ("pi", str(math.pi)),
        ("tau", str(math.tau)),
        ("math.e", str(math.e)),
    ],
)
def test_calculator_whitelisted_functions_and_constants(expression: str, expected: str) -> None:
    assert _calculate(expression) == expected


def test_calculator_normalizes_integral_floats() -> None:
    assert _calculate("15 / 3") == "5"
    assert _calculate("7 / 2") == "3.5"


def test_calculator_empty_expression_is_rejected() -> None:
    with pytest.raises(ToolException) as info:
        _calculate("   ")

    assert info.value.context["expression"] == "   "


def test_calculator_syntax_error_is_rejected() -> None:
    with pytest.raises(ToolException) as info:
        _calculate("1 +")

    assert info.value.context["expression"] == "1 +"


@pytest.mark.parametrize("expression", ["1 / 0", "1 % 0", "17 // 0"])
def test_calculator_division_by_zero_is_rejected(expression: str) -> None:
    with pytest.raises(ToolException) as info:
        _calculate(expression)

    assert info.value.context["expression"] == expression


@pytest.mark.parametrize("expression", ["sqrt(-1)", "log(0)", "log10(-5)", "asin(2)"])
def test_calculator_domain_errors_are_rejected(expression: str) -> None:
    with pytest.raises(ToolException) as info:
        _calculate(expression)

    assert info.value.context["expression"] == expression


def test_calculator_complex_result_is_rejected() -> None:
    with pytest.raises(ToolException) as info:
        _calculate("(-8) ** 0.5")

    assert "不是实数" in info.value.message


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "(1).__class__",
        "1 < 2",
        "1 and 2",
        "[1, 2][0]",
        "x + 1",
        "lambda: 1",
        "1 if 2 else 3",
        "f'{1}'",
        "open('x')",
        "math.__dict__",
    ],
)
def test_calculator_rejects_non_whitelisted_nodes(expression: str) -> None:
    with pytest.raises(ToolException):
        _calculate(expression)


def test_calculator_rejects_oversized_exponent() -> None:
    with pytest.raises(ToolException) as info:
        _calculate("2 ** 10001")

    assert info.value.context["limit"] == 10000
    assert _calculate("2 ** 1000") != ""


def test_calculator_rejects_oversized_factorial() -> None:
    with pytest.raises(ToolException) as info:
        _calculate("factorial(1001)")

    assert info.value.context["limit"] == 1000


def test_calculator_via_registry_accepts_bare_string() -> None:
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    assert registry.execute_tool("calculator", "(15 + 3) * 2") == "36"
    assert registry.execute_tool("calculator", "sqrt(16)") == "4"


def test_calculator_via_registry_missing_required_parameter() -> None:
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    with pytest.raises(ToolException) as info:
        registry.execute_tool("calculator", {})

    assert info.value.context["missing"] == "expression"


# ============================================================== search


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_requires_a_known_backend() -> None:
    with pytest.raises(ToolException) as info:
        SearchTool(backend="bing", env={})

    assert info.value.context["backend"] == "bing"


def test_search_available_backends_reflect_keys() -> None:
    assert SearchTool(env={"TAVILY_API_KEY": "t"}).available_backends == ("tavily",)
    assert SearchTool(env={"SERPAPI_API_KEY": "s"}).available_backends == ("serpapi",)
    assert SearchTool(env={"TAVILY_API_KEY": "t", "SERPAPI_API_KEY": "s"}).available_backends == (
        "tavily",
        "serpapi",
    )
    assert SearchTool(env={}).available_backends == ()


def test_tavily_success_formats_answer_and_truncates_results() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "answer": "Python 是一种通用编程语言。",
                "results": [
                    {"title": "标题0", "url": "https://e.com/0", "content": "A" * 300},
                    {"title": "标题1", "url": "https://e.com/1", "content": "摘要1"},
                    {"title": "标题2", "url": "https://e.com/2", "content": "摘要2"},
                    {"title": "标题3", "url": "https://e.com/3", "content": "摘要3"},
                ],
            },
        )

    tool = SearchTool(backend="tavily", tavily_key="k", client=_client(handler))
    text = tool.run({"query": "python", "limit": 3})

    assert len(seen) == 1
    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.tavily.com/search"
    assert json.loads(request.content) == {
        "api_key": "k",
        "query": "python",
        "max_results": 3,
        "include_answer": True,
    }
    assert "答案：Python 是一种通用编程语言。" in text
    assert text.count("链接：") == 3
    assert "A" * 200 in text
    assert "A" * 201 not in text
    assert "标题3" not in text


def test_tavily_default_limit_is_three() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    tool = SearchTool(backend="tavily", tavily_key="k", client=_client(handler))
    tool.run({"query": "python"})

    assert json.loads(seen[0].content)["max_results"] == 3


def test_explicit_key_takes_priority_over_env() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    tool = SearchTool(
        backend="tavily",
        tavily_key="explicit",
        env={"TAVILY_API_KEY": "from-env"},
        client=_client(handler),
    )
    tool.run({"query": "python"})

    assert json.loads(seen[0].content)["api_key"] == "explicit"


def test_serpapi_success_formats_answer_box() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"answer_box": {"answer": "SerpAPI 的答案"}})

    tool = SearchTool(backend="serpapi", serpapi_key="s", client=_client(handler))
    text = tool.run({"query": "python", "limit": 3})

    request = seen[0]
    assert request.method == "GET"
    assert request.url.host == "serpapi.com"
    assert request.url.path == "/search.json"
    assert dict(request.url.params) == {
        "engine": "google",
        "q": "python",
        "api_key": "s",
        "num": "3",
    }
    assert text == "答案：SerpAPI 的答案"


def test_serpapi_falls_back_to_knowledge_graph_then_organic_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "knowledge_graph": {"description": "知识图谱描述"},
                "organic_results": [
                    {"title": "结果0", "link": "https://e.com/0", "snippet": "片段0"},
                    {"title": "结果1", "link": "https://e.com/1", "snippet": "片段1"},
                ],
            },
        )

    tool = SearchTool(backend="serpapi", serpapi_key="s", client=_client(handler))
    text = tool.run({"query": "python", "limit": 3})

    assert "答案：知识图谱描述" in text
    assert "1. 结果0\n   链接：https://e.com/0\n   摘要：片段0" in text
    assert "2. 结果1" in text


def test_serpapi_explicit_limit_is_applied() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "organic_results": [
                    {"title": "结果0", "link": "https://e.com/0", "snippet": "片段0"},
                    {"title": "结果1", "link": "https://e.com/1", "snippet": "片段1"},
                ]
            },
        )

    tool = SearchTool(backend="serpapi", serpapi_key="s", client=_client(handler))
    text = tool.run({"query": "python", "limit": 1})

    assert dict(seen[0].url.params)["num"] == "1"
    assert text.count("链接：") == 1
    assert "结果1" not in text


def test_hybrid_falls_back_to_serpapi_when_tavily_fails() -> None:
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "api.tavily.com":
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(
            200,
            json={
                "organic_results": [
                    {"title": "降级结果", "link": "https://e.com", "snippet": "降级摘要"}
                ]
            },
        )

    tool = SearchTool(
        backend="hybrid",
        tavily_key="t",
        serpapi_key="s",
        client=_client(handler),
    )
    text = tool.run({"query": "python", "limit": 3})

    assert hosts == ["api.tavily.com", "serpapi.com"]
    assert "降级摘要" in text


def test_hybrid_reports_attempts_when_all_backends_fail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "down"})

    tool = SearchTool(
        backend="hybrid",
        tavily_key="t",
        serpapi_key="s",
        client=_client(handler),
    )

    with pytest.raises(ToolException) as info:
        tool.run({"query": "python", "limit": 3})

    attempts = info.value.context["attempts"]
    assert isinstance(attempts, list)
    assert len(attempts) == 2
    assert attempts[0].startswith("tavily:")
    assert attempts[1].startswith("serpapi:")


def test_search_without_any_key_explains_configuration() -> None:
    tool = SearchTool(backend="hybrid", env={})

    with pytest.raises(ToolException) as info:
        tool.run({"query": "python"})

    assert "TAVILY_API_KEY" in str(info.value)
    assert "SERPAPI_API_KEY" in str(info.value)


def test_single_backend_without_key_is_rejected() -> None:
    tool = SearchTool(backend="tavily", env={})

    with pytest.raises(ToolException) as info:
        tool.run({"query": "python"})

    assert "TAVILY_API_KEY" in info.value.message


def test_search_via_registry_accepts_bare_string_query() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    registry = ToolRegistry()
    registry.register_tool(SearchTool(tavily_key="k", client=_client(handler)))

    result = registry.execute_tool("search", "Python")

    body = json.loads(seen[0].content)
    assert body["query"] == "Python"
    assert body["max_results"] == 3
    assert result == "未找到相关结果。"
