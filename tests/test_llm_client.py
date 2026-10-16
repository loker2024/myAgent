"""LLM 客户端的行为测试。

测试不替换 LLM 方法，而是把网络对端换成 127.0.0.1 上的脚本化服务，
因此真实覆盖 httpx 请求、SSE 解析、错误映射与同步包装。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from myagent import (
    HelloAgentsLLM,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMException,
    LLMRateLimitError,
    LLMServerError,
    Message,
)
from tests.conftest import Scripted, StubLLM

SSE_CHUNKS = ["Hel", "lo", " 世界"]


def _openai_client(stub: StubLLM, **overrides: Any) -> HelloAgentsLLM:
    params: dict[str, Any] = {
        "provider": "openai",
        "base_url": stub.url,
        "api_key": "test-key",
        "model": "stub-model",
        "env": {},
    }
    params.update(overrides)
    return HelloAgentsLLM(**params)


def _anthropic_client(stub: StubLLM, **overrides: Any) -> HelloAgentsLLM:
    params: dict[str, Any] = {
        "provider": "anthropic",
        "base_url": stub.root,
        "api_key": "test-key",
        "model": "stub-model",
        "env": {},
    }
    params.update(overrides)
    return HelloAgentsLLM(**params)


async def test_athink_yields_stream_chunks_in_order(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=True, blocks=SSE_CHUNKS))
    client = _openai_client(stub_llm)

    chunks = [chunk async for chunk in client.athink([Message("hi", "user")])]

    assert chunks == SSE_CHUNKS
    assert "".join(chunks) == "Hello 世界"
    assert stub_llm.requests[0]["payload"]["stream"] is True
    await client.aclose()


async def test_ainvoke_returns_complete_text(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="完整回答"))
    client = _openai_client(stub_llm)

    assert await client.ainvoke([Message("hi", "user")]) == "完整回答"
    assert stub_llm.requests[0]["payload"]["stream"] is False
    await client.aclose()


async def test_athink_falls_back_when_provider_ignores_stream(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=False, content="本地服务忽略了 stream"))
    client = _openai_client(stub_llm)

    chunks = [chunk async for chunk in client.athink([Message("hi", "user")])]

    assert chunks == ["本地服务忽略了 stream"]
    assert stub_llm.requests[0]["payload"]["stream"] is True
    await client.aclose()


async def test_openai_request_contract(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))
    client = _openai_client(stub_llm)

    await client.ainvoke(
        [Message("系统提示", "system"), {"role": "user", "content": "问题"}],
    )

    request = stub_llm.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer test-key"
    assert request["headers"]["content-type"] == "application/json"
    assert request["payload"]["model"] == "stub-model"
    assert request["payload"]["temperature"] == 0.7
    assert request["payload"]["messages"] == [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "问题"},
    ]
    assert "max_tokens" not in request["payload"]  # 未配置时不发送该字段
    await client.aclose()


async def test_temperature_and_max_tokens_overrides(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))
    client = _openai_client(stub_llm, max_tokens=512)

    await client.ainvoke([Message("hi", "user")], temperature=0.1, max_tokens=16)

    assert stub_llm.requests[0]["payload"]["temperature"] == 0.1
    assert stub_llm.requests[0]["payload"]["max_tokens"] == 16

    stub_llm.script.append(Scripted(content="ok"))
    await client.ainvoke([Message("hi", "user")])
    assert stub_llm.requests[1]["payload"]["max_tokens"] == 512  # 回落到实例默认值
    await client.aclose()


async def test_anthropic_request_contract_and_stream(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=True, blocks=["Claude", " 回答"]))
    client = _anthropic_client(stub_llm)

    chunks = [
        chunk
        async for chunk in client.athink([Message("系统提示", "system"), Message("问题", "user")])
    ]

    assert chunks == ["Claude", " 回答"]
    request = stub_llm.requests[0]
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == "test-key"
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert request["payload"]["system"] == "系统提示"
    assert request["payload"]["messages"] == [{"role": "user", "content": "问题"}]
    assert request["payload"]["max_tokens"] == 4096  # Anthropic 必填，取默认值
    assert request["payload"]["stream"] is True
    await client.aclose()


async def test_anthropic_strips_v1_suffix_from_base_url(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))
    client = _anthropic_client(stub_llm, base_url=f"{stub_llm.root}/v1")

    assert await client.ainvoke([Message("hi", "user")]) == "ok"
    assert stub_llm.requests[0]["path"] == "/v1/messages"
    await client.aclose()


async def test_anthropic_rejects_tool_role_before_sending(stub_llm: StubLLM) -> None:
    client = _anthropic_client(stub_llm)

    with pytest.raises(LLMException) as excinfo:
        await client.ainvoke([Message("工具结果", "tool")])

    assert "tool" in str(excinfo.value)
    assert stub_llm.requests == []
    await client.aclose()


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, LLMAuthenticationError),
        (403, LLMAuthenticationError),
        (429, LLMRateLimitError),
        (500, LLMServerError),
        (503, LLMServerError),
        (400, LLMException),
    ],
)
async def test_http_error_mapping(
    stub_llm: StubLLM, status: int, expected: type[LLMException]
) -> None:
    stub_llm.script.append(Scripted(status=status, body={"error": {"message": "boom"}}))
    client = _openai_client(stub_llm)

    with pytest.raises(expected) as excinfo:
        await client.ainvoke([Message("hi", "user")])

    assert excinfo.value.context["status"] == status
    assert "boom" in str(excinfo.value)
    assert isinstance(excinfo.value.retryable, bool)
    await client.aclose()


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(401, False), (429, True), (503, True), (400, False)],
)
async def test_retryable_flag_matches_error_class(
    stub_llm: StubLLM, status: int, retryable: bool
) -> None:
    stub_llm.script.append(Scripted(status=status))
    client = _openai_client(stub_llm)

    with pytest.raises(LLMException) as excinfo:
        await client.ainvoke([Message("hi", "user")])

    assert excinfo.value.retryable is retryable
    await client.aclose()


async def test_connection_failure_maps_to_retryable_error(stub_llm: StubLLM) -> None:
    # 本机存在本地拦截器，任意空闲端口会返回 502，无法用「关闭的端口」构造连接失败；
    # 因此注入一个必然抛 ConnectError 的传输层，其余代码路径仍为真实实现。
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = _openai_client(stub_llm, transport=httpx.MockTransport(handler))

    with pytest.raises(LLMConnectionError) as excinfo:
        await client.ainvoke([Message("hi", "user")])

    assert excinfo.value.retryable is True
    assert excinfo.value.context["provider"] == "openai"
    assert excinfo.value.context["error"] == "ConnectError"
    await client.aclose()


async def test_stream_error_event_raises(stub_llm: StubLLM) -> None:
    stub_llm.script.append(
        Scripted(sse=True, raw_sse='data: {"error": {"message": "quota exceeded"}}\n\n')
    )
    client = _openai_client(stub_llm)

    with pytest.raises(LLMException) as excinfo:
        [chunk async for chunk in client.athink([Message("hi", "user")])]

    assert "quota exceeded" in str(excinfo.value)
    await client.aclose()


async def test_malformed_sse_frame_raises(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=True, raw_sse="data: {not json}\n\n"))
    client = _openai_client(stub_llm)

    with pytest.raises(LLMException) as excinfo:
        [chunk async for chunk in client.athink([Message("hi", "user")])]

    assert "JSON" in str(excinfo.value)
    await client.aclose()


@pytest.mark.parametrize(
    "messages",
    [[], [{"role": "user"}], [{"role": 1, "content": "x"}], [42]],
    ids=["空列表", "缺 content", "role 非字符串", "元素类型错误"],
)
async def test_invalid_message_input_is_rejected(stub_llm: StubLLM, messages: list[Any]) -> None:
    client = _openai_client(stub_llm)

    with pytest.raises(LLMException):
        await client.ainvoke(messages)

    assert stub_llm.requests == []
    await client.aclose()


def test_sync_think_and_invoke_are_real_streaming_wrappers(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=True, blocks=SSE_CHUNKS))
    stub_llm.script.append(Scripted(content="完整回答"))
    client = _openai_client(stub_llm)

    streamed = list(client.think([Message("hi", "user")]))
    answer = client.invoke([Message("hi", "user")])

    assert streamed == SSE_CHUNKS
    assert answer == "完整回答"
    client.close()


def test_sync_calls_reuse_one_loop_and_one_pool(stub_llm: StubLLM) -> None:
    """回归：同步接口不得每次调用新建事件循环。

    httpx 连接池绑定在创建连接的事件循环上；若每次调用都 asyncio.run 新建循环，
    第二次调用会在已关闭的循环上复用连接并抛 ``RuntimeError: Event loop is closed``。
    """
    stub_llm.script.extend(
        [
            Scripted(sse=True, blocks=SSE_CHUNKS),
            Scripted(content="第二次"),
            Scripted(content="第三次"),
        ]
    )
    client = _openai_client(stub_llm)

    streamed = list(client.think([Message("hi", "user")]))
    second = client.invoke([Message("hi", "user")])
    third = client.invoke([Message("hi", "user")])

    assert (streamed, second, third) == (SSE_CHUNKS, "第二次", "第三次")
    loop, sync_client = client._sync_resources()
    assert loop is client._sync_resources()[0]
    assert loop.is_closed() is False
    assert sync_client.is_closed is False

    client.close()
    assert client._sync_loop is None
    assert client._sync_client is None


def test_sync_context_manager_closes_sync_pool(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))

    with _openai_client(stub_llm) as client:
        assert client.invoke([Message("hi", "user")]) == "ok"

    assert client._sync_client is None


def test_sync_pool_closed_by_close_and_unused_async_pool_alongside(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="同步"))
    client = _openai_client(stub_llm)

    assert client.invoke([Message("hi", "user")]) == "同步"
    sync_pool = client._sync_resources()[1]

    client.close()

    assert sync_pool.is_closed is True  # 同步连接池确实被关闭
    assert client._sync_client is None
    assert client._client.is_closed is True  # 异步客户端从未使用，可安全一并关闭


async def test_async_pool_closed_by_aclose_without_touching_sync_pool(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(sse=True, blocks=["异", "步"]))
    client = _openai_client(stub_llm)

    assert [chunk async for chunk in client.athink([Message("hi", "user")])] == ["异", "步"]
    assert client._sync_client is None  # 只走异步路径时不创建同步池

    await client.aclose()

    assert client._client.is_closed is True


def test_close_after_async_use_warns_instead_of_crossing_loops(
    stub_llm: StubLLM, caplog: pytest.LogCaptureFixture
) -> None:
    # 模拟真实场景：用户在协程里用过异步接口，随后又在同步上下文调用 close()。
    stub_llm.script.append(Scripted(content="ok"))
    client = _openai_client(stub_llm)
    asyncio.run(client.ainvoke([Message("hi", "user")]))

    with caplog.at_level(logging.WARNING, logger="myagent.core.llm"):
        client.close()

    assert "异步客户端已在事件循环中使用过" in caplog.text


async def test_sync_interfaces_refuse_to_run_inside_event_loop(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))
    client = _openai_client(stub_llm)

    with pytest.raises(RuntimeError):
        client.invoke([Message("hi", "user")])
    with pytest.raises(RuntimeError):
        list(client.think([Message("hi", "user")]))

    assert stub_llm.requests == []
    await client.aclose()


async def test_async_context_manager_closes_client(stub_llm: StubLLM) -> None:
    stub_llm.script.append(Scripted(content="ok"))

    async with _openai_client(stub_llm) as client:
        assert await client.ainvoke([Message("hi", "user")]) == "ok"
        assert client._client.is_closed is False

    assert client._client.is_closed is True
