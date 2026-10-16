"""测试夹具：本地 stub LLM 服务。

用真实 HTTP + SSE 提供 OpenAI 兼容与 Anthropic 两种协议的响应，测试不打桩 LLM 方法，
只把网络对端换成 127.0.0.1 上的脚本化服务，因此无需任何 API key。
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import pytest


@dataclass
class Scripted:
    """一次 scripted 响应的描述。

    Args:
        status: HTTP 状态码。
        content: 非流式响应的文本内容。
        sse: 是否以 SSE 返回。
        blocks: SSE 增量文本片段。
        raw_sse: 直接给出的 SSE 原始文本，优先于 ``blocks``。
        body: 非流式响应的原始 JSON 体，优先于 ``content``。
    """

    status: int = 200
    content: str = "stub-answer"
    sse: bool = False
    blocks: list[str] = field(default_factory=list)
    raw_sse: str | None = None
    body: dict[str, Any] | None = None


@dataclass
class StubLLM:
    """夹具句柄：base_url、脚本队列与收到的请求。"""

    root: str
    url: str
    script: deque[Scripted]
    requests: list[dict[str, Any]]


def _openai_sse(blocks: list[str]) -> str:
    frames = [
        "data: " + json.dumps({"choices": [{"delta": {"content": block}}]}) + "\n\n"
        for block in blocks
    ]
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def _anthropic_sse(blocks: list[str]) -> str:
    frames = ["event: message_start\ndata: " + json.dumps({"type": "message_start"}) + "\n\n"]
    for block in blocks:
        frames.append(
            "event: content_block_delta\ndata: "
            + json.dumps(
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": block}}
            )
            + "\n\n"
        )
    frames.append("event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n")
    return "".join(frames)


class _StubHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, script: deque[Scripted], requests: list[dict[str, Any]]) -> None:
        self.script = script
        self.requests = requests
        super().__init__(("127.0.0.1", 0), _Handler)

    def next_scripted(self) -> Scripted:
        return self.script.popleft() if self.script else Scripted()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """静默 HTTP 访问日志，避免污染测试输出。"""

    def do_POST(self) -> None:
        server = cast(_StubHTTPServer, self.server)
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw.decode("utf-8", "replace")}
        server.requests.append(
            {
                "path": self.path,
                "headers": {key.lower(): value for key, value in self.headers.items()},
                "payload": payload,
            }
        )

        item = server.next_scripted()
        is_anthropic = self.path.endswith("/v1/messages")

        if item.status >= 400:
            error_body = item.body or {"error": {"message": "stub error"}}
            self._send(item.status, "application/json", json.dumps(error_body).encode())
            return

        if item.sse:
            if item.raw_sse is not None:
                text = item.raw_sse
            elif is_anthropic:
                text = _anthropic_sse(item.blocks)
            else:
                text = _openai_sse(item.blocks)
            self._send(200, "text/event-stream", text.encode())
            return

        body = item.body or (
            {"content": [{"type": "text", "text": item.content}], "type": "message"}
            if is_anthropic
            else {"choices": [{"message": {"role": "assistant", "content": item.content}}]}
        )
        self._send(200, "application/json", json.dumps(body).encode())

    def _send(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def stub_llm() -> Iterator[StubLLM]:
    """启动 stub 服务并返回句柄；测试通过 ``stub.script`` 预置响应。"""
    script: deque[Scripted] = deque()
    requests: list[dict[str, Any]] = []
    server = _StubHTTPServer(script, requests)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast("tuple[str, int]", server.server_address)
    try:
        yield StubLLM(
            root=f"http://{host}:{port}",
            url=f"http://{host}:{port}/v1",
            script=script,
            requests=requests,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
