"""统一 LLM 客户端：一套接口，多家 provider。

教程定位：第七章 7.2（7.2.1 多提供商 / 7.2.2 本地模型 / 7.2.3 自动检测机制）。
与教程的差异见 docs/deviations.md：D3（httpx 取代 openai SDK）、D5（logging 取代 print）、
D10（异常分类）、D14（新增 Anthropic 适配器）。

对外方法名与教程一致：``think`` / ``invoke`` / ``stream_invoke`` / ``provider`` / ``model``；
新增异步接口 ``athink`` / ``ainvoke``（异步是唯一真实实现，同步接口是其包装）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

import httpx

from ._sync import ensure_no_running_loop
from .config import env_float, env_int
from .exceptions import (
    ConfigException,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMException,
    LLMRateLimitError,
    LLMServerError,
)
from .message import Message

__all__ = [
    "PROVIDERS",
    "HelloAgentsLLM",
    "ProviderSpec",
    "default_model_for",
    "detect_provider",
    "resolve_credentials",
    "spec_for",
]

logger = logging.getLogger(__name__)

Protocol = Literal["openai", "anthropic"]

DEFAULT_TIMEOUT: Final[float] = 60.0
ANTHROPIC_API_VERSION: Final[str] = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS: Final[int] = 4096
_STREAM_CONTENT_TYPE: Final[str] = "text/event-stream"


@dataclass(frozen=True)
class ProviderSpec:
    """一个 provider 的静态描述。

    Args:
        name: provider 名。
        protocol: 线上协议族，决定请求体与响应解析方式。
        api_key_envs: 依次查找密钥的环境变量名。
        base_url_envs: 依次查找服务地址的环境变量名。
        default_base_url: 环境变量缺失时的默认地址；``None`` 表示必须显式提供。
        default_model: 环境变量缺失时的默认模型；``None`` 表示必须显式提供。
        fallback_api_key: 本地服务免密钥时的占位密钥。
    """

    name: str
    protocol: Protocol
    api_key_envs: tuple[str, ...]
    base_url_envs: tuple[str, ...]
    default_base_url: str | None
    default_model: str | None
    fallback_api_key: str | None = None


def _generic(name: str) -> ProviderSpec:
    """``auto`` / ``custom``：只认通用环境变量，默认值全部要求显式提供。"""
    return ProviderSpec(
        name=name,
        protocol="openai",
        api_key_envs=("LLM_API_KEY",),
        base_url_envs=("LLM_BASE_URL",),
        default_base_url=None,
        default_model=None,
    )


PROVIDERS: Final[dict[str, ProviderSpec]] = {
    "openai": ProviderSpec(
        "openai",
        "openai",
        ("OPENAI_API_KEY", "LLM_API_KEY"),
        ("LLM_BASE_URL",),
        "https://api.openai.com/v1",
        "gpt-4o-mini",
    ),
    "deepseek": ProviderSpec(
        "deepseek",
        "openai",
        ("DEEPSEEK_API_KEY", "LLM_API_KEY"),
        ("LLM_BASE_URL",),
        "https://api.deepseek.com",
        "deepseek-chat",
    ),
    "qwen": ProviderSpec(
        "qwen",
        "openai",
        ("DASHSCOPE_API_KEY", "LLM_API_KEY"),
        ("LLM_BASE_URL",),
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
    ),
    "modelscope": ProviderSpec(
        "modelscope",
        "openai",
        ("MODELSCOPE_API_KEY", "LLM_API_KEY"),
        ("LLM_BASE_URL",),
        "https://api-inference.modelscope.cn/v1/",
        "Qwen/Qwen2.5-72B-Instruct",
    ),
    "kimi": ProviderSpec(
        "kimi",
        "openai",
        ("KIMI_API_KEY", "MOONSHOT_API_KEY", "LLM_API_KEY"),
        ("LLM_BASE_URL",),
        "https://api.moonshot.cn/v1",
        "moonshot-v1-8k",
    ),
    "zhipu": ProviderSpec(
        "zhipu",
        "openai",
        ("ZHIPU_API_KEY", "GLM_API_KEY", "LLM_API_KEY"),
        ("LLM_BASE_URL",),
        "https://open.bigmodel.cn/api/paas/v4",
        "glm-4",
    ),
    "ollama": ProviderSpec(
        "ollama",
        "openai",
        ("OLLAMA_API_KEY", "LLM_API_KEY"),
        ("OLLAMA_HOST", "LLM_BASE_URL"),
        "http://localhost:11434/v1",
        "llama3.2",
        fallback_api_key="ollama",
    ),
    "vllm": ProviderSpec(
        "vllm",
        "openai",
        ("VLLM_API_KEY", "LLM_API_KEY"),
        ("VLLM_HOST", "LLM_BASE_URL"),
        "http://localhost:8000/v1",
        None,
        fallback_api_key="vllm",
    ),
    "local": ProviderSpec(
        "local",
        "openai",
        ("LLM_API_KEY",),
        ("LLM_BASE_URL",),
        "http://localhost:8000/v1",
        None,
        fallback_api_key="local",
    ),
    "anthropic": ProviderSpec(
        "anthropic",
        "anthropic",
        ("ANTHROPIC_API_KEY", "LLM_API_KEY"),
        ("ANTHROPIC_BASE_URL", "LLM_BASE_URL"),
        "https://api.anthropic.com",
        "claude-3-5-sonnet-latest",
    ),
    "auto": _generic("auto"),
    "custom": _generic("custom"),
}

# 环境变量检测顺序：正文 7.2.3 给出的可见顺序（modelscope → openai → zhipu）+ 其余提供商。
_PROVIDER_ENV_ORDER: Final[tuple[tuple[str, str], ...]] = (
    ("MODELSCOPE_API_KEY", "modelscope"),
    ("OPENAI_API_KEY", "openai"),
    ("ZHIPU_API_KEY", "zhipu"),
    ("GLM_API_KEY", "zhipu"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("DASHSCOPE_API_KEY", "qwen"),
    ("KIMI_API_KEY", "kimi"),
    ("MOONSHOT_API_KEY", "kimi"),
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("OLLAMA_API_KEY", "ollama"),
    ("OLLAMA_HOST", "ollama"),
    ("VLLM_API_KEY", "vllm"),
    ("VLLM_HOST", "vllm"),
)

_BASE_URL_MARKERS: Final[tuple[tuple[str, str], ...]] = (
    ("api-inference.modelscope.cn", "modelscope"),
    ("api.openai.com", "openai"),
    ("api.deepseek.com", "deepseek"),
    ("dashscope.aliyuncs.com", "qwen"),
    ("api.moonshot.cn", "kimi"),
    ("open.bigmodel.cn", "zhipu"),
    ("api.anthropic.com", "anthropic"),
)

_LOCAL_HOST_MARKERS: Final[tuple[str, ...]] = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]")
_LOCAL_PORTS: Final[tuple[tuple[str, str], ...]] = (
    (":11434", "ollama"),
    (":8000", "vllm"),
    (":8080", "local"),
    (":7860", "local"),
    (":5000", "local"),
)


def spec_for(provider: str) -> ProviderSpec:
    """取得 provider 描述。

    Raises:
        ConfigException: provider 名未在 :data:`PROVIDERS` 中登记。
    """
    try:
        return PROVIDERS[provider.lower()]
    except KeyError as exc:
        raise ConfigException(
            f"未知 provider: {provider}", context={"supported": sorted(PROVIDERS)}
        ) from exc


def detect_provider(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """按正文 7.2.3 的优先级推断 provider。

    顺序为：提供商专属环境变量 → ``base_url`` 域名/端口 → 密钥格式 → ``"auto"``。

    Args:
        api_key: 显式传入的密钥，优先于 ``env`` 中的通用密钥。
        base_url: 显式传入的服务地址，优先于 ``env`` 中的通用地址。
        env: 环境变量映射；``None`` 表示 ``os.environ``。
    """
    env_map = os.environ if env is None else env

    for var, provider in _PROVIDER_ENV_ORDER:
        if env_map.get(var):
            return provider

    resolved_base_url = base_url or env_map.get("LLM_BASE_URL")
    resolved_api_key = api_key or env_map.get("LLM_API_KEY")

    if resolved_base_url:
        lower = resolved_base_url.lower()
        for marker, provider in _BASE_URL_MARKERS:
            if marker in lower:
                return provider
        if any(host in lower for host in _LOCAL_HOST_MARKERS):
            for port, provider in _LOCAL_PORTS:
                if port in lower:
                    return provider
            return "local"
        for port, provider in _LOCAL_PORTS:
            if port in lower:
                return provider

    if resolved_api_key:
        lower_key = resolved_api_key.lower()
        if lower_key.startswith("ms-"):
            return "modelscope"
        if lower_key.startswith("sk-ant-"):
            return "anthropic"
        if lower_key in {"ollama", "vllm", "local"}:
            return lower_key
        if resolved_api_key.endswith(".") or "." in resolved_api_key[-20:]:
            return "zhipu"

    return "auto"


def _first_env(env: Mapping[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return None


def resolve_credentials(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """按 provider 解析出 ``(api_key, base_url)``，显式参数优先于环境变量，最后落默认值。"""
    spec = spec_for(provider)
    env_map = os.environ if env is None else env
    resolved_key = api_key or _first_env(env_map, spec.api_key_envs) or spec.fallback_api_key
    resolved_base_url = base_url or _first_env(env_map, spec.base_url_envs) or spec.default_base_url
    return resolved_key, resolved_base_url


def default_model_for(provider: str) -> str | None:
    """取得 provider 的默认模型名；``None`` 表示必须显式指定。"""
    return spec_for(provider).default_model


def _normalize_messages(
    messages: Iterable[Message | Mapping[str, Any]],
) -> list[dict[str, str]]:
    """把 ``Message`` 或裸字典统一成线上协议要求的 ``{"role", "content"}`` 列表。"""
    wire: list[dict[str, str]] = []
    for item in messages:
        if isinstance(item, Message):
            wire.append(item.to_dict())
            continue
        if isinstance(item, Mapping):
            role = item.get("role")
            content = item.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise LLMException(
                    "消息字典必须含字符串类型的 role 与 content",
                    context={"message": dict(item)},
                )
            wire.append({"role": role, "content": content})
            continue
        raise LLMException("不支持的消息类型", context={"type": type(item).__name__})
    if not wire:
        raise LLMException("messages 不能为空")
    return wire


class HelloAgentsLLM:
    """多 provider 的 LLM 客户端。

    Args:
        model: 模型名；缺省时依次取 ``LLM_MODEL_ID``、provider 默认模型。
        api_key: 密钥；缺省时按 provider 查找专属环境变量，再取 ``LLM_API_KEY``。
        base_url: 服务地址；缺省时查找提供商的专属环境变量，再取 ``LLM_BASE_URL``。
        provider: 指定 provider；``None`` 表示按 :func:`detect_provider` 自动检测。
        temperature: 默认采样温度。
        max_tokens: 默认响应上限。
        timeout: 单次请求超时秒数；缺省取 ``LLM_TIMEOUT``，再取 60。
        env: 环境变量映射，供测试注入；``None`` 表示 ``os.environ``。
        transport: httpx 传输层，供测试注入。

    生命周期：异步接口使用内部异步客户端，请用 ``async with`` 或 ``await aclose()`` 收起；
    同步接口（``think`` / ``invoke`` / ``stream_invoke``）独占一个事件循环与一个连接池，
    首次调用时创建，需在同步上下文中用 ``close()``（或 ``with``）回收。两者不要混用后
    只关一半：混用时 ``aclose()`` / ``close()`` 会各管自己那半并给出提示。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float | None = None,
        *,
        env: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        env_map = os.environ if env is None else env
        resolved_provider = (
            provider or detect_provider(api_key=api_key, base_url=base_url, env=env_map)
        ).lower()
        spec = spec_for(resolved_provider)
        resolved_key, resolved_base_url = resolve_credentials(
            resolved_provider, api_key=api_key, base_url=base_url, env=env_map
        )
        resolved_model = model or env_map.get("LLM_MODEL_ID") or spec.default_model

        if not resolved_model:
            raise ConfigException(
                "缺少模型名：请传入 model 或设置 LLM_MODEL_ID",
                context={"provider": resolved_provider},
            )
        if not resolved_key:
            raise ConfigException(
                "缺少 API 密钥：请传入 api_key 或设置对应环境变量",
                context={"provider": resolved_provider, "checked_env": list(spec.api_key_envs)},
            )
        if not resolved_base_url:
            raise ConfigException(
                "缺少服务地址：请传入 base_url 或设置 LLM_BASE_URL",
                context={"provider": resolved_provider, "checked_env": list(spec.base_url_envs)},
            )

        self.provider: str = resolved_provider
        self.model: str = resolved_model
        self.api_key: str = resolved_key
        self.base_url: str = resolved_base_url
        self.temperature: float = temperature
        self.max_tokens: int | None = (
            max_tokens if max_tokens is not None else env_int(env_map, "MAX_TOKENS")
        )
        self.timeout: float = (
            timeout if timeout is not None else env_float(env_map, "LLM_TIMEOUT", DEFAULT_TIMEOUT)
        )

        self._spec = spec
        self._transport = transport
        self._client = httpx.AsyncClient(timeout=self.timeout, transport=transport)
        # 同步接口专用资源：httpx 连接池绑定在创建连接的事件循环上，
        # 因此同步调用必须长期复用同一个循环（见 _sync_resources）。
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_client: httpx.AsyncClient | None = None
        self._async_used = False

    # ---------------------------------------------------------------- 请求构造

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if self._spec.protocol == "anthropic":
            if base.endswith("/v1"):
                base = base[: -len("/v1")]
            return f"{base}/v1/messages"
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        if self._spec.protocol == "anthropic":
            return {
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
            }
        return {"authorization": f"Bearer {self.api_key}", "content-type": "application/json"}

    def _context(self, **extra: object) -> dict[str, object]:
        return {"provider": self.provider, "model": self.model, "base_url": self.base_url, **extra}

    def _build_payload(
        self,
        messages: Iterable[Message | Mapping[str, Any]],
        *,
        stream: bool,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        wire = _normalize_messages(messages)
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_max_tokens = self.max_tokens if max_tokens is None else max_tokens

        if self._spec.protocol == "anthropic":
            system_parts = [item["content"] for item in wire if item["role"] == "system"]
            conversation: list[dict[str, str]] = []
            for item in wire:
                if item["role"] == "system":
                    continue
                if item["role"] == "tool":
                    raise LLMException(
                        "Anthropic 适配器不支持 tool 角色消息",
                        context=self._context(role="tool"),
                    )
                conversation.append({"role": item["role"], "content": item["content"]})
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": conversation,
                "max_tokens": resolved_max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
                "temperature": resolved_temperature,
                "stream": stream,
            }
            if system_parts:
                payload["system"] = "\n\n".join(system_parts)
            return payload

        payload = {
            "model": self.model,
            "messages": wire,
            "temperature": resolved_temperature,
            "stream": stream,
        }
        if resolved_max_tokens is not None:
            payload["max_tokens"] = resolved_max_tokens
        return payload

    # ---------------------------------------------------------------- 响应解析

    def _parse_stream_line(self, line: str) -> str | None:
        """解析一行 SSE，返回其中的文本增量（无文本则 ``None``）。"""
        if not line.startswith("data:"):
            return None
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            return None
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise LLMException(
                "SSE 数据块不是合法 JSON", context=self._context(chunk=data[:120])
            ) from exc
        if not isinstance(event, Mapping):
            return None
        error = event.get("error")
        if error:
            raise LLMException(
                "provider 在流式响应中返回错误", context=self._context(detail=str(error)[:200])
            )
        return self._extract_delta(event)

    def _extract_delta(self, event: Mapping[str, Any]) -> str | None:
        if self._spec.protocol == "anthropic":
            if event.get("type") != "content_block_delta":
                return None
            delta = event.get("delta")
            if not isinstance(delta, Mapping):
                return None
            text = delta.get("text")
            return text if isinstance(text, str) and text else None

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, Mapping):
            return None
        delta = first.get("delta")
        if not isinstance(delta, Mapping):
            return None
        content = delta.get("content")
        return content if isinstance(content, str) and content else None

    def _parse_body(self, body: bytes) -> list[str]:
        """解析非流式响应体，返回文本片段列表。"""
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            raise LLMException(
                "响应体不是合法 JSON",
                context=self._context(body=body[:120].decode("utf-8", "replace")),
            ) from exc
        if not isinstance(data, Mapping):
            raise LLMException(
                "响应体不是 JSON 对象", context=self._context(type=type(data).__name__)
            )

        if self._spec.protocol == "anthropic":
            blocks = data.get("content")
            if not isinstance(blocks, list):
                return []
            text = "".join(
                block.get("text", "")
                for block in blocks
                if isinstance(block, Mapping)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
            return [text] if text else []

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        return [content] if isinstance(content, str) and content else []

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = self._error_detail(await response.aread())
        context = self._context(status=response.status_code, detail=detail)
        if response.status_code in (401, 403):
            raise LLMAuthenticationError("LLM 凭据被拒绝", context=context)
        if response.status_code == 429:
            raise LLMRateLimitError("LLM 触发限流", context=context)
        if response.status_code >= 500:
            raise LLMServerError("LLM 服务端错误", context=context)
        raise LLMException("LLM 请求被拒绝", context=context)

    @staticmethod
    def _error_detail(body: bytes) -> str:
        text = body.decode("utf-8", "replace") if body else ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text[:200]
        if isinstance(data, Mapping):
            error = data.get("error")
            if isinstance(error, Mapping):
                return str(error.get("message") or error)[:200]
            if error:
                return str(error)[:200]
            return str(data.get("message") or "")[:200]
        return text[:200]

    # ---------------------------------------------------------------- 异步接口

    async def athink(
        self,
        messages: Iterable[Message | Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """异步流式调用，逐段产出文本增量。"""
        self._async_used = True
        async for chunk in self._stream_impl(
            self._client, messages, temperature=temperature, max_tokens=max_tokens
        ):
            yield chunk

    async def ainvoke(
        self,
        messages: Iterable[Message | Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """异步非流式调用，返回完整文本。"""
        self._async_used = True
        return await self._invoke_impl(
            self._client, messages, temperature=temperature, max_tokens=max_tokens
        )

    async def _stream_impl(
        self,
        client: httpx.AsyncClient,
        messages: Iterable[Message | Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式请求实现；``client`` 由调用方按事件循环归属传入。"""
        payload = self._build_payload(
            messages, stream=True, temperature=temperature, max_tokens=max_tokens
        )
        endpoint = self._endpoint()
        logger.debug("流式调用 %s: model=%s url=%s", self.provider, self.model, endpoint)
        try:
            async with client.stream(
                "POST", endpoint, json=payload, headers=self._headers()
            ) as response:
                await self._raise_for_status(response)
                if _STREAM_CONTENT_TYPE in response.headers.get("content-type", ""):
                    async for line in response.aiter_lines():
                        chunk = self._parse_stream_line(line)
                        if chunk:
                            yield chunk
                    return
                # 部分本地服务忽略 stream=true：回退到一次性解析响应体。
                logger.debug("provider 未返回 SSE，按非流式响应解析 url=%s", endpoint)
                for chunk in self._parse_body(await response.aread()):
                    yield chunk
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                "LLM 网络请求失败", context=self._context(url=endpoint, error=type(exc).__name__)
            ) from exc

    async def _invoke_impl(
        self,
        client: httpx.AsyncClient,
        messages: Iterable[Message | Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """非流式请求实现；``client`` 由调用方按事件循环归属传入。"""
        payload = self._build_payload(
            messages, stream=False, temperature=temperature, max_tokens=max_tokens
        )
        endpoint = self._endpoint()
        logger.debug("非流式调用 %s: model=%s url=%s", self.provider, self.model, endpoint)
        try:
            response = await client.post(endpoint, json=payload, headers=self._headers())
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                "LLM 网络请求失败", context=self._context(url=endpoint, error=type(exc).__name__)
            ) from exc
        await self._raise_for_status(response)
        return "".join(self._parse_body(response.content))

    # ---------------------------------------------------------------- 同步接口

    def _sync_resources(self) -> tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]:
        """同步接口专用的（事件循环, 客户端）对，按需创建并长期复用。

        httpx 的连接池绑定在创建连接的那个事件循环上。若每次同步调用都用
        ``asyncio.run`` 新建循环，池中的连接就会属于已关闭的循环，第二次调用即
        ``RuntimeError: Event loop is closed``。因此同步接口独占一个循环与客户端。
        """
        if self._sync_loop is None or self._sync_loop.is_closed():
            self._sync_loop = asyncio.new_event_loop()
        if self._sync_client is None:
            self._sync_client = httpx.AsyncClient(timeout=self.timeout, transport=self._transport)
        return self._sync_loop, self._sync_client

    def think(
        self,
        messages: Iterable[Message | Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """同步流式调用（教程同签名）。

        在同步专用事件循环中逐步驱动流式实现，因此是真实增量输出，而非先收全再吐。
        """
        ensure_no_running_loop()
        loop, client = self._sync_resources()
        agen = cast(
            "AsyncGenerator[str, None]",
            self._stream_impl(client, messages, temperature=temperature, max_tokens=max_tokens),
        )
        exhausted = False
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    exhausted = True
                    break
                yield chunk
        finally:
            if not exhausted:  # 调用方提前中断：关闭本次响应，但保留循环与连接池
                loop.run_until_complete(agen.aclose())

    def invoke(
        self,
        messages: Iterable[Message | Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """同步非流式调用（教程同签名）。"""
        ensure_no_running_loop()
        loop, client = self._sync_resources()
        return loop.run_until_complete(
            self._invoke_impl(client, messages, temperature=temperature, max_tokens=max_tokens)
        )

    def stream_invoke(
        self,
        messages: Iterable[Message | Mapping[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """``think`` 的别名，教程中用于向后兼容。"""
        yield from self.think(messages, temperature=temperature, max_tokens=max_tokens)

    # ---------------------------------------------------------------- 生命周期

    async def aclose(self) -> None:
        """关闭异步连接池；应在协程内调用（或交给 ``async with``）。"""
        await self._client.aclose()
        if self._sync_client is not None:
            logger.warning(
                "同步接口的连接池属于另一个事件循环，无法在此关闭；"
                "请在同步上下文调用 close()，或不要混用同步与异步接口"
            )

    def close(self) -> None:
        """同步关闭连接池，并回收同步接口独占的事件循环。"""
        ensure_no_running_loop()
        if self._sync_client is not None:
            loop, _ = self._sync_resources()
            loop.run_until_complete(self._sync_client.aclose())
            self._sync_client = None
        if self._sync_loop is not None:
            if not self._sync_loop.is_closed():
                self._sync_loop.close()
            self._sync_loop = None
        if self._async_used:
            logger.warning(
                "异步客户端已在事件循环中使用过，其连接池只能在创建它的事件循环内关闭"
                "（await aclose() 或退出 async with）"
            )
            return
        if not self._client.is_closed:
            # 异步客户端从未真正建立过连接，用一个临时循环关闭是安全的。
            temp_loop = asyncio.new_event_loop()
            try:
                temp_loop.run_until_complete(self._client.aclose())
            finally:
                temp_loop.close()

    def __enter__(self) -> HelloAgentsLLM:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    async def __aenter__(self) -> HelloAgentsLLM:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return (
            f"HelloAgentsLLM(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r})"
        )
