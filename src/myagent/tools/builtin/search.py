"""内置工具：多源网络搜索（Tavily / SerpAPI）。

教程定位：第七章 7.5.3「多源搜索」的 ``MyAdvancedSearchTool``。
与教程的差异（见 docs/deviations.md D22）：

- 教程依赖 ``tavily`` / ``google-search-results`` 两个第三方 SDK，并用 ``print`` 打日志、
  用 ``except Exception as e: return f"❌ {e}"`` 吞掉一切错误；此处只依赖运行时已有的
  ``httpx``，直接调 REST 接口，日志走 ``logging``，错误抛 :class:`ToolException`
  （hybrid 模式在 context 里携带每个后端的失败原因 ``attempts``）；
- 教程把搜索方法经 ``register_function`` 暴露，此处是 :class:`Tool` 子类，自带 ``query`` / ``limit``
  参数声明与校验；
- 使用**同步** ``httpx.Client`` 并在实例内复用连接池：异步链路由 :meth:`Tool.arun` 的线程池承担，
  避免连接池绑定到某个临时事件循环。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any, Final

import httpx

from ...core.exceptions import ToolException
from ..base import Tool, ToolParameter

__all__ = ["SearchTool"]

logger = logging.getLogger(__name__)

_TAVILY_URL: Final[str] = "https://api.tavily.com/search"
_SERPAPI_URL: Final[str] = "https://serpapi.com/search.json"
_BACKENDS: Final[tuple[str, ...]] = ("hybrid", "tavily", "serpapi")
_CONTENT_LIMIT: Final[int] = 200
_EMPTY_RESULT: Final[str] = "未找到相关结果。"


def _as_text(value: Any) -> str:
    """把响应对应字段规整成去空白的文本；非字符串一律视为空。"""
    return value.strip() if isinstance(value, str) else ""


def _truncate(text: str, limit: int = _CONTENT_LIMIT) -> str:
    """把长文本截断到 ``limit`` 字符，超出时补省略号。"""
    return text if len(text) <= limit else f"{text[:limit]}…"


class SearchTool(Tool):
    """多源搜索工具：Tavily 优先，失败可降级 SerpAPI。

    Args:
        backend: ``hybrid``（Tavily 优先、失败降级 SerpAPI）/ ``tavily`` / ``serpapi``。
        tavily_key: Tavily 密钥；缺省时读 ``env`` 中的 ``TAVILY_API_KEY``。
        serpapi_key: SerpAPI 密钥；缺省时读 ``env`` 中的 ``SERPAPI_API_KEY``。
        timeout: 内部自建 ``httpx.Client`` 的超时（秒）。
        env: 读取密钥的来源；缺省为 ``os.environ``，便于测试注入。
        client: 注入的 ``httpx.Client``（测试用 ``httpx.MockTransport``）；缺省时惰性自建并复用。
    """

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: str | None = None,
        serpapi_key: str | None = None,
        *,
        timeout: float = 15.0,
        env: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            name="search",
            description="多源网络搜索工具，支持 Tavily 与 SerpAPI，返回答案摘要与前若干条结果。",
            parameters=[
                ToolParameter("query", "string", "搜索关键词", required=True),
                ToolParameter("limit", "integer", "返回结果条数", required=False, default=3),
            ],
            tags=("web", "search"),
        )
        if backend not in _BACKENDS:
            raise ToolException(
                "不支持的搜索后端",
                context={"backend": backend, "supported": list(_BACKENDS)},
            )
        self.backend = backend
        source: Mapping[str, str] = env if env is not None else os.environ
        self._tavily_key = tavily_key or source.get("TAVILY_API_KEY")
        self._serpapi_key = serpapi_key or source.get("SERPAPI_API_KEY")
        self._timeout = timeout
        self._client = client

    # ------------------------------------------------------------------ 可用性

    @property
    def available_backends(self) -> tuple[str, ...]:
        """按密钥是否齐备，返回当前可用的后端名（顺序即优先级）。"""
        names = []
        if self._tavily_key:
            names.append("tavily")
        if self._serpapi_key:
            names.append("serpapi")
        return tuple(names)

    # ------------------------------------------------------------------ 执行

    def run(self, parameters: Mapping[str, Any]) -> str:
        """按后端配置执行搜索并返回面向模型的中文文本。"""
        query = str(parameters.get("query", "")).strip()
        if not query:
            raise ToolException("搜索关键词不能为空", context={"query": parameters.get("query")})
        limit = self._normalize_limit(parameters.get("limit", 3))
        if self.backend == "tavily":
            return self._search_tavily(query, limit)
        if self.backend == "serpapi":
            return self._search_serpapi(query, limit)
        return self._search_hybrid(query, limit)

    def _search_hybrid(self, query: str, limit: int) -> str:
        """Tavily 优先；失败则降级 SerpAPI；全都失败或无密钥时抛 :class:`ToolException`。"""
        attempts: list[str] = []
        if self._tavily_key:
            try:
                return self._search_tavily(query, limit)
            except ToolException as exc:
                logger.warning("Tavily 搜索失败，降级 SerpAPI：%s", exc)
                attempts.append(f"tavily: {exc.message}")
        if self._serpapi_key:
            try:
                return self._search_serpapi(query, limit)
            except ToolException as exc:
                logger.warning("SerpAPI 搜索失败：%s", exc)
                attempts.append(f"serpapi: {exc.message}")
        if not attempts:
            raise ToolException(
                "未配置任何搜索后端密钥，请设置 TAVILY_API_KEY 或 SERPAPI_API_KEY",
                context={"query": query, "hint": "请设置 TAVILY_API_KEY 或 SERPAPI_API_KEY"},
            )
        raise ToolException(
            "所有搜索后端均失败",
            context={"query": query, "attempts": attempts},
        )

    # ------------------------------------------------------------------ 后端

    def _search_tavily(self, query: str, limit: int) -> str:
        if not self._tavily_key:
            raise ToolException(
                "未配置 Tavily 密钥，请设置 TAVILY_API_KEY",
                context={"query": query, "hint": "请设置 TAVILY_API_KEY"},
            )
        payload = {
            "api_key": self._tavily_key,
            "query": query,
            "max_results": limit,
            "include_answer": True,
        }
        data = self._request("POST", _TAVILY_URL, "tavily", query, json=payload)
        return self._format_tavily(data, limit)

    def _search_serpapi(self, query: str, limit: int) -> str:
        if not self._serpapi_key:
            raise ToolException(
                "未配置 SerpAPI 密钥，请设置 SERPAPI_API_KEY",
                context={"query": query, "hint": "请设置 SERPAPI_API_KEY"},
            )
        params = {
            "engine": "google",
            "q": query,
            "api_key": self._serpapi_key,
            "num": limit,
        }
        data = self._request("GET", _SERPAPI_URL, "serpapi", query, params=params)
        return self._format_serpapi(data, limit)

    # ------------------------------------------------------------------ HTTP

    def _request(
        self,
        method: str,
        url: str,
        backend: str,
        query: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """发起一次搜索请求并返回解析后的 JSON；网络/状态/解析错误都转成 ToolException。"""
        client = self._http()
        try:
            response = client.request(method, url, json=json, params=params)
            if response.status_code >= 400:
                raise ToolException(
                    "搜索后端返回错误状态",
                    context={
                        "backend": backend,
                        "query": query,
                        "status": response.status_code,
                    },
                    retryable=response.status_code >= 500,
                )
            data = response.json()
        except ToolException:
            raise
        except httpx.HTTPError as exc:
            raise ToolException(
                "搜索请求失败",
                context={"backend": backend, "query": query, "detail": str(exc)},
                retryable=True,
            ) from exc
        except ValueError as exc:
            raise ToolException(
                "搜索后端返回了非法 JSON",
                context={"backend": backend, "query": query},
            ) from exc
        if not isinstance(data, Mapping):
            raise ToolException(
                "搜索后端返回了非对象 JSON",
                context={"backend": backend, "query": query},
            )
        return data

    def _http(self) -> httpx.Client:
        """惰性创建并复用内部客户端；注入的客户端优先。"""
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    # ------------------------------------------------------------------ 格式化

    def _format_tavily(self, data: Mapping[str, Any], limit: int) -> str:
        lines: list[str] = []
        answer = _as_text(data.get("answer"))
        if answer:
            lines.append(f"答案：{answer}")
        results = data.get("results")
        if isinstance(results, list):
            for index, item in enumerate(results[:limit], 1):
                if not isinstance(item, Mapping):
                    continue
                title = _as_text(item.get("title"))
                url = _as_text(item.get("url"))
                content = _truncate(_as_text(item.get("content")))
                lines.append(f"{index}. {title}\n   链接：{url}\n   摘要：{content}")
        return "\n".join(lines) if lines else _EMPTY_RESULT

    def _format_serpapi(self, data: Mapping[str, Any], limit: int) -> str:
        lines: list[str] = []
        answer = self._serpapi_answer(data)
        if answer:
            lines.append(f"答案：{answer}")
        results = data.get("organic_results")
        if isinstance(results, list):
            for index, item in enumerate(results[:limit], 1):
                if not isinstance(item, Mapping):
                    continue
                title = _as_text(item.get("title"))
                link = _as_text(item.get("link"))
                snippet = _truncate(_as_text(item.get("snippet")))
                lines.append(f"{index}. {title}\n   链接：{link}\n   摘要：{snippet}")
        return "\n".join(lines) if lines else _EMPTY_RESULT

    @staticmethod
    def _serpapi_answer(data: Mapping[str, Any]) -> str:
        """按 ``answer_box.answer`` → ``answer_box_list`` → ``knowledge_graph.description`` 取答案。"""
        answer_box = data.get("answer_box")
        if isinstance(answer_box, Mapping):
            for key in ("answer", "snippet"):
                text = _as_text(answer_box.get(key))
                if text:
                    return text
        box_list = data.get("answer_box_list")
        if isinstance(box_list, list):
            for item in box_list:
                if isinstance(item, Mapping):
                    text = _as_text(item.get("answer")) or _as_text(item.get("snippet"))
                else:
                    text = _as_text(item)
                if text:
                    return text
        knowledge_graph = data.get("knowledge_graph")
        if isinstance(knowledge_graph, Mapping):
            return _as_text(knowledge_graph.get("description"))
        return ""

    # ------------------------------------------------------------------ 校验

    @staticmethod
    def _normalize_limit(value: Any) -> int:
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ToolException("limit 必须是整数", context={"limit": value}) from exc
        if limit < 1:
            raise ToolException("limit 必须大于 0", context={"limit": limit})
        return limit
