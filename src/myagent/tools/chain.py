"""工具链：把多个工具按顺序串联执行。

教程定位：第七章 7.5.4「工具系统的高级特性」中的 ``ToolChain`` 与 ``ToolChainManager``。
教程的 ``code/chapter7/`` 没有对应脚本（见 docs/deviations.md G1），因此以正文代码块为准，
并按本项目约定修正以下偏离（见 docs/deviations.md D23）：

- 步骤用不可变的 :class:`ToolStep` 与只读属性 :attr:`ToolChain.steps` 暴露，取代教程里可被
  外部随意改写的 ``List[Dict[str, Any]]``；``output_key`` 缺省命名保持教程的
  ``step_{index}_result`` 语义不变；
- 模板变量缺失时教程返回 ``"❌ ..."`` 错误字符串，这里改抛 :class:`ToolException`
  （context 含 ``chain`` / ``step`` / ``missing``），与「禁止吞异常转字符串」一致；
- 教程用 ``print`` 打印进度，这里改用 :mod:`logging`；
- 教程只有同步执行，这里新增 :meth:`ToolChain.aexecute` /
  :meth:`ToolChainManager.aexecute_chain`；同步与异步共用同一套「渲染 + 回填 + 日志」骨架
  （:meth:`ToolChain._pipeline`），避免复制两份步进循环；
- 补上教程缺失的 :meth:`ToolChainManager.unregister_chain` / :meth:`ToolChainManager.get_chain`。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Generator, Mapping
from dataclasses import dataclass
from typing import Any

from ..core.exceptions import ToolException
from .registry import ToolRegistry

__all__ = ["ToolChain", "ToolChainManager", "ToolStep"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolStep:
    """工具链中的一步。

    Args:
        tool_name: 这一步调用的工具名（注册表中的键）。
        input_template: 输入模板，经 ``str.format`` 渲染；可引用 ``input`` 以及之前各步写入的
            ``output_key``。
        output_key: 这一步输出写回上下文的键名，供后续步骤引用。
    """

    tool_name: str
    input_template: str
    output_key: str


class ToolChain:
    """一串按序执行的工具步骤。

    每一步的输入由模板渲染得到，输出以 ``output_key`` 写回执行上下文，供后续步骤引用。
    执行时只经 :meth:`ToolRegistry.execute_tool`（同步）或
    :meth:`ToolRegistry.aexecute_tool`（异步）这唯一对外路径。
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._steps: list[ToolStep] = []

    @property
    def steps(self) -> tuple[ToolStep, ...]:
        """返回步骤的只读快照；调用方改动它不影响链本身。"""
        return tuple(self._steps)

    def add_step(
        self, tool_name: str, input_template: str, output_key: str | None = None
    ) -> ToolStep:
        """追加一步并返回它。

        Args:
            tool_name: 工具名。
            input_template: 输入模板。
            output_key: 输出键名；缺省为 ``step_{当前步序号}_result``（第 0 步 →
                ``step_0_result``），与教程一致。
        """
        step = ToolStep(
            tool_name=tool_name,
            input_template=input_template,
            output_key=output_key if output_key is not None else f"step_{len(self._steps)}_result",
        )
        self._steps.append(step)
        return step

    # ------------------------------------------------------------------ 执行

    def execute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """同步执行整条链，返回最后一步的输出。

        Args:
            registry: 提供工具的工具注册表。
            initial_input: 初始输入，渲染时以 ``input`` 暴露。
            context: 额外的渲染变量；内部会复制，调用方字典不会被修改。

        Raises:
            ToolException: 链为空、模板变量缺失、或工具执行失败时。
        """
        self._require_steps()
        scope = self._scope(initial_input, context)
        result = self._drive(scope, registry.execute_tool)
        logger.info("工具链 %r 执行完成（%d 步）", self.name, len(self._steps))
        return result

    async def aexecute(
        self,
        registry: ToolRegistry,
        initial_input: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """异步执行整条链，返回最后一步的输出；语义与 :meth:`execute` 一致。"""
        self._require_steps()
        scope = self._scope(initial_input, context)
        result = await self._adrive(scope, registry.aexecute_tool)
        logger.info("工具链 %r 执行完成（%d 步）", self.name, len(self._steps))
        return result

    # ------------------------------------------------------------------ 内部

    def _pipeline(self, scope: dict[str, Any]) -> Generator[tuple[int, ToolStep, str], str, None]:
        """同步/异步共用的步进骨架。

        依次渲染每一步的输入并 ``yield (序号, 步骤, 渲染后的输入)``；调用方执行完工具后用
        ``send(结果)`` 把结果交回，生成器负责写回 ``scope`` 并记录进度日志。
        """
        for index, step in enumerate(self._steps, start=1):
            rendered = self._render(index, step, scope)
            logger.debug("工具链 %r 第 %d 步：%s <- %r", self.name, index, step.tool_name, rendered)
            result = yield index, step, rendered
            scope[step.output_key] = result
            logger.debug("工具链 %r 第 %d 步完成，输出 %d 字符", self.name, index, len(result))

    def _drive(self, scope: dict[str, Any], invoke: Callable[[str, str], str]) -> str:
        """用同步 ``invoke`` 驱动 :meth:`_pipeline`。"""
        pipeline = self._pipeline(scope)
        _, step, rendered = next(pipeline)
        while True:
            result = invoke(step.tool_name, rendered)
            try:
                _, step, rendered = pipeline.send(result)
            except StopIteration:
                break
        return scope[self._steps[-1].output_key]

    async def _adrive(
        self, scope: dict[str, Any], invoke: Callable[[str, str], Awaitable[str]]
    ) -> str:
        """用异步 ``invoke`` 驱动 :meth:`_pipeline`。"""
        pipeline = self._pipeline(scope)
        _, step, rendered = next(pipeline)
        while True:
            result = await invoke(step.tool_name, rendered)
            try:
                _, step, rendered = pipeline.send(result)
            except StopIteration:
                break
        return scope[self._steps[-1].output_key]

    def _render(self, index: int, step: ToolStep, scope: Mapping[str, Any]) -> str:
        """渲染某一步的输入；模板变量缺失时抛 :class:`ToolException`。"""
        try:
            return step.input_template.format(**scope)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else "?"
            raise ToolException(
                "工具链模板变量缺失",
                context={"chain": self.name, "step": index, "missing": missing},
            ) from exc

    def _scope(self, initial_input: str, context: Mapping[str, Any] | None) -> dict[str, Any]:
        """构造渲染上下文：调用方 context 的副本 + ``input``。"""
        scope: dict[str, Any] = dict(context or {})
        scope["input"] = initial_input
        return scope

    def _require_steps(self) -> None:
        if not self._steps:
            raise ToolException("工具链没有任何步骤", context={"chain": self.name})


class ToolChainManager:
    """按名注册与调度多条 :class:`ToolChain`。"""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._chains: dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain) -> None:
        """注册工具链；同名会被覆盖并记一条警告。"""
        if chain.name in self._chains:
            logger.warning("工具链 %s 已存在，将被覆盖", chain.name)
        self._chains[chain.name] = chain
        logger.debug("工具链 %s 已注册", chain.name)

    def unregister_chain(self, name: str) -> bool:
        """注销工具链；返回是否真的移除了它。"""
        return self._chains.pop(name, None) is not None

    def get_chain(self, name: str) -> ToolChain | None:
        """按名取链；不存在返回 ``None``。"""
        return self._chains.get(name)

    def list_chains(self) -> list[str]:
        """列出所有工具链名（注册顺序）。"""
        return list(self._chains)

    def execute_chain(
        self,
        name: str,
        input_data: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """同步执行指定工具链。

        Raises:
            ToolException: 链不存在，或链自身执行失败时。
        """
        return self._require_chain(name).execute(self.registry, input_data, context)

    async def aexecute_chain(
        self,
        name: str,
        input_data: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """异步执行指定工具链；语义与 :meth:`execute_chain` 一致。"""
        return await self._require_chain(name).aexecute(self.registry, input_data, context)

    def _require_chain(self, name: str) -> ToolChain:
        chain = self._chains.get(name)
        if chain is None:
            raise ToolException(
                "工具链未注册",
                context={"chain": name, "available": self.list_chains()},
            )
        return chain
