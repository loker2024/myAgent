"""工具系统基类：``Tool``、``ToolParameter`` 与函数适配器 ``FunctionTool``。

教程定位：第七章 7.5.1「工具基类与注册机制设计」的 ``Tool`` / ``ToolParameter``。
与教程的差异（见 docs/deviations.md D19、D20、D21）：

- ``ToolParameter`` 用 ``@dataclass`` 取代 ``pydantic.BaseModel``（与 D2 同一原因）；
- 参数校验由工具自己声明、由 :meth:`Tool.validate` 统一执行，取代教程里硬编码在 Agent 中的
  ``_parse_tool_parameters``（教程的 ``if tool_name == 'calculator'`` 分支）；
- 新增 :meth:`Tool.arun`：默认在线程池里调用 :meth:`Tool.run`，让同步工具也能被异步链路并发调用。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

from ..core.exceptions import ToolException

__all__ = ["FunctionTool", "Tool", "ToolParameter", "ToolType"]

logger = logging.getLogger(__name__)

ToolType = Literal["string", "integer", "number", "boolean", "array", "object"]

_ANNOTATION_TO_TYPE: Final[dict[type, ToolType]] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
_TYPE_NAMES: Final[dict[str, ToolType]] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "sequence": "array",
    "set": "array",
    "tuple": "array",
    "dict": "object",
    "mapping": "object",
}
_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


@dataclass
class ToolParameter:
    """一个工具参数的声明。

    Args:
        name: 参数名，模型在提示词/JSON 里书写的键。
        type: JSON Schema 意义上的类型。
        description: 参数说明，会进入提示词与 function calling schema。
        required: 是否必填；``False`` 时必须给 ``default``。
        default: 缺省值；``None`` 表示无默认值。
    """

    name: str
    type: ToolType = "string"
    description: str = ""
    required: bool = True
    default: Any = None

    def to_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 中该参数的 schema 片段。"""
        described = self.description
        if self.default is not None:
            suffix = f"默认: {self.default}"
            described = f"{described}（{suffix}）" if described else suffix
        schema: dict[str, Any] = {"type": self.type, "description": described}
        if self.type == "array":
            schema["items"] = {"type": "string"}
        return schema

    def __post_init__(self) -> None:
        if self.required and self.default is not None:
            raise ToolException(
                "参数声明矛盾：required=True 与 default 不能同时给出",
                context={"parameter": self.name, "default": self.default},
            )


def _annotation_to_type(annotation: object) -> ToolType:
    """把 Python 注解折算成 JSON Schema 类型；未知注解按 ``string`` 处理。

    本项目与测试都启用了 ``from __future__ import annotations``，此时
    ``inspect.signature`` 拿到的是**字符串**注解（如 ``"int"``、``"list[str]"``、
    ``"int | None"``），因此字符串形态也要能识别。
    """
    if isinstance(annotation, str):
        return _type_name_to_type(annotation)
    if isinstance(annotation, type) and annotation in _ANNOTATION_TO_TYPE:
        return _ANNOTATION_TO_TYPE[annotation]
    return "string"


def _type_name_to_type(name: str) -> ToolType:
    text = name.strip()
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1]
    if "|" in text:
        parts = [
            item.strip() for item in text.split("|") if item.strip() not in {"None", "NoneType"}
        ]
        text = parts[0] if parts else text
    base = text.split("[")[0].strip().rsplit(".", 1)[-1]
    return _TYPE_NAMES.get(base, "string")


class Tool(ABC):
    """工具抽象基类。

    Args:
        name: 工具名。注册表中的唯一键，也是模型在提示词里书写的标识符。
        description: 工具描述，进入系统提示词与 function calling schema。
        parameters: 参数声明；缺省为空列表（无参工具）。
        tags: 分类标签，便于检索与展示。

    子类只需实现 :meth:`run`（同步的唯一真实实现）。异步链路默认经 :meth:`arun`
    在线程池中调用它，因此同步网络/CPU 工具不会阻塞事件循环。
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Sequence[ToolParameter] | None = None,
        *,
        tags: Iterable[str] = (),
    ) -> None:
        self.name = name
        self.description = description
        self.tags: tuple[str, ...] = tuple(tags)
        self._parameters: list[ToolParameter] = list(parameters or [])

    # ------------------------------------------------------------------ 声明

    def get_parameters(self) -> list[ToolParameter]:
        """返回参数声明的副本，调用方修改副本不影响工具。"""
        return list(self._parameters)

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI function calling 的 ``tools`` 元素格式。"""
        properties = {spec.name: spec.to_schema() for spec in self._parameters}
        required = [spec.name for spec in self._parameters if spec.required]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    # ------------------------------------------------------------------ 执行

    @abstractmethod
    def run(self, parameters: Mapping[str, Any]) -> str:
        """执行工具并返回文本结果。

        Args:
            parameters: 已通过 :meth:`validate` 校验与规整的参数。

        Returns:
            面向模型的结果文本。
        """
        raise NotImplementedError

    async def arun(self, parameters: Mapping[str, Any]) -> str:
        """异步执行工具；默认把 :meth:`run` 放到线程池，避免阻塞事件循环。"""
        return await asyncio.to_thread(self.run, dict(parameters))

    # ------------------------------------------------------------------ 校验

    def validate(self, parameters: Mapping[str, Any]) -> dict[str, Any]:
        """校验并规整参数。

        Returns:
            只含已声明参数、且缺失的可选参数已填默认值的字典。

        Raises:
            ToolException: 出现未声明参数、缺少必填参数，或类型无法转换时。
        """
        declared = {spec.name: spec for spec in self._parameters}
        unknown = sorted(name for name in parameters if name not in declared)
        if unknown:
            raise ToolException(
                "工具收到未声明的参数",
                context={"tool": self.name, "unknown": unknown, "declared": sorted(declared)},
            )
        resolved: dict[str, Any] = {}
        for name, spec in declared.items():
            if name in parameters and parameters[name] is not None:
                resolved[name] = self._coerce(name, parameters[name], spec)
            elif spec.default is not None:
                resolved[name] = spec.default
            elif spec.required:
                raise ToolException(
                    "工具缺少必填参数",
                    context={"tool": self.name, "missing": name, "declared": sorted(declared)},
                )
        return resolved

    def _coerce(self, name: str, value: Any, spec: ToolParameter) -> Any:
        """按声明的类型转换参数；文本协议传来的都是字符串，这里负责折算。"""
        want = spec.type
        try:
            if want == "string":
                return value if isinstance(value, str) else str(value)
            if want == "integer":
                return (
                    value if isinstance(value, int) and not isinstance(value, bool) else int(value)
                )
            if want == "number":
                return value if isinstance(value, float) else float(value)
            if want == "boolean":
                if isinstance(value, bool):
                    return value
                lowered = str(value).strip().lower()
                if lowered in _TRUTHY:
                    return True
                if lowered in _FALSY:
                    return False
                raise ValueError(f"无法把 {value!r} 解释为布尔值")
            if want == "array":
                if isinstance(value, (list, tuple)):
                    return list(value)
                text = str(value).strip()
                if text.startswith("["):
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return parsed
                    raise ValueError("JSON 不是数组")
                return [item.strip() for item in text.split(",") if item.strip()]
            if want == "object":
                if isinstance(value, Mapping):
                    return dict(value)
                parsed = json.loads(str(value))
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("JSON 不是对象")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ToolException(
                "工具参数类型不匹配",
                context={"tool": self.name, "parameter": name, "expected": want, "value": value},
            ) from exc
        raise ToolException(
            "未知的参数类型声明",
            context={"tool": self.name, "parameter": name, "type": want},
        )

    # ------------------------------------------------------------------ 展示

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, parameters={len(self._parameters)})"


class FunctionTool(Tool):
    """把一个普通函数适配成 :class:`Tool`。

    ``ToolRegistry.register_function`` 用它落地，从而让「函数工具」与「类工具」在
    执行、校验、schema 生成上走完全相同的路径（见 docs/deviations.md D6）。

    Args:
        name: 工具名。
        description: 工具描述。
        func: 被适配的函数；参数声明由 ``inspect.signature`` 推断。
        parameter_name: 当函数有多个参数、而调用方只想暴露其中一个时显式指定。

    Raises:
        ToolException: 函数没有可用的位置参数，或 ``parameter_name`` 不存在时。
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., str],
        *,
        parameter_name: str | None = None,
    ) -> None:
        signature = inspect.signature(func)
        positional = [
            item
            for item in signature.parameters.values()
            if item.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if parameter_name is not None:
            selected = [item for item in positional if item.name == parameter_name]
            if not selected:
                raise ToolException(
                    "函数没有名为该值的参数",
                    context={"function": name, "parameter": parameter_name},
                )
            positional = selected
        if not positional:
            raise ToolException(
                "被注册的函数必须至少有一个位置参数",
                context={"function": name, "signature": str(signature)},
            )
        parameters = [
            ToolParameter(
                name=item.name,
                type=_annotation_to_type(
                    None if item.annotation is inspect.Parameter.empty else item.annotation
                ),
                description=f"参数 {item.name}",
                required=item.default is inspect.Parameter.empty,
                default=None if item.default is inspect.Parameter.empty else item.default,
            )
            for item in positional
        ]
        super().__init__(name, description, parameters)
        self._func = func
        self._argument_names = [item.name for item in positional]

    def run(self, parameters: Mapping[str, Any]) -> str:
        arguments = {name: parameters[name] for name in self._argument_names if name in parameters}
        return self._func(**arguments)

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name!r}, arguments={self._argument_names})"
