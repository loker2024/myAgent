"""内置工具：安全的数学计算器。

教程定位：第七章 7.5.2「calculator 工具」的 ``my_calculator``。
与教程的差异（见 docs/deviations.md D21）：

- 教程把表达式交给 ``ast`` 直接分派 ``operator`` 且只覆盖 ``+ - * /`` 与 ``sqrt``，
  用裸 ``except:`` 把一切错误吞成字符串；此处是**白名单 AST 求值**：只放行显式列出的
  节点类型、函数与常量，任何越界节点一律抛 :class:`ToolException`（携带表达式与节点类型）；
- 教程经 ``register_function`` 注册函数，此处做成 :class:`Tool` 子类，自带参数声明与校验；
- 新增规模保护（幂指数、阶乘入参）、复数结果拒绝与整型 float 归一，避免挂死与不可读输出。
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any, Final

from ...core.exceptions import ToolException
from ..base import Tool, ToolParameter

__all__ = ["CalculatorTool"]

logger = logging.getLogger(__name__)

# 幂指数绝对值上限：防止 ``2 ** 10**9`` 这类表达式挂死或吃光内存
_MAX_EXPONENT: Final[int] = 10000
# 阶乘入参上限：``factorial(10**6)`` 会瞬间撑爆内存
_MAX_FACTORIAL: Final[int] = 1000

_BIN_OPS: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Final[dict[type[ast.unaryop], Callable[[Any], Any]]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "fabs": math.fabs,
    "floor": math.floor,
    "ceil": math.ceil,
    "gcd": math.gcd,
    "factorial": math.factorial,
    "pow": math.pow,
}
_CONSTANTS: Final[dict[str, float]] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}
_MATH_NAMESPACE: Final[str] = "math"
_ARITHMETIC_ERRORS: Final[tuple[type[Exception], ...]] = (
    ZeroDivisionError,
    ValueError,
    OverflowError,
    TypeError,
)


class CalculatorTool(Tool):
    """对数学表达式做白名单 AST 求值的计算器工具。

    只接受数字常量、四则/幂/取模/整除、一元正负号、白名单函数与常量，以及
    ``math.<白名单成员>`` 形式；其余节点（比较、布尔、下标、lambda、变量名、任意属性访问…）
    都会被拒绝，因此不存在通过 ``eval`` 注入任意代码的路径。
    """

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description=(
                "计算数学表达式，支持四则运算、幂、取模、括号，"
                "以及 sqrt/sin/cos/log/factorial 等常用数学函数。"
            ),
            parameters=[
                ToolParameter(
                    "expression",
                    "string",
                    "要计算的数学表达式，例如 (15 + 3) * 2 或 sqrt(16)",
                    required=True,
                )
            ],
            tags=("math",),
        )

    # ------------------------------------------------------------------ 执行

    def run(self, parameters: Mapping[str, Any]) -> str:
        """求值 ``expression`` 并返回结果的文本形式。"""
        raw = parameters.get("expression", "")
        expression = str(raw).strip()
        if not expression:
            raise ToolException("计算表达式不能为空", context={"expression": raw})
        node = self._parse(expression)
        result = self._eval_node(node, expression)
        return self._format(result, expression)

    # ------------------------------------------------------------------ 求值

    def _parse(self, expression: str) -> ast.expr:
        """把表达式解析成 AST 表达式节点；语法错误转成 :class:`ToolException`。"""
        try:
            return ast.parse(expression, mode="eval").body
        except SyntaxError as exc:
            raise ToolException(
                "表达式语法错误",
                context={"expression": expression, "detail": str(exc)},
            ) from exc

    def _eval_node(self, node: ast.AST, expression: str) -> Any:
        """递归求值单个 AST 节点；非白名单节点一律拒绝。"""
        if isinstance(node, ast.Constant):
            return self._eval_constant(node, expression)
        if isinstance(node, ast.BinOp):
            return self._eval_binop(node, expression)
        if isinstance(node, ast.UnaryOp):
            return self._eval_unaryop(node, expression)
        if isinstance(node, ast.Call):
            return self._eval_call(node, expression)
        if isinstance(node, ast.Name):
            return self._eval_name(node, expression)
        if isinstance(node, ast.Attribute):
            return self._eval_attribute(node, expression)
        raise ToolException(
            "表达式包含不支持的语法",
            context={"expression": expression, "node": type(node).__name__},
        )

    def _eval_constant(self, node: ast.Constant, expression: str) -> int | float:
        """只接受整数与浮点数字面量；字符串/复数/None 等一律拒绝。"""
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolException(
                "表达式包含不支持的常量",
                context={"expression": expression, "node": type(node).__name__, "value": value},
            )
        return value

    def _eval_binop(self, node: ast.BinOp, expression: str) -> Any:
        operation = _BIN_OPS.get(type(node.op))
        if operation is None:
            raise ToolException(
                "表达式包含不支持的运算符",
                context={"expression": expression, "node": type(node.op).__name__},
            )
        left = self._eval_node(node.left, expression)
        right = self._eval_node(node.right, expression)
        if isinstance(node.op, ast.Pow):
            self._check_exponent(right, expression)
        return self._apply(operation, left, right, expression)

    def _eval_unaryop(self, node: ast.UnaryOp, expression: str) -> Any:
        operation = _UNARY_OPS.get(type(node.op))
        if operation is None:
            raise ToolException(
                "表达式包含不支持的一元运算符",
                context={"expression": expression, "node": type(node.op).__name__},
            )
        operand = self._eval_node(node.operand, expression)
        try:
            return operation(operand)
        except _ARITHMETIC_ERRORS as exc:
            raise ToolException(
                "计算失败",
                context={"expression": expression, "detail": str(exc)},
            ) from exc

    def _eval_call(self, node: ast.Call, expression: str) -> Any:
        if node.keywords:
            raise ToolException(
                "函数调用不支持关键字参数",
                context={"expression": expression, "node": type(node).__name__},
            )
        name, func = self._resolve_callable(node.func, expression)
        arguments = [self._eval_node(arg, expression) for arg in node.args]
        if name == "factorial" and arguments:
            self._check_factorial(arguments[0], expression)
        try:
            return func(*arguments)
        except _ARITHMETIC_ERRORS as exc:
            raise ToolException(
                "函数计算失败",
                context={"expression": expression, "function": name, "detail": str(exc)},
            ) from exc

    def _eval_name(self, node: ast.Name, expression: str) -> float:
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolException(
            "表达式中使用了未定义的变量",
            context={"expression": expression, "name": node.id},
        )

    def _eval_attribute(self, node: ast.Attribute, expression: str) -> Any:
        """只放行 ``math.<白名单成员>``；其余属性访问（如 ``(1).__class__``）拒绝。"""
        if not (isinstance(node.value, ast.Name) and node.value.id == _MATH_NAMESPACE):
            raise ToolException(
                "表达式包含不允许的属性访问",
                context={"expression": expression, "node": type(node).__name__},
            )
        if node.attr in _FUNCTIONS:
            return _FUNCTIONS[node.attr]
        if node.attr in _CONSTANTS:
            return _CONSTANTS[node.attr]
        raise ToolException(
            "访问了未在白名单中的 math 成员",
            context={"expression": expression, "attribute": node.attr},
        )

    def _resolve_callable(self, func: ast.expr, expression: str) -> tuple[str, Callable[..., Any]]:
        """把调用目标解析成白名单函数；``math.sqrt`` 与裸 ``sqrt`` 等价。"""
        if isinstance(func, ast.Name):
            name = func.id
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == _MATH_NAMESPACE
        ):
            name = func.attr
        else:
            raise ToolException(
                "表达式中调用了不允许的函数",
                context={"expression": expression, "node": type(func).__name__},
            )
        resolved = _FUNCTIONS.get(name)
        if resolved is None:
            raise ToolException(
                "调用了未在白名单中的函数",
                context={"expression": expression, "function": name},
            )
        return name, resolved

    # ------------------------------------------------------------------ 防护

    @staticmethod
    def _apply(operation: Callable[[Any, Any], Any], left: Any, right: Any, expression: str) -> Any:
        try:
            return operation(left, right)
        except _ARITHMETIC_ERRORS as exc:
            raise ToolException(
                "计算失败",
                context={"expression": expression, "detail": str(exc)},
            ) from exc

    @staticmethod
    def _check_exponent(exponent: Any, expression: str) -> None:
        if isinstance(exponent, (int, float)) and abs(exponent) > _MAX_EXPONENT:
            raise ToolException(
                "幂运算的指数超出允许范围",
                context={"expression": expression, "exponent": exponent, "limit": _MAX_EXPONENT},
            )

    @staticmethod
    def _check_factorial(value: Any, expression: str) -> None:
        if isinstance(value, int) and value > _MAX_FACTORIAL:
            raise ToolException(
                "阶乘参数超出允许范围",
                context={"expression": expression, "value": value, "limit": _MAX_FACTORIAL},
            )

    # ------------------------------------------------------------------ 输出

    @staticmethod
    def _format(value: Any, expression: str) -> str:
        """把求值结果格式化为文本；复数与非数字结果拒绝，整数值的 float 归一为 int。"""
        if isinstance(value, complex):
            raise ToolException(
                "计算结果不是实数",
                context={"expression": expression, "value": str(value)},
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolException(
                "计算结果不是数字",
                context={"expression": expression, "value": repr(value)},
            )
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
