"""集中配置。

教程定位：第七章 7.3.2 Config 类。
与教程的差异（见 docs/deviations.md D2/D11）：``@dataclass`` 取代 ``pydantic.BaseModel``；
``default_model`` 不再硬编码过时的 ``gpt-3.5-turbo``（交由 provider 决定默认模型）；
``max_history_length`` 真正接线（``Agent.add_message`` 据此裁剪）；``from_env`` 读取 ``.env``。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Final

from dotenv import load_dotenv

from .exceptions import ConfigException

__all__ = ["Config", "env_bool", "env_float", "env_int"]

_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)
_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSY: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def env_int(env: Mapping[str, str], key: str, default: int | None = None) -> int | None:
    """读取整型环境变量；缺省或空串返回 ``default``。"""
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigException(f"环境变量 {key} 必须是整数", context={key: raw}) from exc


def env_float(env: Mapping[str, str], key: str, default: float) -> float:
    """读取浮点环境变量；缺省或空串返回 ``default``。"""
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigException(f"环境变量 {key} 必须是浮点数", context={key: raw}) from exc


def env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    """读取布尔环境变量，接受 1/0、true/false、yes/no、on/off（忽略大小写）。"""
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ConfigException(
        f"环境变量 {key} 必须是布尔值（1/0、true/false、yes/no、on/off）", context={key: raw}
    )


@dataclass
class Config:
    """框架配置。

    Args:
        default_model: 默认模型名；``None`` 表示由 provider 推断。
        default_provider: 默认 provider；``"auto"`` 表示按环境自动检测。
        temperature: 采样温度，取值 0.0~2.0。
        max_tokens: 单次响应上限；``None`` 表示交给 provider 决定。
        debug: 为 ``True`` 时 ``configure_logging`` 使用 DEBUG 级别。
        log_level: 日志级别名。
        max_history_length: Agent 保留的历史消息上限，超出后丢弃最早的。
    """

    default_model: str | None = None
    default_provider: str = "auto"
    temperature: float = 0.7
    max_tokens: int | None = None
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigException(
                "temperature 必须在 0.0~2.0 之间", context={"temperature": self.temperature}
            )
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ConfigException(
                "max_tokens 必须为正整数", context={"max_tokens": self.max_tokens}
            )
        if self.max_history_length <= 0:
            raise ConfigException(
                "max_history_length 必须为正整数",
                context={"max_history_length": self.max_history_length},
            )
        level = self.log_level.upper()
        if level not in _LOG_LEVELS:
            raise ConfigException(
                "log_level 非法",
                context={"log_level": self.log_level, "allowed": sorted(_LOG_LEVELS)},
            )
        self.log_level = level

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        load_dotenv_file: bool = True,
    ) -> Config:
        """从环境变量构造配置。

        Args:
            env: 环境变量映射；``None`` 表示使用 ``os.environ``（测试可注入）。
            load_dotenv_file: 是否先把当前工作目录的 ``.env`` 载入 ``os.environ``。
        """
        if load_dotenv_file and env is None:
            load_dotenv(override=False)
        env_map = os.environ if env is None else env
        history_limit = env_int(env_map, "MAX_HISTORY_LENGTH", 100)
        return cls(
            default_model=env_map.get("LLM_MODEL_ID") or None,
            default_provider=env_map.get("LLM_PROVIDER", "auto"),
            temperature=env_float(env_map, "TEMPERATURE", 0.7),
            max_tokens=env_int(env_map, "MAX_TOKENS"),
            debug=env_bool(env_map, "DEBUG", False),
            log_level=env_map.get("LOG_LEVEL", "INFO"),
            max_history_length=100 if history_limit is None else history_limit,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典（教程的 ``.dict()`` 对应实现）。"""
        return asdict(self)

    def configure_logging(self) -> None:
        """把 ``log_level``（``debug=True`` 时取 DEBUG）应用到 ``myagent`` 日志器。"""
        level = "DEBUG" if self.debug else self.log_level
        logging.getLogger("myagent").setLevel(level)
