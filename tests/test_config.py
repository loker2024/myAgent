"""Config 的行为测试。"""

from __future__ import annotations

import logging

import pytest

from myagent import Config, ConfigException


def test_from_env_reads_every_supported_variable() -> None:
    env = {
        "LLM_MODEL_ID": "deepseek-chat",
        "LLM_PROVIDER": "deepseek",
        "TEMPERATURE": "1.5",
        "MAX_TOKENS": "256",
        "DEBUG": "true",
        "LOG_LEVEL": "debug",
        "MAX_HISTORY_LENGTH": "7",
    }

    config = Config.from_env(env=env)

    assert config.default_model == "deepseek-chat"
    assert config.default_provider == "deepseek"
    assert config.temperature == 1.5
    assert config.max_tokens == 256
    assert config.debug is True
    assert config.log_level == "DEBUG"  # 归一化为大写
    assert config.max_history_length == 7


def test_from_env_without_variables_equals_defaults() -> None:
    assert Config.from_env(env={}) == Config()


@pytest.mark.parametrize(
    "env",
    [
        {"TEMPERATURE": "hot"},
        {"TEMPERATURE": "2.5"},
        {"MAX_TOKENS": "many"},
        {"MAX_TOKENS": "0"},
        {"DEBUG": "maybe"},
        {"LOG_LEVEL": "LOUD"},
        {"MAX_HISTORY_LENGTH": "0"},
    ],
    ids=[
        "非数值温度",
        "温度越界",
        "非数值上限",
        "上限为零",
        "非布尔 debug",
        "非法日志级别",
        "历史上限为零",
    ],
)
def test_from_env_rejects_invalid_values(env: dict[str, str]) -> None:
    with pytest.raises(ConfigException):
        Config.from_env(env=env)


def test_constructor_validates_fields() -> None:
    with pytest.raises(ConfigException):
        Config(temperature=-0.1)
    with pytest.raises(ConfigException):
        Config(max_history_length=0)


def test_to_dict_exposes_all_fields() -> None:
    payload = Config(default_model="m", max_tokens=8).to_dict()

    assert payload["default_model"] == "m"
    assert payload["max_tokens"] == 8
    assert set(payload) == {
        "default_model",
        "default_provider",
        "temperature",
        "max_tokens",
        "debug",
        "log_level",
        "max_history_length",
    }


def test_configure_logging_applies_debug_level_and_is_reversible() -> None:
    logger = logging.getLogger("myagent")
    original = logger.level
    try:
        Config(debug=True, log_level="WARNING").configure_logging()
        assert logger.level == logging.DEBUG

        Config(debug=False, log_level="WARNING").configure_logging()
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(original)
