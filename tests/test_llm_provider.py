"""provider 自动检测与凭据解析（纯函数，无网络）。"""

from __future__ import annotations

import pytest

from myagent import (
    ConfigException,
    HelloAgentsLLM,
    default_model_for,
    detect_provider,
    resolve_credentials,
    spec_for,
)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"MODELSCOPE_API_KEY": "k", "OPENAI_API_KEY": "k"}, "modelscope"),
        ({"OPENAI_API_KEY": "k"}, "openai"),
        ({"DEEPSEEK_API_KEY": "k"}, "deepseek"),
        ({"DASHSCOPE_API_KEY": "k"}, "qwen"),
        ({"MOONSHOT_API_KEY": "k"}, "kimi"),
        ({"GLM_API_KEY": "k"}, "zhipu"),
        ({"OLLAMA_HOST": "http://localhost:11434"}, "ollama"),
        ({"VLLM_API_KEY": "k"}, "vllm"),
        ({"ANTHROPIC_API_KEY": "k"}, "anthropic"),
    ],
    ids=[
        "专属环境变量优先",
        "openai",
        "deepseek",
        "qwen",
        "kimi",
        "zhipu",
        "ollama",
        "vllm",
        "anthropic",
    ],
)
def test_detects_provider_from_dedicated_env_vars(env: dict[str, str], expected: str) -> None:
    assert detect_provider(env=env) == expected


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api-inference.modelscope.cn/v1/", "modelscope"),
        ("https://api.deepseek.com", "deepseek"),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen"),
        ("https://api.anthropic.com", "anthropic"),
        ("http://localhost:11434/v1", "ollama"),
        ("http://127.0.0.1:8000/v1", "vllm"),
        ("http://localhost:8080/v1", "local"),
        ("http://127.0.0.1:9999/v1", "local"),
    ],
    ids=[
        "modelscope",
        "deepseek",
        "qwen",
        "anthropic",
        "本机 ollama 端口",
        "本机 vllm 端口",
        "本机常见端口",
        "本机未知端口",
    ],
)
def test_detects_provider_from_base_url(base_url: str, expected: str) -> None:
    assert detect_provider(base_url=base_url, env={}) == expected


def test_base_url_takes_priority_over_generic_key_for_tutorial_priority_order() -> None:
    assert (
        detect_provider(base_url="https://api.deepseek.com", env={"LLM_API_KEY": "ms-abc"})
        == "deepseek"
    )


@pytest.mark.parametrize(
    ("api_key", "expected"),
    [
        ("ms-abcdef", "modelscope"),
        ("sk-ant-api03-xyz", "anthropic"),
        ("ollama", "ollama"),
        ("vllm", "vllm"),
        ("local", "local"),
        ("0123456789abcdef0123456789abc.", "zhipu"),
    ],
    ids=[
        "modelscope 前缀",
        "anthropic 前缀",
        "字面 ollama",
        "字面 vllm",
        "字面 local",
        "zhipu 点号",
    ],
)
def test_detects_provider_from_key_format(api_key: str, expected: str) -> None:
    assert detect_provider(api_key=api_key, env={}) == expected


def test_falls_back_to_auto_when_nothing_matches() -> None:
    assert detect_provider(api_key="sk-1234567890", env={}) == "auto"


def test_dedicated_env_var_outranks_base_url_per_tutorial_priority() -> None:
    # 正文 7.2.3：提供商专属环境变量优先级最高，即使 base_url 指向别家。
    assert (
        detect_provider(base_url="https://api.deepseek.com", env={"OPENAI_API_KEY": "k"})
        == "openai"
    )


def test_base_url_argument_is_used_when_no_dedicated_env_var_exists() -> None:
    assert (
        detect_provider(base_url="https://api.deepseek.com", env={"LLM_API_KEY": "x"}) == "deepseek"
    )


def test_resolve_credentials_uses_provider_defaults_and_placeholder_keys() -> None:
    key, base_url = resolve_credentials("ollama", env={})

    assert (key, base_url) == ("ollama", "http://localhost:11434/v1")

    key, base_url = resolve_credentials("deepseek", env={"DEEPSEEK_API_KEY": "d"})
    assert (key, base_url) == ("d", "https://api.deepseek.com")


def test_resolve_credentials_prefers_explicit_arguments() -> None:
    key, base_url = resolve_credentials(
        "openai",
        api_key="explicit",
        base_url="https://example.test/v1",
        env={"OPENAI_API_KEY": "env"},
    )

    assert (key, base_url) == ("explicit", "https://example.test/v1")


def test_requires_explicit_base_url_for_auto_provider() -> None:
    key, base_url = resolve_credentials("auto", env={"LLM_API_KEY": "k"})

    assert key == "k"
    assert base_url is None


def test_unknown_provider_is_rejected_with_supported_list() -> None:
    with pytest.raises(ConfigException) as excinfo:
        spec_for("no-such-provider")

    assert "no-such-provider" in str(excinfo.value)
    assert "openai" in str(excinfo.value.context["supported"])


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("deepseek", "deepseek-chat"),
        ("qwen", "qwen-plus"),
        ("vllm", None),
        ("local", None),
        ("auto", None),
    ],
)
def test_default_model_per_provider(provider: str, expected: str | None) -> None:
    assert default_model_for(provider) == expected


def test_client_reports_missing_configuration_with_provider_context() -> None:
    with pytest.raises(ConfigException) as excinfo:
        HelloAgentsLLM(provider="auto", env={})

    assert excinfo.value.context["provider"] == "auto"


def test_client_requires_explicit_model_when_provider_has_no_default() -> None:
    with pytest.raises(ConfigException) as excinfo:
        HelloAgentsLLM(provider="vllm", env={})

    assert "model" in str(excinfo.value)


def test_client_reads_model_and_timeout_from_env() -> None:
    client = HelloAgentsLLM(
        provider="deepseek",
        env={"DEEPSEEK_API_KEY": "k", "LLM_MODEL_ID": "deepseek-reasoner", "LLM_TIMEOUT": "12.5"},
    )

    assert client.model == "deepseek-reasoner"
    assert client.timeout == 12.5
    assert client.provider == "deepseek"
