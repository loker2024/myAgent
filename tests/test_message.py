"""Message 的行为测试。"""

from __future__ import annotations

from datetime import UTC, datetime

from myagent import Message


def test_positional_construction_matches_tutorial_call_form() -> None:
    message = Message("你好", "user")

    assert message.content == "你好"
    assert message.role == "user"
    assert message.metadata == {}


def test_timestamp_is_timezone_aware_utc_by_default() -> None:
    message = Message("hi", "assistant")

    assert message.timestamp.tzinfo is UTC
    assert abs((datetime.now(UTC) - message.timestamp).total_seconds()) < 60


def test_metadata_is_not_shared_between_instances() -> None:
    first = Message("a", "user")
    second = Message("b", "user")

    first.metadata["tool"] = "calculator"

    assert second.metadata == {}


def test_to_dict_keeps_only_role_and_content() -> None:
    message = Message("hi", "system", metadata={"source": "test"})

    assert message.to_dict() == {"role": "system", "content": "hi"}


def test_str_renders_role_and_content() -> None:
    assert str(Message("hi", "tool")) == "[tool] hi"
