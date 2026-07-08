import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.utils import (
    invoke_structured,
    message_text,
    parse_last_json,
    split_thinking,
    strip_thinking,
    visible_response,
)


class TestSplitThinking:
    def test_no_thinking_block(self):
        thinking, visible = split_thinking("Just a plain answer.")
        assert thinking is None
        assert visible == "Just a plain answer."

    def test_single_thinking_block(self):
        thinking, visible = split_thinking("<think>hidden reasoning</think>The answer.")
        assert thinking == "hidden reasoning"
        assert visible == "The answer."

    def test_multiple_thinking_blocks(self):
        text = "<think>first</think>Hello <think>second</think>world"
        thinking, visible = split_thinking(text)
        assert thinking == "first\n\nsecond"
        assert visible == "Hello world"

    def test_case_insensitive_tag(self):
        thinking, visible = split_thinking("<THINK>reasoning</THINK>Reply")
        assert thinking == "reasoning"
        assert visible == "Reply"

    def test_non_string_input(self):
        thinking, visible = split_thinking(None)
        assert thinking is None
        assert visible == ""


class TestStripThinking:
    def test_removes_thinking(self):
        assert strip_thinking("<think>x</think>  visible  ") == "visible"

    def test_plain_text_untouched(self):
        assert strip_thinking("hello") == "hello"

    def test_non_string_input(self):
        assert strip_thinking(None) == ""
        assert strip_thinking(42) == "42"


class TestParseLastJson:
    def test_valid_json(self):
        msg = AIMessage(content='{"node": "dialogue", "response": "hi"}')
        assert parse_last_json([msg]) == {"node": "dialogue", "response": "hi"}

    def test_invalid_json_returns_empty(self):
        msg = AIMessage(content="not json at all")
        assert parse_last_json([msg]) == {}

    def test_empty_messages(self):
        assert parse_last_json([]) == {}

    def test_uses_last_message(self):
        msgs = [
            AIMessage(content='{"node": "first"}'),
            AIMessage(content='{"node": "last"}'),
        ]
        assert parse_last_json(msgs) == {"node": "last"}


class TestMessageText:
    def test_ai_message_json_wire_format(self):
        msg = AIMessage(content=json.dumps({"node": "dialogue", "response": "Story text"}))
        assert message_text(msg) == "Story text"

    def test_ai_message_plain_content(self):
        msg = AIMessage(content="plain reply")
        assert message_text(msg) == "plain reply"

    def test_human_message(self):
        msg = HumanMessage(content="user input")
        assert message_text(msg) == "user input"

    def test_null_response_field(self):
        msg = AIMessage(content=json.dumps({"node": "begin_story", "response": None}))
        assert message_text(msg) == ""


class TestVisibleResponse:
    def test_json_wire_format(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content=json.dumps({"node": "dialogue", "response": "Narrative."})),
        ]
        assert visible_response(msgs) == "Narrative."

    def test_plain_ai_message(self):
        msgs = [AIMessage(content="plain narration")]
        assert visible_response(msgs) == "plain narration"

    def test_strips_thinking_from_response(self):
        payload = json.dumps({"response": "<think>internal</think>Visible part"})
        assert visible_response([AIMessage(content=payload)]) == "Visible part"

    def test_no_ai_message(self):
        assert visible_response([HumanMessage(content="only human")]) == ""

    def test_empty_messages(self):
        assert visible_response([]) == ""


class _Decision(BaseModel):
    node: str


class _FakeStructured:
    def __init__(self, failures: int, result: _Decision):
        self.failures = failures
        self.result = result
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("transient failure")
        return self.result


class _FakeLLM:
    def __init__(self, structured: _FakeStructured):
        self._structured = structured

    def with_structured_output(self, schema):
        return self._structured


class TestInvokeStructured:
    def test_succeeds_first_try(self):
        structured = _FakeStructured(failures=0, result=_Decision(node="dialogue"))
        result = asyncio.run(invoke_structured(_FakeLLM(structured), _Decision, []))
        assert result.node == "dialogue"
        assert structured.calls == 1

    def test_retries_then_succeeds(self):
        structured = _FakeStructured(failures=1, result=_Decision(node="dialogue"))
        result = asyncio.run(invoke_structured(_FakeLLM(structured), _Decision, [], retries=1))
        assert result.node == "dialogue"
        assert structured.calls == 2

    def test_raises_after_exhausted_retries(self):
        structured = _FakeStructured(failures=5, result=_Decision(node="dialogue"))
        with pytest.raises(RuntimeError):
            asyncio.run(invoke_structured(_FakeLLM(structured), _Decision, [], retries=1))
        assert structured.calls == 2
