from __future__ import annotations

import json
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from app.memory import get_memory, list_memories
from app.state.schemas import StorytellerState
from app.utils import logger


class _MemoryIntent(BaseModel):
    action: Literal["list", "get"] = Field(
        description="Whether to list memories or fetch a specific memory by id"
    )
    kind: Literal["all", "character", "world"] = Field(
        default="all",
        description=(
            "For list: whether the user asked for all objects, only characters, or only worlds. "
            "Use 'world' when they ask about worlds/settings. "
            "Use 'character' when they ask only about characters/personas. "
            "Use 'all' when they ask generally (what do we have, list everything)."
        ),
    )
    memory_id: Optional[str] = Field(
        default=None,
        description="Memory id to fetch when action is 'get'. Leave null if unknown.",
    )
    namespace: Optional[tuple[str, ...]] = Field(
        default=None,
        description="Namespace tuple. If null, default to (user_id, 'memories')",
    )


def _value_matches_kind(value: dict[str, Any], kind: Literal["all", "character", "world"]) -> bool:
    if kind == "all":
        return True
    return isinstance(value.get(kind), dict)


def _memory_record_line(memory_id: str, value: dict[str, Any]) -> str:
    """Full stored payload as indented JSON (all keys / nested fields)."""
    if not value:
        return memory_id
    payload = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    return f"{memory_id}\n{payload}"


def _format_numbered_list(lines: list[str]) -> str:
    if not lines:
        return "No memories stored."
    blocks: list[str] = []
    for i, block in enumerate(lines):
        blocks.append(f"{i + 1}. {block}")
    return "\n\n".join(blocks)


def _router_message(response: str) -> str:
    """Same wire format as dialogue END path / main.py expectation."""
    return json.dumps({"node": "dialogue", "response": response}, ensure_ascii=False)


class MemoryAgent:
    """Memory management agent (list/get) backed by a LangGraph store."""

    def __init__(
        self,
        llm: ChatOpenAI,
        *,
        system_prompt: str,
        memory_store: Optional[BaseStore] = None,
        max_trim_messages: int = 12,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.memory_store = memory_store
        self.max_trim_messages = max_trim_messages
        self.graph = self.build_graph()

    def _trim_conversation(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return trim_messages(
            messages,
            max_tokens=self.max_trim_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
        )

    async def _infer_intent(self, state: StorytellerState) -> _MemoryIntent:
        logger.debug("Running memory agent")
        messages = state.get("messages", [])
        trimmed = self._trim_conversation(messages)

        system = SystemMessage(content=self.system_prompt)

        llm_structured = self.llm.with_structured_output(_MemoryIntent)
        return await llm_structured.ainvoke([system, *trimmed])

    async def run(self, state: StorytellerState) -> StorytellerState:
        try:
            intent = await self._infer_intent(state)
        except Exception as e:
            logger.error(f"Failed to infer memory intent: {e}")
            text = f"Memory tool failed: {e}"
            return {"messages": [AIMessage(content=_router_message(text))]}

        user_id = state.get("user_id", "default")
        namespace = intent.namespace or (user_id, "memories")

        if intent.action == "get":
            if not intent.memory_id:
                text = "No memory id was found for this request."
                return {"messages": [AIMessage(content=_router_message(text))]}
            record = get_memory(self.memory_store, namespace, intent.memory_id)
            if record is None:
                text = f"Memory not found: {intent.memory_id}"
                return {"messages": [AIMessage(content=_router_message(text))]}
            value = record.get("value") if isinstance(record.get("value"), dict) else {}
            line = _memory_record_line(intent.memory_id, value)
            text = f"1. {line}"
            logger.debug(f"Memory agent list (get): {text!r}")
            return {"messages": [AIMessage(content=_router_message(text))]}

        # list
        result = list_memories(self.memory_store, namespace)
        kind = intent.kind
        lines: list[str] = []
        for mid in sorted(result.keys()):
            rec = result[mid]
            val = rec.get("value") if isinstance(rec.get("value"), dict) else {}
            if not _value_matches_kind(val, kind):
                continue
            lines.append(_memory_record_line(mid, val))
        text = _format_numbered_list(lines)
        logger.debug(f"Memory agent list (list): kind={kind} {len(lines)} items")
        return {"messages": [AIMessage(content=_router_message(text))]}

    def build_graph(self):
        builder = StateGraph(StorytellerState)
        builder.add_node("memory_tool", self.run)
        builder.set_entry_point("memory_tool")
        return builder.compile()
