from __future__ import annotations

import json
from typing import Literal, Optional

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
    memory_id: Optional[str] = Field(
        default=None,
        description="Memory id to fetch when action is 'get'. Leave null if unknown.",
    )
    namespace: Optional[tuple[str, ...]] = Field(
        default=None,
        description="Namespace tuple. If null, default to (user_id, 'memories')",
    )


class MemoryAgent:
    """Memory management agent (list/get) backed by a LangGraph store."""

    def __init__(
        self,
        llm: ChatOpenAI,
        *,
        memory_store: Optional[BaseStore] = None,
        max_trim_messages: int = 12,
    ) -> None:
        self.llm = llm
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

        system = SystemMessage(
            content=(
                "You are a memory management assistant.\n"
                "Decide whether the user wants to list stored memories or retrieve a specific memory by id.\n"
                "- If user asks to see/list memories: action='list'.\n"
                "- If user asks for a specific memory and provides/mentions an id: action='get' and extract memory_id.\n"
                "- If namespace is explicitly requested or mentioned, extract it as a tuple of strings.\n"
                "- Otherwise, leave namespace null.\n"
            )
        )

        llm_structured = self.llm.with_structured_output(_MemoryIntent)
        return await llm_structured.ainvoke([system, *trimmed])

    async def run(self, state: StorytellerState) -> StorytellerState:
        try:
            intent = await self._infer_intent(state)
        except Exception as e:
            logger.error(f"Failed to infer memory intent: {e}")
            payload = {"error": f"Failed to infer memory intent: {e}"}
            return {"messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))]}

        user_id = state.get("user_id", "default")
        namespace = intent.namespace or (user_id, "memories")

        if intent.action == "get":
            if not intent.memory_id:
                payload = {
                    "tool": "get_memory",
                    "namespace": namespace,
                    "error": "No memory_id provided or found in conversation.",
                }
            else:
                result = get_memory(self.memory_store, namespace, intent.memory_id)
                payload = {
                    "tool": "get_memory",
                    "namespace": namespace,
                    "memory_id": intent.memory_id,
                    "result": result,
                }
        else:  # list
            result = list_memories(self.memory_store, namespace)
            payload = {"tool": "list_memories", "namespace": namespace, "result": result}
        logger.debug(f"Memory agent result: {payload}")
        return {"messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))]}

    def build_graph(self):
        builder = StateGraph(StorytellerState)
        builder.add_node("memory_tool", self.run)
        builder.set_entry_point("memory_tool")
        return builder.compile()
