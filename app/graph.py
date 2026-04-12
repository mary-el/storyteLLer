import json
from typing import Literal, Optional

import dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.character_gen import CharacterGenerator
from app.agents.memory_agent import MemoryAgent
from app.agents.world_gen import WorldGenerator
from app.config import AppConfig, load_app_config
from app.state.schemas import (
    CharacterObject,
    Story,
    StoryStep,
    StorytellerState,
    WorldObject,
    coerce_story,
)
from app.utils import logger

dotenv.load_dotenv()


class RouterResponse(BaseModel):
    """Structured response from setup router (world exists; characters until begin)."""

    node: Literal["generate_character", "dialogue", "begin_story"] = Field(
        description="Next node: character subgraph, end turn, or enter story phase"
    )
    response: str = Field(description="User-visible message when node is dialogue or routing hint")


class StoryResponse(BaseModel):
    """Structured response from story narrator."""

    node: Literal["memory_tool", "dialogue"] = Field(description="Memory lookup or narrative reply")
    response: str = Field(description="Narrative or reply when node is dialogue")


class Storyteller:
    def __init__(
        self,
        langdev: bool = False,
        memory_store: Optional[InMemoryStore] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_app_config()
        self.llm = ChatOpenAI(
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
        )
        self.checkpointer = MemorySaver() if not langdev else None
        self.memory_store = memory_store
        self.langdev = langdev
        self.character_generator = CharacterGenerator(
            self.llm,
            self.checkpointer,
            memory_store,
            langdev=langdev,
            app_config=self.config,
        )
        self.memory_agent = MemoryAgent(
            self.llm,
            system_prompt=self.config.memory_agent.system_prompt,
            memory_store=memory_store,
        )
        self.world_generator = WorldGenerator(
            self.llm,
            self.checkpointer,
            memory_store,
            langdev=langdev,
            app_config=self.config,
        )
        self.router_template = ChatPromptTemplate(
            [
                MessagesPlaceholder(variable_name="conversation", optional=True),
                ("system", self.config.router.system_prompt),
            ]
        )
        self.router_max_len = self.config.router.max_trim_tokens
        self.story_max_len = self.config.story_narrator.max_trim_tokens
        self.graph = self.build_graph()
        self.waiting_for_feedback = False

    def next_after_start(
        self, state: StorytellerState
    ) -> Literal["generate_world", "router", "story"]:
        if state.get("phase") == "story":
            return "story"
        story = coerce_story(state.get("story"))
        if story is None or story.world is None:
            return "generate_world"
        return "router"

    def route_from_router(
        self, state: StorytellerState
    ) -> Literal["generate_character", "begin_story", "dialogue"]:
        messages = state.get("messages", [])
        if not messages:
            return "dialogue"
        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        try:
            data = json.loads(content)
            node = data.get("node", "dialogue")
            if node in ["begin_story", "generate_character"]:
                return node
            return "dialogue"
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning(f"Failed to parse router JSON: {content!r}")
            return "dialogue"

    def route_from_story(self, state: StorytellerState) -> Literal["memory_tool", "dialogue"]:
        messages = state.get("messages", [])
        if not messages:
            return "dialogue"
        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        try:
            data = json.loads(content)
            node = data.get("node", "dialogue")
            return node if node == "memory_tool" else "dialogue"
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning(f"Failed to parse story JSON: {content!r}")
            return "dialogue"

    @staticmethod
    def _story_context_block(story: Story | None) -> str:
        if not story:
            return "No story context."
        parts: list[str] = []
        if story.world is not None:
            parts.append("World:\n" + story.world.model_dump_json(indent=2))
        for co in story.characters:
            parts.append(f"Character ({co.object_id}):\n" + co.model_dump_json(indent=2))
        if not parts:
            return "Story setup in progress."
        return "\n\n".join(parts)

    async def router_node(self, state: StorytellerState) -> StorytellerState:
        messages = trim_messages(
            state["messages"],
            max_tokens=self.router_max_len,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=True,
        )
        logger.debug("routing to router")
        prompt = self.router_template.invoke({"conversation": messages})
        llm_with_structure = self.llm.with_structured_output(RouterResponse)
        response = await llm_with_structure.ainvoke(prompt)
        json_content = json.dumps(
            {"node": response.node, "response": response.response},
            ensure_ascii=False,
        )
        return {"messages": [AIMessage(content=json_content)], "status": None}

    async def story_node(self, state: StorytellerState) -> StorytellerState:
        story = coerce_story(state.get("story"))
        context = self._story_context_block(story)
        combined_system = (
            f"{self.config.story_narrator.system_prompt}\n\n--- Current story setup ---\n{context}"
        )
        messages = trim_messages(
            state["messages"],
            max_tokens=self.story_max_len,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=True,
        )
        logger.debug("routing to story narrator")
        # Do not pass JSON context through ChatPromptTemplate — `{` in JSON would be parsed as variables.
        llm_messages = [SystemMessage(content=combined_system), *messages]
        llm_with_structure = self.llm.with_structured_output(StoryResponse)
        response = await llm_with_structure.ainvoke(llm_messages)
        json_content = json.dumps(
            {"node": response.node, "response": response.response},
            ensure_ascii=False,
        )
        return {"messages": [AIMessage(content=json_content)], "status": None}

    def enter_story(self, state: StorytellerState) -> StorytellerState:
        return {"phase": "story"}

    def finalize_object(self, state: StorytellerState) -> StorytellerState:
        logger.debug("Finalizing object generation")
        generated_object = state.get("generated_object")
        if not generated_object:
            return {}
        story = coerce_story(state.get("story")) or Story()
        if isinstance(generated_object, WorldObject):
            story = story.model_copy(update={"world": generated_object.world})
            phase: StoryStep = "characters"
        elif isinstance(generated_object, CharacterObject):
            story = story.model_copy(update={"characters": [*story.characters, generated_object]})
            phase = state.get("phase", "characters")
        else:
            phase = state.get("phase", "characters")
        return {
            "messages": [
                SystemMessage(
                    content=f"SYSTEM_EVENT: OBJECT_CREATED: {generated_object.model_dump_json()}"
                )
            ],
            "generated_object": None,
            "story": story,
            "phase": phase,
        }

    def build_graph(self):
        graph = StateGraph(StorytellerState)

        graph.add_node("router", self.router_node)
        graph.add_node("story", self.story_node)
        graph.add_node("enter_story", self.enter_story)
        graph.add_node("memory_tool", self.memory_agent.graph)

        graph.add_node("generate_character", self.character_generator.graph)
        graph.add_node("generate_world", self.world_generator.graph)
        graph.add_node("finalize_object", self.finalize_object)

        graph.add_conditional_edges(
            START,
            self.next_after_start,
            {
                "generate_world": "generate_world",
                "router": "router",
                "story": "story",
            },
        )

        graph.add_conditional_edges(
            "router",
            self.route_from_router,
            {
                "generate_character": "generate_character",
                "begin_story": "enter_story",
                "dialogue": END,
            },
        )

        graph.add_edge("enter_story", "story")

        graph.add_conditional_edges(
            "story",
            self.route_from_story,
            {
                "memory_tool": "memory_tool",
                "dialogue": END,
            },
        )

        graph.add_edge("generate_character", "finalize_object")
        graph.add_edge("generate_world", "finalize_object")
        graph.add_edge("finalize_object", "router")
        # End after memory so the list/get reply stays the last message; next user turn goes START → story.
        graph.add_edge("memory_tool", END)

        return graph.compile(checkpointer=self.checkpointer)

    async def arun(self, query: str | Command, user_id: str, thread_id: str = None) -> dict:
        """Run the storyteller asynchronously"""
        if isinstance(query, Command):
            input_data = query
        else:
            input_data = {"messages": [HumanMessage(content=query)], "user_id": user_id}
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id or user_id}}
        result = await self.graph.ainvoke(input_data, config)
        return result

    async def tell(self, query: str, user_id: str, thread_id: str = None) -> str:
        # Route resume vs new input from checkpoint truth, not an in-memory flag (avoids stale
        # waiting_for_feedback sending normal story turns as Command(resume) into object_gen).
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id or user_id}}
        snap = await self.graph.aget_state(config)
        if snap.interrupts:
            logger.info(f"Sending command: {query}")
            result = await self.arun(Command(resume=query), user_id, thread_id)
        else:
            result = await self.arun(query, user_id, thread_id)
        interrupts = result.get("__interrupt__")
        self.waiting_for_feedback = bool(interrupts)
        if interrupts:
            if isinstance(interrupts, list) and interrupts:
                first = interrupts[0]
                val = first.value if hasattr(first, "value") else first
                if isinstance(val, dict):
                    text = (val.get("draft") or val.get("hint") or "") or ""
                    return text.strip() if text.strip() else str(val)
                return str(val) if val is not None else ""
            return str(interrupts)
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        return getattr(last_message, "content", "") if last_message else ""
