import asyncio
import json
import uuid
from pathlib import Path
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
from trustcall import create_extractor

from app import persistence
from app.agents.character_gen import CharacterGenerator
from app.agents.memory_agent import MemoryAgent
from app.agents.world_gen import WorldGenerator
from app.config import AppConfig, load_app_config
from app.state.schemas import (
    Character,
    CharacterObject,
    Story,
    StoryStep,
    StorytellerState,
    WorldObject,
    coerce_story,
)
from app.utils import (
    STRUCTURED_OUTPUT_ERROR,
    invoke_structured,
    logger,
    message_text,
    parse_last_json,
    strip_thinking,
    visible_response,
)

dotenv.load_dotenv()


class RouterResponse(BaseModel):
    """Structured response from setup router (world exists; characters until begin)."""

    node: Literal["generate_character", "dialogue", "begin_story"] = Field(
        description="Next node: character subgraph, end turn, or enter story phase"
    )
    response: str = Field(description="User-visible message when node is dialogue")


class SummaryResponse(BaseModel):
    """Structured rolling summary of the story transcript."""

    summary: str = Field(
        description=(
            "A neutral 2-5 sentence summary of the story so far. "
            "Include only events from the transcript; do not add new information or continue the narrative."
        )
    )
    title: str = Field(
        description="A title for the story so far. It must be short and enthralling.",
        default="Untitled",
    )


class StoryResponse(BaseModel):
    """Structured response from story narrator."""

    node: Literal["memory_tool", "dialogue", "update_characters"] = Field(
        description="Memory lookup, narrative reply, or character state update"
    )
    response: str = Field(description="Narrative or reply (always fill this)")
    character_ids: list[str] = Field(
        default_factory=list,
        description=(
            "object_ids of characters whose state changed this turn. "
            "Empty list unless node is update_characters."
        ),
    )


class Storyteller:
    def __init__(
        self,
        langdev: bool = False,
        memory_store: Optional[InMemoryStore] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_app_config()
        self.saves_dir: Optional[Path] = (
            Path(self.config.saves_dir) if self.config.saves_dir else None
        )
        self.llm = ChatOpenAI(**self.config.llm)
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
        self.character_extractor = create_extractor(
            self.llm, tools=[Character], tool_choice="required", enable_inserts=False
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

    # ── routing ────────────────────────────────────────────────────────────────

    def next_after_start(
        self, state: StorytellerState
    ) -> Literal["greeting", "generate_world", "router", "story"]:
        if state.get("phase") == "story":
            return "story"
        story = coerce_story(state.get("story"))
        if story is None or story.world is None:
            if not state.get("messages"):
                return "greeting"
            return "generate_world"
        return "router"

    def route_from_router(
        self, state: StorytellerState
    ) -> Literal["generate_character", "begin_story", "dialogue"]:
        node = parse_last_json(state.get("messages", [])).get("node", "dialogue")
        if node in ("begin_story", "generate_character"):
            return node
        return "dialogue"

    def route_from_story(self, state: StorytellerState) -> Literal["memory_tool"] | list[str]:
        node = parse_last_json(state.get("messages", [])).get("node", "dialogue")
        if node == "memory_tool":
            return "memory_tool"
        return ["summary", "update_characters"]

    # ── story context ──────────────────────────────────────────────────────────

    @staticmethod
    def _story_context_block(story: Story | None) -> str:
        if not story:
            return "No story context."
        parts: list[str] = []
        if story.world is not None:
            parts.append("World:\n" + story.world.model_dump_json(indent=2))
        for co in story.characters:
            parts.append(f"Character ({co.object_id}):\n" + co.model_dump_json(indent=2))
        return "\n\n".join(parts) if parts else "Story setup in progress."

    # ── nodes ──────────────────────────────────────────────────────────────────

    def greeting_node(self, state: StorytellerState) -> StorytellerState:
        return {"messages": [AIMessage(content=self.config.greeting)]}

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
        try:
            response = await invoke_structured(self.llm, RouterResponse, prompt)
        except Exception as e:
            logger.error(f"router_node failed: {e}")
            response = RouterResponse(node="dialogue", response=STRUCTURED_OUTPUT_ERROR)
        payload = {"node": response.node, "response": response.response}
        return {
            "messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))],
            "status": None,
        }

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
        llm_messages = [SystemMessage(content=combined_system), *messages]
        try:
            response = await invoke_structured(self.llm, StoryResponse, llm_messages)
        except Exception as e:
            logger.error(f"story_node failed: {e}")
            response = StoryResponse(
                node="dialogue", response=STRUCTURED_OUTPUT_ERROR, character_ids=[]
            )
        payload = {
            "node": response.node,
            "response": response.response,
            "character_ids": response.character_ids,
        }
        return {
            "messages": [AIMessage(content=json.dumps(payload, ensure_ascii=False))],
            "status": None,
        }

    async def _patch_character(self, state: StorytellerState, character_id: str) -> Story | None:
        """Extract updated state for one character; return a patched Story or None on failure."""
        story = coerce_story(state.get("story"))
        if not story:
            return None
        target = next((c for c in story.characters if c.object_id == character_id), None)
        if target is None:
            logger.warning(f"_patch_character: {character_id!r} not found")
            return None
        messages = trim_messages(
            state.get("messages", []),
            max_tokens=self.config.story_update.world_patch_max_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
        )
        clean_messages = [
            AIMessage(content=message_text(m)) if isinstance(m, AIMessage) else m for m in messages
        ]
        try:
            result = await self.character_extractor.ainvoke(
                {
                    "messages": clean_messages
                    + [SystemMessage(content="Update the character based on the story so far.")],
                    "existing": {"Character": target.character.model_dump()},
                }
            )
            if not result["responses"]:
                return None
            target = target.model_copy(update={"character": result["responses"][0]})
            namespace = (state.get("user_id", "default"), "memories")
            if self.memory_store:
                self.memory_store.put(namespace, target.object_id, target)
            logger.debug(f"_patch_character: updated {character_id}")
            updated_chars = [target if c.object_id == character_id else c for c in story.characters]
            return story.model_copy(update={"characters": updated_chars})
        except Exception as e:
            logger.error(f"_patch_character failed for {character_id}: {e}")
            return None

    async def summary_node(self, state: StorytellerState) -> StorytellerState:
        """Generate a new rolling summary when due; stored in _turn_summary for finalize_turn."""
        story = coerce_story(state.get("story"))
        turn = (state.get("turn") or 0) + 1
        if turn % self.config.story_update.summary_every_n_turns != 0:
            return {}
        messages = trim_messages(
            state.get("messages", []),
            max_tokens=self.config.story_update.max_trim_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
        )

        transcript_lines: list[str] = []
        for m in messages:
            text = message_text(m).strip()
            if not text:
                continue
            role = "User" if isinstance(m, HumanMessage) else "Narrator"
            transcript_lines.append(f"{role}: {text}")
        transcript = "\n".join(transcript_lines)

        summary_prompt = self.config.story_update.summary_prompt.format(
            previous_summary=story.summary, transcript=transcript, title=story.title
        )
        try:
            response = await self.llm.with_structured_output(SummaryResponse, strict=False).ainvoke(
                [
                    SystemMessage(content=summary_prompt),
                ]
            )
            return {"_turn_summary": response.summary.strip(), "_turn_title": response.title}
        except Exception as e:
            logger.error(f"summary_node failed: {e}")
            return {}

    async def update_characters_node(self, state: StorytellerState) -> StorytellerState:
        """Patch changed characters."""
        story = coerce_story(state.get("story"))
        if not story:
            return {}

        character_ids: list[str] = parse_last_json(state.get("messages", [])).get(
            "character_ids", []
        )

        coros = [self._patch_character(state, cid) for cid in character_ids]
        results = await asyncio.gather(*coros, return_exceptions=True) if coros else []
        character_results = list(results)

        # Merge each successfully patched character back into the story.
        for char_result in character_results:
            if isinstance(char_result, BaseException) or char_result is None:
                continue
            patched = {c.object_id: c for c in char_result.characters}
            story = story.model_copy(
                update={"characters": [patched.get(c.object_id, c) for c in story.characters]}
            )

        n_updated = sum(1 for r in character_results if r and not isinstance(r, BaseException))
        logger.debug(f"update_characters_node: characters_updated={n_updated}")
        return {"story": story}

    def finalize_turn_node(self, state: StorytellerState) -> StorytellerState:
        """Fan-in: merge _turn_summary into story after parallel summary + update_characters,
        increment turn"""
        turn_summary = state.get("_turn_summary")
        turn_title = state.get("_turn_title")
        turn = (state.get("turn") or 0) + 1
        if not turn_summary:
            return {"turn": turn}
        story = coerce_story(state.get("story"))
        if not story:
            return {"_turn_summary": None, "_turn_title": None, "turn": turn}
        story = story.model_copy(update={"summary": turn_summary, "title": turn_title})
        logger.debug("finalize_turn_node: merged _turn_summary into story.summary")
        return {"story": story, "_turn_summary": None, "_turn_title": None, "turn": turn}

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

    # ── graph construction ─────────────────────────────────────────────────────

    def _add_nodes(self, graph: StateGraph) -> None:
        graph.add_node("greeting", self.greeting_node)
        graph.add_node("router", self.router_node)
        graph.add_node("story", self.story_node)
        graph.add_node("enter_story", self.enter_story)
        graph.add_node("memory_tool", self.memory_agent.graph)
        graph.add_node("update_characters", self.update_characters_node)
        graph.add_node("generate_character", self.character_generator.graph)
        graph.add_node("generate_world", self.world_generator.graph)
        graph.add_node("finalize_object", self.finalize_object)
        graph.add_node("summary", self.summary_node)
        graph.add_node("finalize_turn", self.finalize_turn_node)

    def _add_setup_phase_edges(self, graph: StateGraph) -> None:
        graph.add_conditional_edges(
            START,
            self.next_after_start,
            {
                "greeting": "greeting",
                "generate_world": "generate_world",
                "router": "router",
                "story": "story",
            },
        )
        graph.add_edge("greeting", END)
        graph.add_edge("generate_world", "finalize_object")
        graph.add_edge("generate_character", "finalize_object")
        graph.add_edge("finalize_object", "router")
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

    def _add_story_phase_edges(self, graph: StateGraph) -> None:
        graph.add_conditional_edges(
            "story",
            self.route_from_story,
            ["memory_tool", "summary", "update_characters"],
        )
        graph.add_edge("memory_tool", END)
        graph.add_edge("summary", "finalize_turn")
        graph.add_edge("update_characters", "finalize_turn")
        graph.add_edge("finalize_turn", END)

    def build_graph(self):
        graph = StateGraph(StorytellerState)
        self._add_nodes(graph)
        self._add_setup_phase_edges(graph)
        self._add_story_phase_edges(graph)
        return graph.compile(checkpointer=self.checkpointer)

    # ── public API ─────────────────────────────────────────────────────────────

    async def init(self, user_id: str, thread_id: str = None) -> str:
        """Emit the fixed greeting for a fresh session; must be called before the first tell()."""
        effective_thread = thread_id or user_id
        config = {"configurable": {"user_id": user_id, "thread_id": effective_thread}}
        result = await self.graph.ainvoke({"user_id": user_id}, config)
        return visible_response(result.get("messages", []))

    async def arun(self, query: str | Command, user_id: str, thread_id: str = None) -> dict:
        """Invoke the graph with a user message or a resume Command."""
        if isinstance(query, Command):
            input_data = query
        else:
            input_data = {"messages": [HumanMessage(content=query)], "user_id": user_id}
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id or user_id}}
        return await self.graph.ainvoke(input_data, config)

    async def _auto_save(self, result: dict, user_id: str, thread_id: str) -> None:
        """Persist current state to disk after each turn; silently skips if no story yet."""
        if not self.saves_dir:
            return
        story = coerce_story(result.get("story"))
        if story is None or story.world is None:
            return
        messages = result.get("messages", [])
        phase = result.get("phase", "world")
        turn = result.get("turn", 0)
        try:
            path = persistence.save_story(story, messages, phase, turn, thread_id, self.saves_dir)
            logger.debug(f"Auto-saved to {path}")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")

    async def tell(self, query: str, user_id: str, thread_id: str = None) -> str:
        # Check checkpoint for pending interrupts — avoids stale in-memory flag sending
        # normal story turns as Command(resume) into the object generator subgraph.
        effective_thread = thread_id or user_id
        config = {"configurable": {"user_id": user_id, "thread_id": effective_thread}}
        try:
            snap = await self.graph.aget_state(config)
            if snap.interrupts:
                logger.info(f"Sending command: {query}")
                result = await self.arun(Command(resume=query), user_id, thread_id)
            else:
                result = await self.arun(query, user_id, thread_id)
            await self._auto_save(result, user_id, effective_thread)
            interrupts = result.get("__interrupt__")
            self.waiting_for_feedback = bool(interrupts)
            if interrupts:
                if isinstance(interrupts, list) and interrupts:
                    first = interrupts[0]
                    val = first.value if hasattr(first, "value") else first
                    if isinstance(val, dict):
                        text = (val.get("draft") or val.get("hint") or "") or ""
                        text = strip_thinking(text.strip()) if text.strip() else str(val)
                        return text if text.strip() else str(val)
                    return str(val) if val is not None else ""
                return str(interrupts)
            return visible_response(result.get("messages", []))
        except Exception as e:
            logger.error(f"tell() failed: {e}")
            return STRUCTURED_OUTPUT_ERROR

    async def load(self, save_data: dict, user_id: str, thread_id: str = None) -> None:
        """Restore a saved session into this Storyteller, replacing in-memory state."""
        effective_thread = thread_id or user_id
        story = Story.model_validate(save_data["story"])
        messages = persistence.reconstruct_messages(save_data.get("messages", []))
        phase = save_data.get("phase", "world")
        turn = save_data.get("turn", 0)

        # Fresh checkpointer so no old state bleeds in.
        self.checkpointer = MemorySaver()
        self.graph = self.build_graph()

        config = {"configurable": {"user_id": user_id, "thread_id": effective_thread}}
        await self.graph.aupdate_state(
            config,
            {
                "messages": messages,
                "story": story,
                "phase": phase,
                "turn": turn,
                "user_id": user_id,
            },
        )

        # Rebuild InMemoryStore so the memory agent can look up objects and events.
        if self.memory_store:
            namespace_mem = (user_id, "memories")
            namespace_ev = (user_id, "events")
            if story.world:
                wo = WorldObject(world=story.world)
                self.memory_store.put(namespace_mem, wo.object_id, wo.model_dump())
            for co in story.characters:
                self.memory_store.put(namespace_mem, co.object_id, co.model_dump())
            for ev in story.events:
                self.memory_store.put(
                    namespace_ev, str(uuid.uuid4()), {"turn": ev.turn, "event": ev.event}
                )
        logger.info(f"Loaded save '{story.title}' (phase={phase}, turn={turn})")
