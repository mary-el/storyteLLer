import uuid
from pathlib import Path
from typing import Literal, Optional

import dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from app import persistence
from app.agents.character_agent import CharacterAgent
from app.agents.memory_agent import MemoryAgent
from app.agents.world_gen import WorldGenerator
from app.config import AppConfig, load_app_config
from app.state.schemas import Story, StoryEvent, StorytellerState, WorldObject
from app.utils import (
    STRUCTURED_OUTPUT_ERROR,
    EventResponse,
    StoryResponse,
    count_tokens,
    invoke_structured,
    logger,
    message_text,
    strip_thinking,
    visible_response,
)

dotenv.load_dotenv()


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
        self.character_agent = CharacterAgent(
            self.llm,
            memory_store,
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
        self.story_max_len = self.config.story_narrator.max_trim_tokens
        self.graph = self.build_graph()
        self.waiting_for_feedback = False

    # ── routing ────────────────────────────────────────────────────────────────

    def next_after_start(
        self, state: StorytellerState
    ) -> Literal["greeting", "generate_world", "story"]:
        story = state.get("story")
        if story is None or story.world is None:
            if not state.get("messages"):
                return "greeting"
            return "generate_world"
        return "story"

    def after_finalize_world(self, state: StorytellerState) -> Literal["story", "generate_world"]:
        """Guard against entering `story` without a world — e.g. if generate_world's wizard
        was re-entered without actually finishing (resumed with a plain message instead of a
        proper Command, leaving `generated_object`/`story` unset for this turn)."""
        story = state.get("story")
        if story is None or story.world is None:
            logger.warning(
                "after_finalize_world: no world in state yet; routing back to generate_world "
                "instead of entering story"
            )
            return "generate_world"
        return "story"

    def route_from_story(self, state: StorytellerState) -> str | list[str]:
        node = state.get("_pending_node", "dialogue")
        if node == "memory_tool":
            return "memory_tool"
        commands = state.get("_pending_character_commands") or []
        add_event: bool = state.get("_pending_add_event") or False
        logger.debug(f"route_from_story: commands={commands}, add_event={add_event}")
        branches: list[str] = []
        if commands:
            branches.append("character_agent")
        if add_event:
            branches.append("archive")
        return branches or ["finalize_turn"]

    # ── story context ──────────────────────────────────────────────────────────

    def _get_story_context(self, story: Story) -> str:
        characters = "\n".join(
            f"Character ({co.object_id}):\n{co.model_dump_json(indent=2)}"
            for co in story.characters
        )
        return self.config.story_narrator.system_prompt.format(
            world=story.world.model_dump_json(indent=2),
            story_summary=story.summary,
            characters=characters,
        )

    # ── nodes ──────────────────────────────────────────────────────────────────

    def greeting_node(self, state: StorytellerState) -> StorytellerState:
        return {"messages": [AIMessage(content=self.config.greeting)]}

    async def story_node(self, state: StorytellerState) -> StorytellerState:
        story = state.get("story")
        context = self._get_story_context(story)
        messages = trim_messages(
            state["messages"],
            max_tokens=self.story_max_len,
            token_counter=count_tokens,
            strategy="last",
            start_on="human",
            include_system=True,
        )
        logger.debug("routing to story narrator")
        llm_messages = [SystemMessage(content=context), *messages]
        try:
            response = await invoke_structured(self.llm, StoryResponse, llm_messages)
        except Exception as e:
            logger.error(f"story_node failed: {e}")
            response = StoryResponse(
                node="dialogue",
                response=STRUCTURED_OUTPUT_ERROR,
                character_commands=[],
                add_event=False,
            )
        return {
            "messages": [AIMessage(content=response.response)],
            "_pending_node": response.node,
            "_pending_response": response.response,
            "_pending_character_commands": response.character_commands,
            "_pending_add_event": response.add_event,
            "phase": "story",
            "status": None,
        }

    async def archive_node(self, state: StorytellerState) -> StorytellerState:
        """Combined archive: always generates a summary; records a key event when add_event=True."""
        story = state.get("story")
        if not story:
            return {}

        add_event: bool = state.get("_pending_add_event") or False

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

        previous_events = (
            "\n".join(f"- (turn {e.turn}) {e.event}" for e in story.events)
            if story.events
            else "None yet."
        )

        archive_prompt = self.config.story_update.archive_prompt.format(
            previous_summary=story.summary,
            previous_events=previous_events,
            transcript=transcript,
            title=story.title,
        )
        try:
            response = await invoke_structured(
                self.llm, EventResponse, [SystemMessage(content=archive_prompt)]
            )
        except Exception as e:
            logger.error(f"archive_node failed: {e}")
            return {}

        new_event: StoryEvent | None = None
        if add_event and response.event.strip():
            new_event = StoryEvent(
                turn=0, event=response.event.strip()
            )  # turn stamped by finalize_turn
            logger.debug(f"archive_node: queued event: {new_event.event!r}")

        logger.debug("archive_node: queued summary update")
        return {
            "_turn_summary": response.summary.strip(),
            "_turn_title": response.title,
            "_new_event": new_event,
        }

    async def finalize_turn_node(self, state: StorytellerState) -> StorytellerState:
        """Fan-in: merge _turn_summary + _new_event into story; increment turn."""
        turn_summary = state.get("_turn_summary")
        turn_title = state.get("_turn_title")
        new_event: StoryEvent | None = state.get("_new_event")  # type: ignore[assignment]
        turn = (state.get("turn") or 0) + 1
        story = state.get("story")

        updates: dict = {
            "turn": turn,
            "_turn_summary": None,
            "_turn_title": None,
            "_new_event": None,
        }

        if not story:
            return updates

        patch: dict = {}
        if turn_summary:
            patch["summary"] = turn_summary
            patch["title"] = turn_title or story.title
            logger.debug("finalize_turn_node: merged _turn_summary into story.summary")
        if new_event:
            stamped = StoryEvent(turn=turn, event=new_event.event)
            patch["events"] = [*story.events, stamped]
            if self.memory_store:
                namespace = (state.get("user_id", "default"), "events")
                await self.memory_store.aput(
                    namespace,
                    str(uuid.uuid4()),
                    {"turn": stamped.turn, "event": stamped.event},
                )
            logger.debug(f"finalize_turn_node: archived event at turn {turn}: {stamped.event!r}")
        if patch:
            updates["story"] = story.model_copy(update=patch)

        return updates

    def finalize_world_node(self, state: StorytellerState) -> StorytellerState:
        """Merge the freshly generated world into the story and enter the story phase."""
        logger.debug("Finalizing world generation")
        generated_object = state.get("generated_object")
        if not generated_object:
            return {}
        story = state.get("story") or Story()
        story = story.model_copy(update={"world": generated_object.world})
        return {
            "messages": [
                SystemMessage(
                    content=f"SYSTEM_EVENT: OBJECT_CREATED: {generated_object.model_dump_json()}"
                )
            ],
            "generated_object": None,
            "story": story,
            "phase": "story",
        }

    # ── graph construction ─────────────────────────────────────────────────────

    def _add_nodes(self, graph: StateGraph) -> None:
        graph.add_node("greeting", self.greeting_node)
        graph.add_node("story", self.story_node)
        graph.add_node("memory_tool", self.memory_agent.graph)
        graph.add_node("archive", self.archive_node)
        graph.add_node("character_agent", self.character_agent.run)
        graph.add_node("generate_world", self.world_generator.graph)
        graph.add_node("finalize_world", self.finalize_world_node)
        graph.add_node("finalize_turn", self.finalize_turn_node)

    def _add_world_phase_edges(self, graph: StateGraph) -> None:
        graph.add_conditional_edges(
            START,
            self.next_after_start,
            {
                "greeting": "greeting",
                "generate_world": "generate_world",
                "story": "story",
            },
        )
        graph.add_edge("greeting", END)
        graph.add_edge("generate_world", "finalize_world")
        graph.add_conditional_edges(
            "finalize_world",
            self.after_finalize_world,
            {"story": "story", "generate_world": "generate_world"},
        )

    def _add_story_phase_edges(self, graph: StateGraph) -> None:
        graph.add_conditional_edges(
            "story",
            self.route_from_story,
            ["memory_tool", "archive", "character_agent", "finalize_turn"],
        )
        graph.add_edge("memory_tool", "story")
        graph.add_edge("archive", "finalize_turn")
        graph.add_edge("character_agent", "finalize_turn")
        graph.add_edge("finalize_turn", END)

    def build_graph(self):
        graph = StateGraph(StorytellerState)
        self._add_nodes(graph)
        self._add_world_phase_edges(graph)
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
        story = result.get("story")
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
                await self.memory_store.aput(namespace_mem, wo.object_id, wo.model_dump())
            for co in story.characters:
                await self.memory_store.aput(namespace_mem, co.object_id, co.model_dump())
            for ev in story.events:
                await self.memory_store.aput(
                    namespace_ev, str(uuid.uuid4()), {"turn": ev.turn, "event": ev.event}
                )
        logger.info(f"Loaded save '{story.title}' (phase={phase}, turn={turn})")
