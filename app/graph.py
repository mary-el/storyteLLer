from pathlib import Path
from typing import AsyncIterator, Literal, Optional

import dotenv
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app import persistence
from app.agents.character_agent import CharacterAgent
from app.agents.world_gen import WorldGenerator
from app.config import AppConfig, load_app_config
from app.state.schemas import Story, StoryEvent, StorytellerState, TurnSummary, coerce_story
from app.utils import (
    STRUCTURED_OUTPUT_ERROR,
    EventResponse,
    StoryResponse,
    invoke_structured,
    logger,
    message_text,
    strip_thinking,
    trim_history,
    visible_response,
)

dotenv.load_dotenv()


def _story_message_turns(messages: list, turn: int) -> dict[str, int]:
    """Map each story user-message id to the turn that line produced.

    The newest story line is ``turn``. Earlier story lines count backward.
    Lines before the world was created are not story turns.
    """
    world_at = next(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, SystemMessage) and "OBJECT_CREATED" in message_text(message)
        ),
        None,
    )
    start = 0 if world_at is None else world_at + 1
    story_ids = [
        message.id
        for message in messages[start:]
        if isinstance(message, HumanMessage) and message.id
    ]
    return {
        message_id: turn - (len(story_ids) - 1 - index)
        for index, message_id in enumerate(story_ids)
    }


def _apply_event_cut(story: Story, cut_turn: int) -> tuple[int, Story]:
    """Drop events and summaries from ``cut_turn`` onward. World and characters stay."""
    if cut_turn < 1:
        cut_turn = 1
    kept = [item for item in story.turn_summaries if item.turn < cut_turn]
    events = [event for event in story.events if event.turn < cut_turn]
    if kept:
        summary = kept[-1].summary
        title = kept[-1].title
        restored_turn = max(kept[-1].turn, cut_turn - 1)
    elif cut_turn <= 1:
        summary = "None"
        title = "Untitled"
        restored_turn = 0
    else:
        summary = story.summary
        title = story.title
        restored_turn = cut_turn - 1
    return restored_turn, story.model_copy(
        update={
            "summary": summary,
            "title": title,
            "events": events,
            "turn_summaries": kept,
        }
    )


def _rewound_story(
    story: Optional[Story],
    user_message_id: str,
    *,
    messages: list | None = None,
    turn: int = 0,
) -> tuple[int, Optional[Story]]:
    """Drop events and summaries from the chosen user message's turn onward.

    World, characters, and lock fields stay. A snapshotted line cuts at its
    recorded turn. A story line with no snapshot still cuts at the turn counted
    back from the current turn, so later events leave with that line. A
    world-building line clears the story events.
    """
    if story is None:
        return 0, None

    target_turns = [
        item.turn for item in story.turn_summaries if item.user_message_id == user_message_id
    ]
    if target_turns:
        return _apply_event_cut(story, min(target_turns))

    if messages:
        inferred = _story_message_turns(messages, turn)
        if user_message_id in inferred:
            return _apply_event_cut(story, inferred[user_message_id])
        return _apply_event_cut(story, 1)

    return turn, story


class Storyteller:
    def __init__(
        self,
        langdev: bool = False,
        config: Optional[AppConfig] = None,
    ) -> None:
        self.config = config or load_app_config()
        self.saves_dir: Optional[Path] = (
            Path(self.config.saves_dir) if self.config.saves_dir else None
        )
        self.llm = ChatOpenAI(**self.config.llm)
        self.checkpointer = MemorySaver() if not langdev else None
        self.langdev = langdev
        self.character_agent = CharacterAgent(
            self.llm,
            app_config=self.config,
        )
        self.world_generator = WorldGenerator(
            self.llm,
            self.checkpointer,
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

    def _format_characters(self, story: Story) -> str:
        return "\n".join(
            f"Character ({co.object_id}):\n{co.model_dump_json(indent=2)}"
            for co in story.characters
        )

    def _format_previous_events(self, story: Story) -> str:
        if not story.events:
            return "None yet."
        limit = self.config.story_narrator.max_previous_events
        return "\n".join(f"- (turn {e.turn}) {e.event}" for e in story.events[-limit:])

    def _get_story_context(self, story: Story) -> str:
        return self.config.story_narrator.system_prompt.format(
            world=story.world.model_dump_json(indent=2),
            story_summary=story.summary,
            characters=self._format_characters(story),
            events=self._format_previous_events(story),
        )

    # ── nodes ──────────────────────────────────────────────────────────────────

    def greeting_node(self, state: StorytellerState) -> StorytellerState:
        return {"messages": [AIMessage(content=self.config.greeting)]}

    async def story_node(self, state: StorytellerState) -> StorytellerState:
        story = state.get("story")
        context = self._get_story_context(story)
        messages = trim_history(
            state["messages"],
            self.story_max_len,
            include_system=True,
        )
        logger.debug("routing to story narrator")
        llm_messages = [SystemMessage(content=context), *messages]
        try:
            response = await invoke_structured(self.llm, StoryResponse, llm_messages)
        except Exception as e:
            logger.error(f"story_node failed: {e}")
            response = StoryResponse(
                response=STRUCTURED_OUTPUT_ERROR,
                character_commands=[],
                add_event=False,
            )
        return {
            "messages": [AIMessage(content=response.response)],
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

        messages = trim_history(
            state.get("messages", []),
            self.config.story_update.max_trim_tokens,
        )
        transcript_lines: list[str] = []
        for m in messages:
            text = message_text(m).strip()
            if not text:
                continue
            role = "User" if isinstance(m, HumanMessage) else "Narrator"
            transcript_lines.append(f"{role}: {text}")
        transcript = "\n".join(transcript_lines)

        archive_prompt = self.config.story_update.archive_prompt.format(
            previous_summary=story.summary,
            previous_events=self._format_previous_events(story),
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

        user_message_id = ""
        for message in reversed(state.get("messages") or []):
            if isinstance(message, HumanMessage):
                user_message_id = message.id or ""
                break

        patch: dict = {}
        if turn_summary:
            summary_text = turn_summary
            title_text = turn_title or story.title
            patch["summary"] = summary_text
            patch["title"] = title_text
            logger.debug("finalize_turn_node: merged _turn_summary into story.summary")
        else:
            summary_text = story.summary
            title_text = story.title
        patch["turn_summaries"] = [
            *story.turn_summaries,
            TurnSummary(
                turn=turn,
                summary=summary_text,
                title=title_text,
                user_message_id=user_message_id,
            ),
        ]
        if new_event:
            stamped = StoryEvent(turn=turn, event=new_event.event)
            patch["events"] = [*story.events, stamped]
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
            ["archive", "character_agent", "finalize_turn"],
        )
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

    @staticmethod
    def _reply_text(values: dict, interrupts) -> str:
        """User-facing text for a finished turn: the interrupt prompt if paused, else last AI reply."""
        if interrupts:
            if isinstance(interrupts, (list, tuple)):
                first = interrupts[0]
                val = first.value if hasattr(first, "value") else first
                if isinstance(val, dict):
                    text = (val.get("draft") or val.get("hint") or "") or ""
                    text = strip_thinking(text.strip()) if text.strip() else str(val)
                    return text if text.strip() else str(val)
                return str(val) if val is not None else ""
            return str(interrupts)
        return visible_response(values.get("messages", []))

    async def tell_stream(
        self, query: str, user_id: str, thread_id: str = None
    ) -> AsyncIterator[dict]:
        """Run one turn, yielding {"type": "node"} per graph step, then a final "message" or "error"."""
        # Check checkpoint for pending interrupts — avoids stale in-memory flag sending
        # normal story turns as Command(resume) into the object generator subgraph.
        effective_thread = thread_id or user_id
        config = {"configurable": {"user_id": user_id, "thread_id": effective_thread}}
        try:
            snap = await self.graph.aget_state(config)
            if snap.interrupts:
                logger.info(f"Sending command: {query}")
                input_data = Command(resume=query)
            else:
                input_data = {"messages": [HumanMessage(content=query)], "user_id": user_id}
            async for _namespace, update in self.graph.astream(
                input_data, config, stream_mode="updates", subgraphs=True
            ):
                for node in update:
                    if node != "__interrupt__":
                        yield {"type": "node", "node": node}
            snap = await self.graph.aget_state(config)
            await self._auto_save(snap.values, user_id, effective_thread)
            self.waiting_for_feedback = bool(snap.interrupts)
            yield {
                "type": "message",
                "text": self._reply_text(snap.values, snap.interrupts),
                "awaiting_feedback": bool(snap.interrupts),
            }
        except Exception as e:
            logger.error(f"tell() failed: {e}")
            yield {"type": "error", "text": STRUCTURED_OUTPUT_ERROR}

    async def tell(self, query: str, user_id: str, thread_id: str = None) -> str:
        text = STRUCTURED_OUTPUT_ERROR
        async for event in self.tell_stream(query, user_id, thread_id):
            if event["type"] in ("message", "error"):
                text = event["text"]
        return text

    async def get_state(self, user_id: str, thread_id: str = None) -> Optional[dict]:
        """Snapshot of a thread for display; None if the thread has never run."""
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id or user_id}}
        snap = await self.graph.aget_state(config)
        values = snap.values or {}
        if not values:
            return None
        history: list[dict] = []
        for m in values.get("messages", []):
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                continue
            text = strip_thinking(message_text(m))
            if text:
                history.append({"role": role, "content": text, "id": m.id or ""})
        story = coerce_story(values.get("story"))
        return {
            "phase": values.get("phase", "world"),
            "turn": values.get("turn", 0),
            "story": story.model_dump(mode="json") if story else None,
            "awaiting_feedback": bool(snap.interrupts),
            "pending_prompt": (
                self._reply_text(values, snap.interrupts) if snap.interrupts else None
            ),
            "messages": history,
        }

    async def load(self, save_data: dict, user_id: str, thread_id: str = None) -> None:
        """Restore a saved session into this Storyteller, replacing in-memory state."""
        effective_thread = thread_id or user_id
        story = Story.model_validate(save_data["story"])
        messages = persistence.reconstruct_messages(save_data.get("messages", []))
        phase = save_data.get("phase", "world")
        turn = save_data.get("turn", 0)

        # Reset only this thread so no old state bleeds in; other threads stay intact.
        if self.checkpointer is not None:
            await self.checkpointer.adelete_thread(effective_thread)

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
        logger.info(f"Loaded save '{story.title}' (phase={phase}, turn={turn})")

    async def update_story(self, story: Story, user_id: str, thread_id: str = None) -> None:
        """Replace the thread's story with a manually edited one and persist it."""
        effective_thread = thread_id or user_id
        config = {"configurable": {"user_id": user_id, "thread_id": effective_thread}}
        # As finalize_turn, so the next tell() starts a fresh turn from START.
        await self.graph.aupdate_state(config, {"story": story}, as_node="finalize_turn")
        snap = await self.graph.aget_state(config)
        await self._auto_save(snap.values, user_id, effective_thread)

    async def rewind_to_user_message(
        self, user_id: str, thread_id: str, user_message_id: str
    ) -> None:
        """Drop a user message and everything after it, then persist.

        Raises KeyError when the id is missing or is not a user message.
        """
        effective_thread = thread_id or user_id
        config = {"configurable": {"user_id": user_id, "thread_id": effective_thread}}
        snap = await self.graph.aget_state(config)
        messages = list((snap.values or {}).get("messages") or [])
        cut = next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, HumanMessage) and message.id == user_message_id
            ),
            None,
        )
        if cut is None:
            raise KeyError(user_message_id)

        turn, story = _rewound_story(
            coerce_story((snap.values or {}).get("story")),
            user_message_id,
            messages=messages,
            turn=(snap.values or {}).get("turn") or 0,
        )
        update: dict = {
            "messages": [RemoveMessage(id=message.id) for message in messages[cut:] if message.id],
            "turn": turn,
        }
        if story is not None:
            update["story"] = story
        # As finalize_turn, so the next tell() starts a fresh turn from START.
        await self.graph.aupdate_state(config, update, as_node="finalize_turn")
        saved = await self.graph.aget_state(config)
        await self._auto_save(saved.values, user_id, effective_thread)
