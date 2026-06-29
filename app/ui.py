"""Streamlit chat UI for storyteLLer."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

import dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

from app import persistence
from app.graph import Storyteller
from app.state.schemas import coerce_story
from app.utils import STRUCTURED_OUTPUT_ERROR, logger, split_thinking

dotenv.load_dotenv()

_USER_ID = "1"


# ── session init ──────────────────────────────────────────────────────────────


def _init_session() -> None:
    if "store" not in st.session_state:
        st.session_state.store = InMemoryStore()
    if "storyteller" not in st.session_state:
        st.session_state.storyteller = Storyteller(memory_store=st.session_state.store)
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []
    if "bootstrapped" not in st.session_state:
        st.session_state.bootstrapped = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


def _thread_id() -> str:
    return st.session_state.thread_id


def _tell(text: str) -> str:
    return asyncio.run(st.session_state.storyteller.tell(text, _USER_ID, _thread_id()))


def _parse_response(raw: str) -> tuple[str, str | None]:
    thinking, visible = split_thinking(raw)
    try:
        parsed = json.loads(visible)
        response = parsed.get("response", visible) or visible
        return response, thinking
    except (json.JSONDecodeError, AttributeError):
        return visible, thinking


def _render_assistant_message(content: str, thinking: str | None = None) -> None:
    st.markdown(content)
    if thinking:
        with st.expander("Model reasoning", expanded=False):
            st.markdown(thinking)


def _current_story():
    config = {"configurable": {"user_id": _USER_ID, "thread_id": _thread_id()}}
    snap = asyncio.run(st.session_state.storyteller.graph.aget_state(config))
    return coerce_story((snap.values or {}).get("story"))


# ── sidebar ───────────────────────────────────────────────────────────────────


def _messages_from_save(save_data: dict) -> list[dict]:
    """Rebuild UI chat history from persisted graph messages."""
    restored: list[dict] = []
    for msg in persistence.reconstruct_messages(save_data.get("messages", [])):
        if isinstance(msg, HumanMessage):
            text = str(msg.content).strip() if msg.content is not None else ""
            if text:
                restored.append({"role": "user", "content": text})
            continue
        if isinstance(msg, AIMessage):
            text, thinking = _parse_response(str(msg.content) if msg.content is not None else "")
            if text:
                restored.append({"role": "assistant", "content": text, "thinking": thinking})
    return restored


def _load_save(save_meta: dict) -> None:
    """Load a save file into a fresh Storyteller and restore chat history."""
    save_data = persistence.load_story(save_meta["path"])
    active_thread_id = save_data.get("thread_id") or str(uuid.uuid4())
    new_store = InMemoryStore()
    new_storyteller = Storyteller(memory_store=new_store)
    asyncio.run(new_storyteller.load(save_data, _USER_ID, active_thread_id))
    st.session_state.store = new_store
    st.session_state.storyteller = new_storyteller
    st.session_state.thread_id = active_thread_id
    st.session_state.messages = _messages_from_save(save_data)
    st.session_state.bootstrapped = True  # skip auto-bootstrap; story already loaded


def _new_story() -> None:
    """Start a fresh story on a new thread id."""
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.bootstrapped = False


def _render_load_section() -> None:
    saves_dir = st.session_state.storyteller.saves_dir
    saves = persistence.list_saves(saves_dir) if saves_dir else []
    with st.expander(f"Load a world ({len(saves)} saved)", expanded=False):
        if not saves:
            st.caption("No saved worlds yet. Play a turn to create one.")
            return
        for save in saves:
            saved_dt = save["saved_at"]
            try:
                saved_dt = datetime.fromisoformat(save["saved_at"]).strftime("%b %d %H:%M")
            except ValueError:
                pass
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{save['title']}**")
                st.caption(f"Phase: {save['phase']} · Turn {save['turn']} · {saved_dt}")
            with col2:
                if st.button("Resume", key=f"load_{save['story_id']}"):
                    with st.spinner(f"Loading {save['title']}…"):
                        _load_save(save)
                    st.rerun()


def _render_sidebar(story) -> None:
    with st.sidebar:
        st.title("storyteLLer")
        if st.button("New Story", use_container_width=True):
            _new_story()
            st.rerun()

        config = {"configurable": {"user_id": _USER_ID, "thread_id": _thread_id()}}
        snap = asyncio.run(st.session_state.storyteller.graph.aget_state(config))
        phase = (snap.values or {}).get("phase", "world")
        phase_color = {"world": "🌍", "characters": "🧙", "story": "📖"}.get(phase, "•")
        st.markdown(f"**Phase:** {phase_color} `{phase}`")

        st.divider()

        if story and story.world:
            st.markdown(f"**World:** {story.world.name}")
            with st.expander("World details"):
                st.json(story.world.model_dump(), expanded=False)
        else:
            st.caption("No world yet.")

        st.divider()

        if story and story.characters:
            st.markdown(f"**Characters** ({len(story.characters)})")
            for co in story.characters:
                with st.expander(co.character.name):
                    st.json(co.character.model_dump(), expanded=False)
        else:
            st.caption("No characters yet.")

        st.divider()

        if story and story.summary:
            st.markdown("**Story summary**")
            st.caption(story.summary)

        if story and story.events:
            with st.expander(f"Events ({len(story.events)})"):
                for ev in reversed(story.events):
                    st.markdown(f"- **Turn {ev.turn}:** {ev.event}")

        st.divider()
        _render_load_section()


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="storyteLLer", page_icon="📖", layout="wide")
    _init_session()

    # Show fixed greeting on a fresh session (no LLM call needed).
    if not st.session_state.bootstrapped:
        greeting = asyncio.run(st.session_state.storyteller.init(_USER_ID, _thread_id()))
        if greeting:
            st.session_state.messages.append(
                {"role": "assistant", "content": greeting, "thinking": None}
            )
        st.session_state.bootstrapped = True

    story = _current_story()
    _render_sidebar(story)

    st.header("storyteLLer", divider="gray")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_message(msg["content"], msg.get("thinking"))
            else:
                st.markdown(msg["content"])

    # User input
    if user_input := st.chat_input("Your reply…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Narrating…"):
                try:
                    raw = _tell(user_input)
                except Exception as e:
                    logger.error(f"UI tell failed: {e}")
                    st.error(STRUCTURED_OUTPUT_ERROR)
                    raw = STRUCTURED_OUTPUT_ERROR
            text, thinking = _parse_response(raw)
            _render_assistant_message(text, thinking)

        st.session_state.messages.append(
            {"role": "assistant", "content": text, "thinking": thinking}
        )
        st.rerun()


if __name__ == "__main__":
    main()
