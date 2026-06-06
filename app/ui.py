"""Streamlit chat UI for storyteLLer."""

from __future__ import annotations

import asyncio
import json

import dotenv
import streamlit as st
from langgraph.store.memory import InMemoryStore

from app.graph import Storyteller
from app.state.schemas import coerce_story

dotenv.load_dotenv()

_BOOTSTRAP = "Begin world creation."
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


def _tell(text: str) -> str:
    return asyncio.run(st.session_state.storyteller.tell(text, _USER_ID))


def _parse_response(raw: str) -> str:
    try:
        return json.loads(raw).get("response", raw) or raw
    except (json.JSONDecodeError, AttributeError):
        return raw


def _current_story():
    config = {"configurable": {"user_id": _USER_ID, "thread_id": _USER_ID}}
    snap = asyncio.run(st.session_state.storyteller.graph.aget_state(config))
    return coerce_story((snap.values or {}).get("story"))


# ── sidebar ───────────────────────────────────────────────────────────────────


def _render_sidebar(story) -> None:
    with st.sidebar:
        st.title("storyteLLer")

        config = {"configurable": {"user_id": _USER_ID, "thread_id": _USER_ID}}
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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="storyteLLer", page_icon="📖", layout="wide")
    _init_session()

    # Bootstrap: fire the first message automatically
    if not st.session_state.bootstrapped:
        with st.spinner("Starting the story…"):
            raw = _tell(_BOOTSTRAP)
        text = _parse_response(raw)
        if text:
            st.session_state.messages.append({"role": "assistant", "content": text})
        st.session_state.bootstrapped = True

    story = _current_story()
    _render_sidebar(story)

    st.header("storyteLLer", divider="gray")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    if user_input := st.chat_input("Your reply…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Narrating…"):
                raw = _tell(user_input)
            text = _parse_response(raw)
            st.markdown(text)

        st.session_state.messages.append({"role": "assistant", "content": text})
        st.rerun()


if __name__ == "__main__":
    main()
