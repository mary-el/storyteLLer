"""Streamlit page for viewing and editing the current story's world and characters."""

from __future__ import annotations

import asyncio

import streamlit as st
from pydantic import BaseModel

from app.state.schemas import Character, Story, World, coerce_story


def _config() -> dict:
    return {
        "configurable": {
            "user_id": st.session_state.user_id,
            "thread_id": st.session_state.thread_id,
        }
    }


def _snapshot():
    return asyncio.run(st.session_state.storyteller.graph.aget_state(_config()))


def _save(story: Story) -> None:
    asyncio.run(
        st.session_state.storyteller.update_story(
            story, st.session_state.user_id, st.session_state.thread_id
        )
    )


def _model_fields(
    model: BaseModel, locked: list[str], prefix: str, disabled: bool
) -> tuple[dict, list[str]]:
    """Render one text field + lock checkbox per model field; return (values, locked)."""
    values: dict = {}
    new_locked: list[str] = []
    for name, info in type(model).model_fields.items():
        current = getattr(model, name)
        is_locked = name in locked
        # Keys include the current value so widgets refresh when the narrator changes it.
        key = f"{prefix}_{name}_{hash((current, is_locked))}"
        label = name.replace("_", " ").capitalize()
        field_col, lock_col = st.columns([6, 1], vertical_alignment="center")
        with field_col:
            widget = st.text_input if name == "name" else st.text_area
            values[name] = widget(
                label, value=current, help=info.description, disabled=disabled, key=key
            )
        with lock_col:
            if st.checkbox(
                "Lock",
                value=is_locked,
                key=f"{key}_lock",
                disabled=disabled,
                help="Keep the narrator from changing this field",
            ):
                new_locked.append(name)
    return values, new_locked


def _render_world(story: Story, disabled: bool) -> None:
    st.subheader(f"World: {story.world.name}")
    with st.form("world_form", border=True):
        values, locked = _model_fields(story.world, story.world_locked_fields, "world", disabled)
        if st.form_submit_button("Save world", disabled=disabled, type="primary"):
            _save(
                story.model_copy(update={"world": World(**values), "world_locked_fields": locked})
            )
            st.toast("World saved")
            st.rerun()


def _render_characters(story: Story, disabled: bool) -> None:
    head_col, add_col = st.columns([6, 1], vertical_alignment="bottom")
    with head_col:
        st.subheader(f"Characters ({len(story.characters)})")
    with add_col:
        st.button("Add character", disabled=True, help="Coming soon", use_container_width=True)

    if not story.characters:
        st.caption("No characters yet. They appear as the narrator introduces them.")
        return

    for co in story.characters:
        with st.expander(co.character.name):
            with st.form(f"character_form_{co.object_id}", border=False):
                values, locked = _model_fields(
                    co.character, co.locked_fields, co.object_id, disabled
                )
                if st.form_submit_button("Save character", disabled=disabled, type="primary"):
                    updated = co.model_copy(
                        update={"character": Character(**values), "locked_fields": locked}
                    )
                    characters = [
                        updated if c.object_id == co.object_id else c for c in story.characters
                    ]
                    _save(story.model_copy(update={"characters": characters}))
                    st.toast(f"{values['name']} saved")
                    st.rerun()
            st.button("Delete", key=f"delete_{co.object_id}", disabled=True, help="Coming soon")


def render() -> None:
    st.header("World & Characters", divider="gray")
    snap = _snapshot()
    story = coerce_story((snap.values or {}).get("story"))
    if story is None or story.world is None:
        st.caption("No world yet. Create one in Chat first.")
        return

    disabled = bool(snap.interrupts)
    if disabled:
        st.info("Finish the current step in Chat before editing.")

    _render_world(story, disabled)
    st.divider()
    _render_characters(story, disabled)

if __name__ == "__main__":
    render()
