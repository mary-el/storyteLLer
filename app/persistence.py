"""Disk persistence for storyteLLer saves."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from app.state.schemas import Story


def save_story(
    story: Story,
    messages: list[BaseMessage],
    phase: str,
    turn: int,
    thread_id: str,
    saves_dir: str | Path,
) -> Path:
    """Serialize current session to <saves_dir>/<story_id>.json (overwrites on each turn)."""
    saves_path = Path(saves_dir)
    saves_path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "save_id": str(uuid.uuid4()),
        "story_id": story.story_id,
        "title": story.title,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "phase": phase,
        "turn": turn,
        "story": story.model_dump(mode="json"),
        "messages": [message_to_dict(m) for m in messages],
    }

    file_path = saves_path / f"{story.story_id}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def load_story(save_path: str | Path) -> dict[str, Any]:
    """Load and return a raw save dict from disk."""
    return json.loads(Path(save_path).read_text(encoding="utf-8"))


def list_saves(saves_dir: str | Path) -> list[dict[str, Any]]:
    """Return save metadata for all worlds, sorted newest-first."""
    saves_path = Path(saves_dir)
    if not saves_path.exists():
        return []

    saves: list[dict[str, Any]] = []
    for file in saves_path.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            saves.append(
                {
                    "story_id": data.get("story_id", ""),
                    "title": data.get("title", "Untitled"),
                    "saved_at": data.get("saved_at", ""),
                    "phase": data.get("phase", ""),
                    "turn": data.get("turn", 0),
                    "path": str(file),
                }
            )
        except Exception:
            continue

    saves.sort(key=lambda s: s["saved_at"], reverse=True)
    return saves


def reconstruct_messages(raw_messages: list[dict]) -> list:
    """Deserialize a list of message dicts back into LangChain message objects."""
    return messages_from_dict(raw_messages)
