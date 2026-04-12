import uuid
from typing import Any, Literal, NotRequired, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class BaseObject(BaseModel):
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class Character(BaseModel):
    """A character in the story. Do not add nonexistent fields."""

    name: str = Field("Unknown", description="The name of the character")
    appearance: str = Field("Unknown", description="The appearance of the character")
    personality: str = Field("Unknown", description="The personality of the character")
    backstory: str = Field("Unknown", description="The backstory of the character")
    goals: str = Field("Unknown", description="The goals of the character")
    abilities: str = Field("Unknown", description="The abilities of the character")
    inventory: str = Field("Empty", description="The inventory of the character")
    relationships: str = Field("Unknown", description="The relationships of the character")


class World(BaseModel):
    """A world in the story. Do not add nonexistent fields."""

    name: str = Field("Unknown", description="The name of the world")
    description: str = Field("Unknown", description="The description of the world")
    history: str = Field("Unknown", description="The history of the world")
    culture: str = Field("Unknown", description="The culture of the world")
    technology: str = Field("Unknown", description="The technology of the world")
    politics: str = Field("Unknown", description="The politics of the world")
    economy: str = Field("Unknown", description="The economy of the world")


class CharacterObject(BaseObject):
    character: Character = Field(default_factory=Character)


class WorldObject(BaseObject):
    world: World = Field(default_factory=World)


StoryStep = Literal["world", "characters", "story"]


class Story(BaseModel):
    """One world, multiple characters, running summary."""

    story_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field("Untitled")
    world: Optional[World] = None
    characters: list[CharacterObject] = Field(default_factory=list)
    summary: str = Field("")


def coerce_story(raw: Any) -> Optional[Story]:
    """Normalize checkpoint / wire values into a Story model."""
    if raw is None:
        return None
    if isinstance(raw, Story):
        return raw
    if isinstance(raw, dict):
        return Story.model_validate(raw)
    return None


class State(MessagesState):
    """State for ObjectGenerator subgraphs and Storyteller graph"""

    status: NotRequired[Literal["created", "in_progress"]]
    user_id: NotRequired[str]
    generated_object: NotRequired[BaseObject]


class StorytellerState(State):
    """Top-level graph state: running story aggregate + pipeline phase."""

    story: NotRequired[Story]
    phase: NotRequired[StoryStep]
