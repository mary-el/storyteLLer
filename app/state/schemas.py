from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, NotRequired
from langgraph.graph import StateGraph, MessagesState, START, END
import uuid
class BaseObject(BaseModel):
    object_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
class Character(BaseObject):
    name: str = Field("Unknown", description="The name of the character")
    appearance: str = Field("Unknown", description="The appearance of the character")
    personality: str = Field("Unknown", description="The personality of the character")
    backstory: str = Field("Unknown", description="The backstory of the character")
    goals: str = Field("Unknown", description="The goals of the character")
    abilities: str = Field("Unknown", description="The abilities of the character")
    inventory: str = Field("Empty", description="The inventory of the character")
    relationships: str = Field("Unknown", description="The relationships of the character")

class World(BaseObject):
    name: str = Field("Unknown", description="The name of the world")
    description: str = Field("Unknown", description="The description of the world")
    history: str = Field("Unknown", description="The history of the world")
    culture: str = Field("Unknown", description="The culture of the world")
    technology: str = Field("Unknown", description="The technology of the world")
    politics: str = Field("Unknown", description="The politics of the world")
    economy: str = Field("Unknown", description="The economy of the world")

def reducer(left: Character | None, right: Character | None) -> Character:
    """Reducer function to ensure character is always initialized"""
    if right is not None:
        return right
    if left is not None:
        return left
    return Character()

class State(MessagesState):
    status: NotRequired[Literal["created", "in_progress"]]
    user_id: NotRequired[str]
    generated_object: NotRequired[BaseObject]

class GenerateCharacterState(State):
    pass

class GenerateWorldState(State):
    pass
