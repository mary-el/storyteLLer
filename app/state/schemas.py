from pydantic import BaseModel, Field
from typing import TypedDict, Literal, Annotated, NotRequired
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import BaseMessage

class Character(BaseModel):
    name: str = Field("Unknown", description="The name of the character")
    appearance: str = Field("Unknown", description="The appearance of the character")
    personality: str = Field("Unknown", description="The personality of the character")
    backstory: str = Field("Unknown", description="The backstory of the character")
    goals: str = Field("Unknown", description="The goals of the character")
    abilities: str = Field("Unknown", description="The abilities of the character")
    inventory: str = Field("Empty", description="The inventory of the character")
    relationships: str = Field("Unknown", description="The relationships of the character")

def character_reducer(left: Character | None, right: Character | None) -> Character:
    """Reducer function to ensure character is always initialized"""
    if right is not None:
        return right
    if left is not None:
        return left
    return Character()

class GenerateCharacterState(MessagesState):
    character: Annotated[Character, character_reducer]
    status: NotRequired[Literal["created", "in_progress"]]
