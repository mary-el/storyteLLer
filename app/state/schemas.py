from pydantic import BaseModel, Field
from typing import List, TypedDict


class Character(BaseModel):
    name: str = Field(..., description="The name of the character")
    appearance: str = Field(..., description="The appearance of the character")
    personality: str = Field(..., description="The personality of the character")
    backstory: str = Field(..., description="The backstory of the character")
    goals: str = Field(..., description="The goals of the character")
    abilities: str = Field(..., description="The abilities of the character")
    inventory: str = Field(..., description="The inventory of the character")
    relationships: str = Field(..., description="The relationships of the character")

class GenerateCharacterState(TypedDict):
    query: str # Query to generate the character
    character: Character # Generated character
    human_character_feedback: str # Human feedback
