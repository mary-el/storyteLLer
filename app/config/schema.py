from pydantic import BaseModel, Field


class StoryNarratorConfig(BaseModel):
    """Story phase: narrator prompt and history limits."""

    system_prompt: str = Field(description="Narrator system prompt (structured routing JSON)")
    max_trim_tokens: int = Field(
        default=1000, ge=1, description="Max tokens of conversation passed to the narrator"
    )
    max_previous_events: int = Field(
        default=15,
        ge=1,
        description="How many recent story events to include in the narrator prompt",
    )


class StoryUpdateConfig(BaseModel):
    archive_prompt: str = Field(
        description="System message for archive node; uses {previous_summary}, {previous_events}, {transcript}, {title}"
    )
    max_trim_tokens: int = Field(
        default=8192,
        ge=1,
        description="Max tokens of transcript passed to the archive prompt",
    )
    world_patch_max_tokens: int = Field(
        default=8192,
        ge=1,
        description="Max tokens of conversation passed to the character agent",
    )
    event_history_length: int = Field(default=10, ge=1)


class ObjectGeneratorConfig(BaseModel):
    max_trim_tokens: int = Field(
        default=8192,
        ge=1,
        description="Max tokens of conversation kept during world generation",
    )


class ObjectAgentConfig(BaseModel):
    """Shared prompts for object generators (currently just world)."""

    generation_instructions: str
    extraction_instructions: str


class CharacterAgentConfig(BaseModel):
    """Prompts for the on-demand CharacterAgent (no interactive wizard)."""

    instructions: str = Field(
        description=(
            "System instructions for creating/updating characters in one trustcall call; "
            "supports {instruction} (the narrator's plain-language character commands for this turn)"
        )
    )


class AgentsConfig(BaseModel):
    character: CharacterAgentConfig
    world: ObjectAgentConfig


class AppConfig(BaseModel):
    greeting: str = Field(
        default="Welcome to StoryteLLer! Let's build your world. Describe the setting you want to play in.",
        description="Fixed opening message shown to the user on a fresh session (no LLM call).",
    )
    llm: dict = Field(description="OpenAI-compatible model config", default_factory=dict)
    story_narrator: StoryNarratorConfig
    story_update: StoryUpdateConfig
    object_generator: ObjectGeneratorConfig = Field(default_factory=ObjectGeneratorConfig)
    agents: AgentsConfig
    saves_dir: str = "saves"
