from pydantic import BaseModel, Field


class RouterConfig(BaseModel):
    """Setup phase: characters until the user begins the story."""

    system_prompt: str = Field(description="Router system prompt (structured routing JSON)")
    max_trim_tokens: int = Field(
        default=1000, ge=1, description="Max tokens passed to trim_messages (uses LLM tokenizer)"
    )


class StoryNarratorConfig(BaseModel):
    """Story phase: narrator + optional memory_tool routing."""

    system_prompt: str = Field(description="Narrator system prompt (structured routing JSON)")
    max_trim_tokens: int = Field(
        default=1000, ge=1, description="Max tokens passed to trim_messages (uses LLM tokenizer)"
    )


class StoryUpdateConfig(BaseModel):
    archive_prompt: str = Field(
        description="System message for archive node; uses {previous_summary}, {previous_events}, {transcript}, {title}"
    )
    max_trim_messages: int = Field(default=24, ge=1)
    world_patch_max_messages: int = Field(default=16, ge=1)
    event_history_length: int = Field(default=10, ge=1)


class ObjectGeneratorConfig(BaseModel):
    extract_every_n_messages: int = Field(default=3, ge=1)
    max_messages: int = Field(default=5, ge=1)


class ObjectAgentConfig(BaseModel):
    """Shared prompts for character/world (and similar) object generators."""

    generation_instructions: str
    extraction_instructions: str


class AgentsConfig(BaseModel):
    character: ObjectAgentConfig
    world: ObjectAgentConfig


class MemoryAgentConfig(BaseModel):
    system_prompt: str = Field(
        description="Prompt for classifying memory list/get intent and kind from the conversation"
    )


class AppConfig(BaseModel):
    greeting: str = Field(
        default="Welcome to StoryteLLer! Let's build your world. Describe the setting you want to play in.",
        description="Fixed opening message shown to the user on a fresh session (no LLM call).",
    )
    llm: dict = Field(description="OpenAI-compatible model config", default_factory=dict)
    router: RouterConfig
    story_narrator: StoryNarratorConfig
    story_update: StoryUpdateConfig
    object_generator: ObjectGeneratorConfig = Field(default_factory=ObjectGeneratorConfig)
    agents: AgentsConfig
    memory_agent: MemoryAgentConfig
    saves_dir: str = "saves"
