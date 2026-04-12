from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str = Field(description="OpenAI-compatible model name")
    base_url: str = Field(description="API base URL")
    temperature: float = 0.3


class RouterConfig(BaseModel):
    """Setup phase: characters until the user begins the story."""

    system_prompt: str = Field(description="Router system prompt (structured routing JSON)")
    max_trim_tokens: int = Field(
        default=1000, ge=1, description="Max tokens for trim_messages (token_counter=len)"
    )


class StoryNarratorConfig(BaseModel):
    """Story phase: narrator + optional memory_tool routing."""

    system_prompt: str = Field(description="Narrator system prompt (structured routing JSON)")
    max_trim_tokens: int = Field(
        default=1000, ge=1, description="Max tokens for trim_messages (token_counter=len)"
    )


class StoryUpdateConfig(BaseModel):
    summary_prompt: str = Field(
        description="System message for rolling summary; may use {previous_summary}"
    )
    max_trim_messages: int = Field(default=24, ge=1)
    world_patch_max_messages: int = Field(default=16, ge=1)


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
    llm: LLMConfig
    router: RouterConfig
    story_narrator: StoryNarratorConfig
    story_update: StoryUpdateConfig
    object_generator: ObjectGeneratorConfig = Field(default_factory=ObjectGeneratorConfig)
    agents: AgentsConfig
    memory_agent: MemoryAgentConfig
