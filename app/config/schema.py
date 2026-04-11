from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str = Field(description="OpenAI-compatible model name")
    base_url: str = Field(description="API base URL")
    temperature: float = 0.3


class DialogueConfig(BaseModel):
    system_prompt: str = Field(description="Storyteller routing / dialogue system prompt")
    max_trim_tokens: int = Field(
        default=1000, ge=1, description="Max tokens for trim_messages (token_counter=len)"
    )


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
    dialogue: DialogueConfig
    object_generator: ObjectGeneratorConfig = Field(default_factory=ObjectGeneratorConfig)
    agents: AgentsConfig
    memory_agent: MemoryAgentConfig
