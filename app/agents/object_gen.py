import time
from abc import ABC, abstractmethod
from typing import Optional

import dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from trustcall import create_extractor

from app.config.schema import AppConfig, ObjectAgentConfig
from app.state.schemas import StorytellerState
from app.utils import logger, split_thinking, strip_thinking, trim_history

dotenv.load_dotenv()


def _last_assistant_text(messages: list) -> str | None:
    """Latest non-empty AIMessage string in this subgraph (reliable before interrupt)."""
    for msg in reversed(messages or []):
        if not isinstance(msg, AIMessage):
            continue
        c = msg.content
        if isinstance(c, str) and c.strip():
            return strip_thinking(c)
    return None


class ObjectGenerator(ABC):
    @classmethod
    @abstractmethod
    def _agent_prompts(cls, app_config: AppConfig) -> ObjectAgentConfig:
        """Agent-specific prompts from the shared app config."""

    def __init__(
        self,
        llm: ChatOpenAI,
        checkpointer: Optional[MemorySaver] = None,
        langdev: bool = False,
        *,
        app_config: AppConfig,
    ):
        self.app_config = app_config
        agent = type(self)._agent_prompts(app_config)
        og = app_config.object_generator
        self.llm = llm
        self.checkpointer = checkpointer
        self.langdev = langdev
        self.generation_instructions = agent.generation_instructions
        self.extraction_instructions = agent.extraction_instructions
        self.max_trim_tokens = og.max_trim_tokens
        # Create extractor using the object class
        self.trustcall_extractor = self._create_extractor()
        self.graph = self.build_graph()

    def _create_extractor(self):
        """Create the trustcall extractor using the object class"""
        return create_extractor(
            self.llm, tools=[self.entity_class], tool_choice="required", enable_inserts=False
        )

    @property
    @abstractmethod
    def object_class(self):
        """Return the object class (e.g., CharacterObject, WorldObject)"""
        raise NotImplementedError("Subclasses must implement this property")

    @property
    @abstractmethod
    def entity_class(self):
        """Return the entity class (e.g., Character, World)"""
        raise NotImplementedError("Subclasses must implement this property")

    @property
    @abstractmethod
    def object_field_name(self):
        """Return the field name in state (e.g., 'character', 'world')"""
        raise NotImplementedError("Subclasses must implement this property")

    async def initialize_object_node(self, state: StorytellerState):
        """Initialize the object with an ID if it doesn't exist."""
        obj = self.object_class()
        logger.debug(f"Object initialized: {obj}")
        return {"generated_object": obj}

    async def extract(self, state: StorytellerState):
        """
        Extract the object from the conversation.
        """
        logger.debug(f"Extracting object {state.get('user_id', 'default')} from description")
        messages = state.get("messages", [])
        # Trim messages to avoid token limits and potential serialization issues
        trimmed_messages = self.trim_messages(messages)
        existing_object = state.get("generated_object")
        # Access attribute directly since existing_object is a Pydantic model, not a dict
        field_value = getattr(existing_object, self.object_field_name)
        existing_object_obj = field_value.model_dump()
        # Get the tool name (class name)
        tool_name = self.entity_class.__name__
        try:
            # Invoke extractor
            result = await self.trustcall_extractor.ainvoke(
                {
                    "messages": trimmed_messages
                    + [SystemMessage(content=self.extraction_instructions)],
                    "existing": {tool_name: existing_object_obj},
                }
            )

            if len(result["responses"]) == 0:
                return {}

            # Extract object - handle both dict and object cases
            object_response = result["responses"][0]
            logger.debug(f"Object response: {object_response}")
            # Set attribute directly since existing_object is a Pydantic model, not a dict
            setattr(existing_object, self.object_field_name, object_response)
            logger.debug(f"Object extracted: {existing_object.object_id}, {existing_object}")
            # Write the object to state
            return {"generated_object": existing_object}
        except Exception as e:
            logger.error(f"Error extracting object: {e}")
            return {}

    async def human_feedback(self, state: StorytellerState):
        """Node that interrupts to request human feedback, returns feedback on resume"""
        logger.debug("Requesting human feedback via interrupt()")
        # interrupt() will pause execution and wait for Command(resume=...)
        # On resume, it returns the value passed to Command(resume=value)
        # Payload shown to the client while paused — must not be the user's last line (that
        # looked like the model "echoing" the first query). Resume still uses the next message.
        hint = (
            f"Continue the conversation to build this {self.object_field_name}: "
            "add detail, refine, or say when you are satisfied."
        )
        draft = _last_assistant_text(state.get("messages", []))
        payload: dict = {"hint": hint, "draft": draft}

        # Call interrupt() - this will pause on first call, return feedback value on resume
        feedback = interrupt(payload)
        logger.info(f"Received feedback from interrupt: {feedback}")

        # Add feedback as HumanMessage to continue the conversation
        return {"messages": [HumanMessage(content=feedback)]}

    def should_extract(self, state: StorytellerState):
        """Extract exactly once, right before finalizing; otherwise keep gathering feedback."""
        status = state.get("status", "in_progress")
        if status == "created":
            logger.debug("Object finished — extracting before finalizing")
            return "extract"

        logger.debug("Continuing to human feedback")
        return "human_feedback"

    async def generate_description(self, state: StorytellerState):
        """Generate a description of the object."""
        messages = self.trim_messages(state.get("messages", []))
        start_time = time.time()
        logger.debug(
            f"Generating description: {messages + [SystemMessage(content=self.generation_instructions)]}"
        )
        answer = await self.llm.ainvoke(
            messages + [SystemMessage(content=self.generation_instructions)]
        )
        thinking, visible = split_thinking(answer.content)
        logger.debug(f"Generated description: {visible}")
        if thinking:
            logger.debug(f"Model thinking ({len(thinking)} chars)")
        end_time = time.time()
        logger.debug(f"Time taken: {end_time - start_time} seconds")
        if visible.lower().strip() == "done":
            logger.debug("Object created")
            return {"status": "created"}
        return {
            "messages": [AIMessage(content=visible, response_metadata=answer.response_metadata)]
        }

    def trim_messages(self, messages: list[BaseMessage]):
        """Trim the messages to the token budget, stripping model thinking from AIMessages."""
        cleaned = [
            AIMessage(content=strip_thinking(m.content)) if isinstance(m, AIMessage) else m
            for m in messages
        ]
        return trim_history(cleaned, self.max_trim_tokens)

    def build_graph(self):
        """Build the graph"""
        logger.debug("Building graph for ObjectGenerator")
        builder = StateGraph(StorytellerState)
        builder.add_node("initialize_object", self.initialize_object_node)
        builder.add_node("human_feedback", self.human_feedback)
        builder.set_entry_point("initialize_object")
        builder.add_node("generate_description", self.generate_description)
        builder.add_node("extract", self.extract)

        # Run generation on the first user message immediately; interrupt only after a draft exists.
        builder.add_edge("initialize_object", "generate_description")
        builder.add_edge("human_feedback", "generate_description")
        # Keep gathering feedback until the object is finished, then extract exactly once.
        builder.add_conditional_edges(
            "generate_description", self.should_extract, ["extract", "human_feedback"]
        )
        builder.add_edge("extract", END)
        # Compile with checkpoint saver
        # NodeInterrupt in human_feedback will propagate to parent graph automatically
        self.graph = builder.compile(checkpointer=self.checkpointer)
        logger.debug("Graph built successfully")
        return self.graph
