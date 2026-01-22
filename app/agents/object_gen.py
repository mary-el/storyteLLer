import asyncio
import threading
import time
from abc import abstractmethod
from typing import Optional

import dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, PrivateAttr
from trustcall import create_extractor

from app.state.schemas import State
from app.utils import logger

dotenv.load_dotenv()

EXTRACT_EVERY_N_MESSAGES = 3
MAX_MESSAGES = 5


class ObjectGenerator:
    def __init__(
        self,
        llm: ChatOpenAI,
        checkpointer: Optional[MemorySaver] = None,
        memory_store: Optional[InMemoryStore] = None,
        langdev: bool = False,
    ):
        self.llm = llm
        self.checkpointer = checkpointer
        self.langdev = langdev
        self.generation_instructions = "Generate a description of the object."
        self.extraction_instructions = "Extract the object from the following conversation."
        self.max_messages = MAX_MESSAGES
        self.extract_every_n_messages = EXTRACT_EVERY_N_MESSAGES
        self.current_message_count = 0
        self.memory_store = memory_store
        # Create extractor using the object class
        self.trustcall_extractor = self._create_extractor()
        self.graph = self.build_graph()
        self.tool = ObjectGeneratorTool(self)

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

    def initialize_object_node(self, state: State, store: BaseStore = None):
        """Initialize the object with an ID if it doesn't exist."""
        store = store or self.memory_store
        logger.debug(f"Initializing object with store: {store}")
        obj = self.object_class()
        namespace = (state.get("user_id", "default"), "memories")
        if store:
            store.put(namespace, obj.object_id, obj)
        return {"generated_object": obj}

    async def extract(self, state: State, store: BaseStore = None):
        """
        Extract the object from the conversation.
        """
        store = store or self.memory_store
        logger.debug(f"Extracting object with store: {store}")
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
            # Save to memory store using namespace and object ID
            namespace = (state.get("user_id", "default"), "memories")
            if store:
                store.put(namespace, existing_object.object_id, existing_object)
            logger.debug(
                f"Object saved to store: {namespace}, {existing_object.object_id}, {existing_object}"
            )
            # Write the object to state
            return {"generated_object": existing_object}
        except Exception as e:
            logger.error(f"Error extracting object: {e}")
            return {}

    async def human_feedback(self, state: State, store: BaseStore = None):
        """No-op node that should be interrupted on"""
        feedback = interrupt("Please provide feedback on the object.")
        return {"messages": [HumanMessage(content=feedback)]}

    def should_extract(self, state: State, store: BaseStore = None):
        """Return if the extract node should be executed"""
        # Check if object is created or max messages reached
        status = state.get("status", "in_progress")
        if status == "created" or self.current_message_count >= self.extract_every_n_messages:
            self.current_message_count = 0
            logger.debug("Extracting object")
            return "extract"

        # Otherwise continue to human feedback
        logger.debug("Continuing to human feedback")
        return "human_feedback"

    def should_continue(self, state: State, store: BaseStore = None):
        """Return the next node to execute"""
        # Check if object is created
        status = state.get("status", "in_progress")
        logger.debug(f"Should continue: {status}")
        if status == "created":
            return END
        # Otherwise continue to human feedback
        return "human_feedback"

    async def generate_description(self, state: State, store: BaseStore = None):
        """Generate a description of the object."""
        # Generate character description
        messages = self.trim_messages(state.get("messages", []))
        self.current_message_count += 1
        start_time = time.time()
        answer = await self.llm.ainvoke(
            messages + [SystemMessage(content=self.generation_instructions)]
        )
        logger.debug(f"Generated description: {answer}")
        end_time = time.time()
        logger.debug(f"Time taken: {end_time - start_time} seconds")
        if answer.content.lower().strip() == "done":
            logger.debug("Object created")
            return {"status": "created"}
        return {"messages": [answer]}

    def trim_messages(self, messages: list[BaseMessage]):
        """Trim the messages to the max messages."""
        return trim_messages(
            messages,
            max_tokens=self.max_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
        )

    def build_graph(self):
        """Build the graph"""
        logger.debug("Building graph for ObjectGenerator")
        builder = StateGraph(State)
        builder.add_node("initialize_object", self.initialize_object_node)
        builder.add_node("human_feedback", self.human_feedback)
        builder.set_entry_point("initialize_object")
        builder.add_node("generate_description", self.generate_description)
        builder.add_node("extract", self.extract)

        builder.add_edge("initialize_object", "human_feedback")
        builder.add_edge("human_feedback", "generate_description")
        # Route to extract and human_feedback in parallel when extraction is needed
        builder.add_conditional_edges(
            "generate_description", self.should_extract, ["extract", "human_feedback"]
        )
        builder.add_conditional_edges("extract", self.should_continue, ["human_feedback", END])
        # Compile with checkpoint saver to enable run tracking
        interrupt_before = ["human_feedback"] if self.langdev else []
        self.graph = builder.compile(
            checkpointer=self.checkpointer, interrupt_before=interrupt_before
        )
        logger.debug("Graph built successfully")
        return self.graph

    async def run(self, input: str, config: dict):
        """Run the graph"""
        logger.debug("Running graph for ObjectGenerator")
        # Extract user_id from config if provided, otherwise use default as user_id
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id", "default")
        initial_state = {"messages": [HumanMessage(content=input)]}
        initial_state["user_id"] = user_id
        return await self.graph.astream(initial_state, config, stream_mode="values")

    async def send_feedback(self, input: str, config: dict):
        return self.graph.astream(Command(resume=(input)), config, stream_mode="values")

    async def initialize_object(self, config: dict, store: BaseStore = None) -> str:
        """
        Initialize the object and return its ID.
        Invokes the graph to run initialize_object_node which creates the object with an ID.
        """
        logger.debug("Initializing new object")
        store = store or self.memory_store
        # Extract user_id from config if provided, otherwise use thread_id as user_id
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id", "default")

        # Use stream to get state after initialization node runs
        # The graph will run: initialize_object -> human_feedback (interrupts)
        state_updates = await self.graph.ainvoke({"messages": [], "user_id": user_id}, config)
        # Get the object from the initial state (created by initialize_object_node)
        object_data = state_updates.get("generated_object")
        logger.debug(f"Object data: {object_data}")
        if object_data:
            object_id = (
                object_data.get("object_id")
                if isinstance(object_data, dict)
                else object_data.object_id
            )
            logger.debug(f"Object ID: {object_id}")
            return object_id
        else:
            raise ValueError(
                f"{self.object_class.__name__} not found in state after initialization"
            )


class ObjectGeneratorTool(StructuredTool):
    """
    A StructuredTool subclass that wraps an ObjectGenerator's async feedback loop.
    This tool handles the full generation process including interruptions for user feedback.
    """

    _generator: "ObjectGenerator" = PrivateAttr()

    def __init__(self, generator: "ObjectGenerator"):
        """
        Initialize the tool with an ObjectGenerator instance.

        Args:
            generator: The ObjectGenerator instance to use for object generation
        """

        # Create input schema for the tool
        class ToolInput(BaseModel):
            description: Optional[str] = Field(
                default="",
                description=f"Initial description or prompt for generating the {generator.object_field_name} (optional).",
            )
            user_id: Optional[str] = Field(
                default=None,
                description="User ID for the session (optional, defaults to 'default')",
            )
            thread_id: Optional[str] = Field(
                default=None,
                description="Thread ID for checkpointing (optional, defaults to user_id)",
            )

        tool_name = f"generate_{generator.object_field_name}"
        tool_description = f"""
        Generate a {generator.object_field_name} based on a description.
        This tool will interactively refine the {generator.object_field_name}
        based on feedback until it's complete. When interrupted,
         the tool will pause and wait for feedback to be provided
         through the Storyteller.
        """

        super().__init__(
            name=tool_name,
            description=tool_description,
            args_schema=ToolInput,
        )

        self._generator = generator

    async def _run_async(
        self,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """
        Async function that runs the generator graph with initial description.
        When the graph interrupts (at human_feedback node), it returns early.
        External code should call generator.send_feedback() to continue.
        """
        # Set up config
        if user_id is None:
            user_id = "default"
        if thread_id is None:
            thread_id = user_id

        if description is None:
            description = ""

        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

        try:
            # Initialize the object
            object_id = await self._generator.initialize_object(config)
            logger.debug(
                f"Initialized {self._generator.object_class.__name__} with ID: {object_id}"
            )

            # Process the stream until interruption or completion
            status = "in_progress"

            # Check if generation is complete
            if status == "created":
                # Retrieve the final object if generation is complete
                if self._generator.memory_store:
                    namespace = (user_id, "memories")
                    final_object = self._generator.memory_store.get(namespace, object_id)
                    if final_object:
                        # Get the entity (character or world) from the object
                        entity = getattr(final_object, self._generator.object_field_name)
                        entity_name = (
                            getattr(entity, "name", "Unnamed")
                            if hasattr(entity, "name")
                            else "Unnamed"
                        )
                        summary = f"""{self._generator.entity_class.__name__} '{entity_name}'
                        generated successfully. Object ID: {object_id}"""
                        return summary
                return f"{self._generator.object_class.__name__} generated successfully. Object ID: {object_id}"
            else:
                # Graph interrupted - return status indicating feedback is needed
                # External code should call generator.send_feedback(feedback, config) to continue
                return f"""{self._generator.object_class.__name__} generation in progress.
                Object ID: {object_id}.
                The graph is waiting for feedback.
                Call generator.send_feedback(feedback, config) to continue."""

        except Exception as e:
            logger.error(f"Error generating {self._generator.object_class.__name__}: {e}")
            return f"Error generating {self._generator.object_class.__name__}: {str(e)}"

    def _run(
        self,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """
        Synchronous entry point for the tool that wraps the async execution.
        Uses asyncio.run() or a new thread with event loop to execute the async function.
        """
        try:
            # Check if there's already an event loop running
            try:
                # If we're in an async context, we need to handle it differently
                # For LangGraph tools, we can use create_task or run in executor
                result = None
                exception = None

                def run_in_thread():
                    nonlocal result, exception
                    try:
                        # Create a new event loop for this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(
                            self._run_async(description, user_id, thread_id)
                        )
                        new_loop.close()
                    except Exception as e:
                        exception = e

                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join()

                if exception:
                    raise exception
                return result
            except RuntimeError:
                # No running loop, we can use asyncio.run
                return asyncio.run(self._run_async(description, user_id, thread_id))
        except Exception as e:
            logger.error(f"Error in sync wrapper: {e}")
            return f"Error: {str(e)}"
