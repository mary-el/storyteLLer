import time
from abc import abstractmethod
from typing import Optional

import dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from trustcall import create_extractor

from app.state.schemas import State
from app.utils import logger

dotenv.load_dotenv()

EXTRACT_EVERY_N_MESSAGES = 3
MAX_MESSAGES = 5


class ObjectGenerator:
    def __init__(self, langdev: bool = False, memory_store: Optional[InMemoryStore] = None):
        self.llm = ChatOpenAI(
            model="rag-runtime", base_url="http://127.0.0.1:8000/v1", temperature=0.3
        )

        self.generation_instructions = "Generate a description of the object."
        self.extraction_instructions = "Extract the object from the following conversation."
        self.max_messages = MAX_MESSAGES
        self.extract_every_n_messages = EXTRACT_EVERY_N_MESSAGES
        self.current_message_count = 0
        self.memory_store = memory_store
        self.langdev = langdev
        self.checkpointer = MemorySaver() if not langdev else None
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

    def initialize_object_node(self, state: State, store: BaseStore = None):
        """Initialize the object with an ID if it doesn't exist."""
        store = store or self.memory_store
        logger.debug(f"Initializing object with store: {store}")
        obj = self.object_class()
        namespace = (state.get("user_id", "default"), "memories")
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
