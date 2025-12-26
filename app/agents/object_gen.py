from abc import abstractmethod

import dotenv
from langchain_core.messages import SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.store.memory import InMemoryStore
from typing import Optional
from app.state.schemas import State
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt, Command
from trustcall import create_extractor

dotenv.load_dotenv()

EXTRACT_EVERY_N_MESSAGES = 3
MAX_MESSAGES = 5
class ObjectGenerator:
    def __init__(self, langdev: bool = False, memory_store: Optional[InMemoryStore] = None, enable_inserts: bool = False):
        self.llm = ChatOpenAI(model="rag-runtime", base_url="http://127.0.0.1:3000/v1", temperature=0.3)

        self.generation_instructions = "Generate a description of the object."
        self.extraction_instructions = "Extract the object from the following conversation."
        self.max_messages = MAX_MESSAGES
        self.extract_every_n_messages = EXTRACT_EVERY_N_MESSAGES
        self.current_message_count = 0
        self.memory_store = memory_store
        self.langdev = langdev
        self.checkpointer = MemorySaver() if not langdev else None
        self.enable_inserts = enable_inserts
        # Create extractor using the object class
        self.trustcall_extractor = self._create_extractor()
        self.graph = self.build_graph()
    
    def _create_extractor(self):
        """Create the trustcall extractor using the object class"""
        return create_extractor(
            self.llm,
            tools=[self.object_class],
            tool_choice="required",
            enable_inserts=self.enable_inserts
        )

    @property
    @abstractmethod
    def object_class(self):
        """ Return the object class (e.g., Character, World) """
        raise NotImplementedError("Subclasses must implement this property")
    
    @property
    @abstractmethod
    def object_field_name(self):
        """ Return the field name in state (e.g., 'character', 'world') """
        raise NotImplementedError("Subclasses must implement this property")
    
    def initialize_object_node(self, state: State):
        """ Initialize the object with an ID if it doesn't exist. """
        obj = self.object_class()
        return {"generated_object": obj}

    def extract(self, state: State):
        """
        Extract the object from the conversation.   
        """
        messages = state.get('messages', [])
        # Trim messages to avoid token limits and potential serialization issues
        trimmed_messages = self.trim_messages(messages)
        existing_object_obj = state.get("generated_object").model_dump()
        object_id = existing_object_obj.get("object_id")
        existing_object_obj.pop("object_id")
        # Get the tool name (class name)
        tool_name = self.object_class.__name__
        
        # Invoke extractor
        result = self.trustcall_extractor.invoke({
            "messages": trimmed_messages + [SystemMessage(content=self.extraction_instructions)],
            "existing": {tool_name: existing_object_obj}
        })
        
        if len(result["responses"]) == 0:
            return {}
        
        # Extract object - handle both dict and object cases
        object_response = result["responses"][0]
        object_response.object_id = object_id
        
        # Get the object ID - handle both dict and object cases
        
        # Save to memory store using namespace and object ID
        namespace = (state.get('user_id'), "memories")
        if self.memory_store and namespace and object_id:
            self.memory_store.put(namespace, object_id, object_response)
        # Write the object to state
        return {"generated_object": object_response}

    def human_feedback(self, state: State):
        """ No-op node that should be interrupted on """
        feedback = interrupt("Please provide feedback on the object.")
        return {"messages": [HumanMessage(content=feedback)]}
    
    def should_extract(self, state: State):
        """ Return if the extract node should be executed """
        # Check if object is created or max messages reached
        status = state.get('status', 'in_progress')
        if status == "created" or self.current_message_count >= self.extract_every_n_messages:
            self.current_message_count = 0
            return "extract"
        
        # Otherwise continue to human feedback
        return "human_feedback"
    
    def should_continue(self, state: State):
        """ Return the next node to execute """
        # Check if object is created
        status = state.get('status', 'in_progress')
        if status == "created":
            return END
        # Otherwise continue to human feedback
        return "human_feedback"
    
    def generate_description(self, state: State):
        """ Generate a description of the object. """
        # Generate character description
        messages = self.trim_messages(state.get('messages', []))
        self.current_message_count += 1
        answer = self.llm.invoke(messages + [SystemMessage(content=self.generation_instructions)])

        if answer.content.lower().strip() == "done":
            return {"status": "created"}

        return {"messages": [answer]}

    def trim_messages(self, messages: list[BaseMessage]):
        """ Trim the messages to the max messages. """
        return trim_messages(messages, 
                             max_tokens=self.max_messages,
                             token_counter=len,
                             strategy="last",
                             start_on="human",
                             include_system=False)

    def build_graph(self):
        """ Build the graph """
        
        builder = StateGraph(State)
        builder.add_node("initialize_object", self.initialize_object_node)
        builder.add_node("human_feedback", self.human_feedback) 
        builder.set_entry_point("initialize_object")
        builder.add_node("generate_description", self.generate_description)
        builder.add_node("extract", self.extract)       

        builder.add_edge("initialize_object", "human_feedback")
        builder.add_edge("human_feedback", "generate_description")
        # Route to extract and human_feedback in parallel when extraction is needed
        builder.add_conditional_edges("generate_description", self.should_extract, ["extract", "human_feedback"])
        builder.add_conditional_edges("extract", self.should_continue, ["human_feedback", END])
        # Compile with checkpoint saver to enable run tracking
        interrupt_before = ['human_feedback'] if self.langdev else []
        self.graph = builder.compile(checkpointer=self.checkpointer, interrupt_before=interrupt_before)

        return self.graph

    def run(self, input: str, config: dict):
        """ Run the graph """
        # Extract user_id from config if provided, otherwise use thread_id as user_id
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id") or configurable.get("thread_id")
        initial_state = {"messages": [HumanMessage(content=input)]}
        initial_state["user_id"] = user_id
        return self.graph.stream(initial_state, config, stream_mode="values")

    def send_feedback(self, input: str, config: dict):
        return self.graph.stream(Command(resume=(input)), config, stream_mode="values")
    
    def initialize_object(self, config: dict) -> str:
        """
        Initialize the object and return its ID.
        Invokes the graph to run initialize_object_node which creates the object with an ID.
        """
        # Extract user_id from config if provided, otherwise use thread_id as user_id
        configurable = config.get("configurable", {})
        user_id = configurable.get("user_id") or configurable.get("thread_id")
        thread_id = configurable.get("thread_id")
        
        # Use stream to get state after initialization node runs
        # The graph will run: initialize_object -> human_feedback (interrupts)
        state_updates = self.graph.invoke({"messages": [], "user_id": user_id}, config)
        # Get the object from the initial state (created by initialize_object_node)
        object_data = state_updates.get("generated_object")
        if object_data:
            object_id = object_data.get("object_id") if isinstance(object_data, dict) else object_data.object_id
            return object_id
        else:
            raise ValueError(f"{self.object_class.__name__} not found in state after initialization")
