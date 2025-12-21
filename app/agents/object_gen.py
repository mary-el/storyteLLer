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

dotenv.load_dotenv()

EXTRACT_EVERY_N_MESSAGES = 3
MAX_MESSAGES = 5
class ObjectGenerator:
    def __init__(self, langdev: bool = True, memory_store: Optional[InMemoryStore] = None, namespace: Optional[str] = None):
        self.llm = ChatOpenAI(model="rag-runtime", base_url="http://127.0.0.1:3000/v1", temperature=0.3)

        self.generation_instructions = "Generate a description of the object."
        self.extraction_instructions = "Extract the object from the following conversation."
        self.max_messages = MAX_MESSAGES
        self.extract_every_n_messages = EXTRACT_EVERY_N_MESSAGES
        self.current_message_count = 0
        self.memory_store = memory_store
        self.langdev = langdev
        self.checkpointer = MemorySaver() if langdev else None
        self.namespace = namespace
        self.graph = self.build_graph()

    @abstractmethod
    def extract(self, state: State):
        """ Extract the object from the conversation. """
        raise NotImplementedError("Subclasses must implement this method")

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
        messages = state.get('messages', [])
        self.current_message_count += 1
        answer = self.llm.invoke([SystemMessage(content=self.generation_instructions)] + messages)

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
        builder.add_node("human_feedback", self.human_feedback) 
        builder.set_entry_point("human_feedback")
        builder.add_node("generate_description", self.generate_description)
        builder.add_node("extract", self.extract)       

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
        return self.graph.stream({"messages": [HumanMessage(content=input)]}, config, stream_mode="values")

    def send_feedback(self, input: str, config: dict):
        return self.graph.stream(Command(resume=(input)), config, stream_mode="values")
