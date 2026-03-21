import json
from typing import Literal, Optional

import dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.agents.character_gen import CharacterGenerator
from app.agents.world_gen import WorldGenerator
from app.state.schemas import StorytellerState
from app.utils import logger

dotenv.load_dotenv()


class DialogueResponse(BaseModel):
    """Structured response from dialogue node"""

    node: Literal["generate_character", "generate_world", "dialogue"] = Field(
        description="Next node to route to"
    )
    response: str = Field(description="Response message if node is 'dialogue'")


class Storyteller:
    def __init__(self, langdev: bool = False, memory_store: Optional[InMemoryStore] = None) -> None:
        self.llm = ChatOpenAI(
            model="rag-runtime", base_url="http://127.0.0.1:19000/v1", temperature=0.3
        )
        self.checkpointer = MemorySaver() if not langdev else None
        self.memory_store = memory_store
        self.langdev = langdev
        # Pass shared checkpointer to subgraphs for proper Command handling
        self.character_generator = CharacterGenerator(
            self.llm, self.checkpointer, memory_store, langdev=langdev
        )
        self.world_generator = WorldGenerator(
            self.llm, self.checkpointer, memory_store, langdev=langdev
        )
        # Instance variable to pass generator_prompt between nodes
        self.prompt = """
        You are the Storyteller - a helpful assistant that helps the user create a story.
        You can generate the world and the characters for the story.
        Respond with JSON containing:
        - "node": "generate_character", "generate_world", or "dialogue"
        - "response": if node is dialogue, provide your response message

        If the most recent conversation message is a system event
        "SYSTEM_EVENT: OBJECT_CREATED", acknowledge that the object was created
        and ask the user for next instructions.

        As soon as user mentions creation of a character or world, set node to "generate_character" or "generate_world" and provide a prompt.
        Otherwise, set node to "dialogue" and provide your response message.
        """
        self.template = ChatPromptTemplate(
            [
                MessagesPlaceholder(variable_name="conversation", optional=True),
                ("system", self.prompt),
            ]
        )
        self.max_len = 1000
        self.graph = self.build_graph()
        self.waiting_for_feedback = False

    def route_generator(
        self, state: StorytellerState
    ) -> Literal["generate_character", "generate_world", "dialogue"]:
        """Route to appropriate generator based on structured response."""
        messages = state.get("messages", [])
        if not messages:
            return "dialogue"

        last_message = messages[-1]
        content = getattr(last_message, "content", "")

        try:
            data = json.loads(content)
            node = data.get("node", "dialogue")
            return node
        except (json.JSONDecodeError, AttributeError):
            # Fallback to dialogue if JSON parsing fails
            logger.warning(f"Failed to parse JSON: {content}")
            return "dialogue"

    async def dialogue(self, state: StorytellerState) -> StorytellerState:
        """Dialogue node for the storyteller"""
        messages = trim_messages(
            state["messages"],
            max_tokens=self.max_len,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=True,
        )
        logger.debug("routing to dialogue")
        prompt = self.template.invoke({"conversation": messages})
        llm_with_structure = self.llm.with_structured_output(DialogueResponse)
        response = await llm_with_structure.ainvoke(prompt)
        # Return JSON string as message content with ensure_ascii=False to preserve Unicode
        json_content = json.dumps(
            {"node": response.node, "response": response.response}, ensure_ascii=False
        )
        return {"messages": [AIMessage(content=json_content)], "status": None}

    def finalize_object(self, state: StorytellerState) -> StorytellerState:
        """Finalize object generation and extract object_id"""
        logger.debug("Finalizing object generation")
        generated_object = state.get("generated_object")
        if generated_object:
            return {
                "messages": [
                    SystemMessage(
                        content=f"SYSTEM_EVENT: OBJECT_CREATED: {generated_object.model_dump_json()}"
                    )
                ],
                "generated_object": None,
            }
        return {}

    def build_graph(self):
        """Build the graph for the storyteller with integrated subgraphs"""
        graph = StateGraph(StorytellerState)

        # Add dialogue node
        graph.add_node("dialogue", self.dialogue)

        # Add subgraphs directly as nodes - LangGraph will handle Command propagation
        graph.add_node("generate_character", self.character_generator.graph)
        graph.add_node("generate_world", self.world_generator.graph)

        # Add finalize nodes
        graph.add_node("finalize_object", self.finalize_object)

        # Start with dialogue
        graph.add_edge(START, "dialogue")

        # Route from dialogue to prepare nodes or end
        graph.add_conditional_edges(
            "dialogue",
            self.route_generator,
            {
                "generate_character": "generate_character",
                "generate_world": "generate_world",
                "dialogue": END,
            },
        )
        # Connect subgraphs to finalize nodes
        graph.add_edge("generate_character", "finalize_object")
        graph.add_edge("generate_world", "finalize_object")

        # Return to dialogue after object is finalized
        graph.add_edge("finalize_object", "dialogue")

        return graph.compile(checkpointer=self.checkpointer)

    async def arun(self, query: str | Command, user_id: str, thread_id: str = None) -> dict:
        """Run the storyteller asynchronously"""
        if isinstance(query, Command):
            input_data = query
        else:
            input_data = {"messages": [HumanMessage(content=query)], "user_id": user_id}
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id or user_id}}
        result = await self.graph.ainvoke(input_data, config)
        return result

    async def tell(self, query: str, user_id: str, thread_id: str = None) -> str:
        if not self.waiting_for_feedback:
            result = await self.arun(query, user_id, thread_id)
        else:
            logger.info(f"Sending command: {query}")
            result = await self.arun(Command(resume=query), user_id, thread_id)
        interrupts = result.get("__interrupt__")
        self.waiting_for_feedback = interrupts is not None
        if interrupts:
            if isinstance(interrupts, list) and interrupts:
                first = interrupts[0]
                return first.value if hasattr(first, "value") else str(first)
            return str(interrupts)
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None
        return getattr(last_message, "content", "") if last_message else ""
