from typing import Optional

import dotenv
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.memory import InMemoryStore

from app.agents.character_gen import CharacterGenerator
from app.agents.world_gen import WorldGenerator
from app.utils import logger

dotenv.load_dotenv()


class Storyteller:
    def __init__(self, langdev: bool = False, memory_store: Optional[InMemoryStore] = None) -> None:
        self.llm = ChatOpenAI(
            model="rag-runtime", base_url="http://127.0.0.1:19000/v1", temperature=0.3
        )
        self.instruction = """You are the Storyteller - a helpful assistant that helps the user create a story.
        You can use tools to generate the world and the characters for the story.
        Invite the user to create a story by providing a prompt.
        """
        self.checkpointer = MemorySaver() if not langdev else None
        self.memory_store = memory_store
        self.langdev = langdev
        self.generators = [
            WorldGenerator(self.llm, self.checkpointer, memory_store, langdev=langdev),
            CharacterGenerator(self.llm, self.checkpointer, memory_store, langdev=langdev),
        ]
        self.tools = [generator.tool for generator in self.generators]
        self.prompt = f"""
        You are the Storyteller - a helpful assistant that helps the user create a story.
        You can use tools to generate the world and the characters for the story.
        Invite the user to create a story by providing a prompt.
        You can use tools to generate this objects:
        {", ".join([generator.object_class.__name__ for generator in self.generators])}
        """
        self.template = ChatPromptTemplate(
            [
                MessagesPlaceholder(variable_name="conversation", optional=True),
                ("system", self.prompt),
            ]
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.max_len = 1000
        self.graph = self.build_graph()

    async def dialogue(self, state: MessagesState) -> MessagesState:
        """Dialogue node for the storyteller"""
        messages = trim_messages(
            state["messages"],
            max_tokens=self.max_len,
            token_counter=len,
            strategy="last",
            start_on="human",
            include_system=False,
        )
        prompt = self.template.invoke({"conversation": messages})
        response = await self.llm_with_tools.ainvoke(prompt)
        return {"messages": [response]}

    def build_graph(self):
        """Build the graph for the storyteller"""
        graph = StateGraph(MessagesState)
        graph.add_node("dialogue", self.dialogue)
        tools_node = ToolNode(self.tools)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "dialogue")
        graph.add_conditional_edges("dialogue", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "dialogue")
        return graph.compile(checkpointer=self.checkpointer)

    async def arun(self, query: str, user_id: str) -> str:
        """Run the storyteller asynchronously"""
        state = MessagesState(messages=[HumanMessage(content=query)])
        logger.info(state)
        config = {"user_id": user_id, "thread_id": user_id}
        result = await self.graph.ainvoke(state, config)
        return result
