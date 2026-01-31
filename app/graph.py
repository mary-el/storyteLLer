import json
from typing import Optional

import dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
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
        {", ".join([generator.entity_class.__name__ for generator in self.generators])}
        Use them as soon as the user mentions the object with the users's prompt.
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
        # Track pending tool runs per user and map tool name -> generator.
        self.pending_tool_by_user: dict[str, dict] = {}
        self.tool_map = {generator.tool.name: generator for generator in self.generators}

    def _extract_pending_payload(self, messages: list[BaseMessage]) -> Optional[dict]:
        """Parse tool JSON payloads and return the latest pending state."""
        payload = None
        for message in messages:
            content = getattr(message, "content", None)
            if not isinstance(content, str):
                continue
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                continue
            if data.get("status") == "in_progress" and data.get("tool_name"):
                payload = data
        return payload

    async def _resume_tool(self, user_id: str, feedback: str) -> dict:
        """Resume a pending tool run using the human feedback."""
        pending = self.pending_tool_by_user.get(user_id)
        if not pending:
            return {}
        tool_name = pending.get("tool_name")
        generator = self.tool_map.get(tool_name)
        if not generator:
            logger.warning(f"Pending tool '{tool_name}' not found. Dropping pending state.")
            self.pending_tool_by_user.pop(user_id, None)
            return {}

        thread_id = pending.get("thread_id") or user_id
        # Use configurable IDs so checkpoints align with the tool run.
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

        result_stream = await generator.send_feedback(feedback, config)
        last_message = None
        status = "in_progress"

        async for event in result_stream:
            if event.get("status"):
                status = event["status"]
            if len(event.get("messages", [])) > 0:
                last_message = event["messages"][-1]

        if status == "created":
            self.pending_tool_by_user.pop(user_id, None)

        if last_message:
            return {"messages": [last_message], "status": status}
        return {"messages": [], "status": status}

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
        # If a tool is waiting, resume it directly with the new feedback.
        pending = self.pending_tool_by_user.get(user_id)
        if pending:
            logger.debug(f"Resuming tool {pending.get('tool_name')} for user {user_id}")
            resumed = await self._resume_tool(user_id, query)
            if resumed:
                return resumed

        state = MessagesState(messages=[HumanMessage(content=query)])
        logger.info(state)
        # Always pass configurable IDs to the storyteller graph.
        config = {"configurable": {"user_id": user_id, "thread_id": user_id}}
        result = await self.graph.ainvoke(state, config)

        pending_payload = self._extract_pending_payload(result.get("messages", []))
        if pending_payload:
            self.pending_tool_by_user[user_id] = pending_payload
            message = pending_payload.get("message", "")
            logger.debug(f"Pending tool {pending_payload.get('tool_name')} for user {user_id}")
            return {"messages": [AIMessage(content=message)], "status": "in_progress"}

        return result
