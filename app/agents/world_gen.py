from app.state.schemas import GenerateWorldState, World
from app.agents.object_gen import ObjectGenerator
from langchain_core.messages import SystemMessage

import dotenv
from trustcall import create_extractor

dotenv.load_dotenv()

class WorldGenerator(ObjectGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", base_url="http://127.0.0.1:1234/v1")

        self.generation_instructions = """
        You are a world generator helper for the roleplaying game. Follow these instructions:
        1. Review the conversation;
        2. If human is satisfied with the world, return only the word "done".
        3. Otherwise, (re)write a world description based on the human feedback or ask for more feedback.
        Be concise, use the same language as the user.
        """

        self.extraction_instructions = """Extract the world object from the following conversation.
        Always use the same language as the user!
        Don't add any information that is not in the conversation.
        """

        # Create the extractor
        self.trustcall_extractor = create_extractor(
            self.llm,
            tools=[World],
            tool_choice="required", # Enforces use of the World tool
            enable_inserts=True
        )

    def extract(self, state: GenerateWorldState):
        """
        Create the world from the conversation.
        """
        messages = state.get('messages', [])
        existing_world_obj = state.get('world', World())

        if isinstance(existing_world_obj, dict):
            existing_world = existing_world_obj
        else:
            existing_world = existing_world_obj.model_dump()
        print(existing_world)
        response = self.trustcall_extractor.invoke({
            "messages": [SystemMessage(content=self.extraction_instructions)] + self.trim_messages(messages),
            "existing": {"World": existing_world} if existing_world else {}
        })
        if len(response["responses"]) == 0:
            return {}
        # Extract world - handle both dict and World object
        world_response = response["responses"][0]
        # Write the world to memory
        if self.memory_store and self.namespace:
            self.memory_store.put(self.namespace, "world", world_response)

        # Write the world to state
        return {"world": world_response}
