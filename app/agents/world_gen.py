from app.state.schemas import GenerateWorldState, World, WorldObject
from app.agents.object_gen import ObjectGenerator
import dotenv

dotenv.load_dotenv()

class WorldGenerator(ObjectGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generation_instructions = """
        You are a world generator helper for the roleplaying game. Follow these instructions:
        1. Review the conversation;
        2. If human is satisfied with the world and tells it explicitly, return only the word "done".
        3. Otherwise, (re)write a world description based on the human feedback or ask for more feedback.
        Be concise, use the same language as the user.
        """

        self.extraction_instructions = """Extract the world object from the following conversation.
        Always use the same language as the user!
        Don't add any information that is not in the conversation.
        """

    @property
    def object_field_name(self):
        return "world"
    
    @property
    def entity_class(self):
        return World
    
    @property
    def object_class(self):
        return WorldObject
