from app.state.schemas import GenerateCharacterState, Character, CharacterObject
from app.agents.object_gen import ObjectGenerator
import dotenv

dotenv.load_dotenv()

class CharacterGenerator(ObjectGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.generation_instructions = """
        You are a character generator helper. Follow these instructions:
        1. Review the conversation;
        2. If human is satisfied with the character and tells it explicitly, return only the word "done".
        3. Otherwise, (re)write a character description based on the human feedback or ask for more feedback.
        Be concise, use the same language as the user.
        """

        self.extraction_instructions = """Extract the character from the following conversation.
        Always use the same language as the user!
        Don't add any information that is not in the conversation.
        Pay attention to both human and bot messages.
        """

    @property
    def object_field_name(self):
        return "character"
        
    @property
    def entity_class(self):
        return Character
    

    @property
    def object_class(self):
        return CharacterObject
