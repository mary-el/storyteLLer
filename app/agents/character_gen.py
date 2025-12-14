from app.state.schemas import GenerateCharacterState, Character
from app.agents.object_gen import ObjectGenerator
from langchain_core.messages import SystemMessage
import dotenv
from trustcall import create_extractor

dotenv.load_dotenv()

class CharacterGenerator(ObjectGenerator):
    def __init__(self):
        super().__init__()

        self.generation_instructions = """
        You are a character generator helper. Follow these instructions:
        1. Review the conversation;
        2. If human is satisfied with the character, return only the word "done".
        3. Otherwise, (re)write a character description based on the human feedback or ask for more feedback.
        Be concise, use the same language as the user.
        """

        self.extraction_instructions = """Extract the character from the following conversation.
        Always use the same language as the user!
        Don't add any information that is not in the conversation.
        """

        # Create the extractor
        self.trustcall_extractor = create_extractor(
            self.llm,
            tools=[Character],
            tool_choice="required", # Enforces use of the Character tool
        )


    def extract(self, state: GenerateCharacterState):
        """
        Create the character from the conversation.
        """
        messages = state.get('messages', [])
        # existing_character_obj = state.get('character')
        # if isinstance(existing_character_obj, dict):
        #     existing_character = existing_character_obj
        # else:
        #     existing_character = existing_character_obj.model_dump()
        # # Handle both dict (from state) and Character object cases
        character = self.trustcall_extractor.invoke({
            "messages": [SystemMessage(content=self.extraction_instructions)] + messages,
            # "existing": {"Character": existing_character} if existing_character else {}
        })
        if len(character["responses"]) == 0:
            return {}
        # Extract character - handle both dict and Character object
        character_response = character["responses"][0]
        # Write the character to state
        return {"character": character_response}
