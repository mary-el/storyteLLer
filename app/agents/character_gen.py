import dotenv

from app.agents.object_gen import ObjectGenerator
from app.config.schema import AppConfig, ObjectAgentConfig
from app.state.schemas import Character, CharacterObject
from app.utils import logger

dotenv.load_dotenv()


class CharacterGenerator(ObjectGenerator):
    @classmethod
    def _agent_prompts(cls, app_config: AppConfig) -> ObjectAgentConfig:
        return app_config.agents.character

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("CharacterGenerator initialized successfully")

    @property
    def object_field_name(self):
        return "character"

    @property
    def entity_class(self):
        return Character

    @property
    def object_class(self):
        return CharacterObject
