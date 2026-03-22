import dotenv

from app.agents.object_gen import ObjectGenerator
from app.config.schema import AppConfig, ObjectAgentConfig
from app.state.schemas import World, WorldObject
from app.utils import logger

dotenv.load_dotenv()


class WorldGenerator(ObjectGenerator):
    @classmethod
    def _agent_prompts(cls, app_config: AppConfig) -> ObjectAgentConfig:
        return app_config.agents.world

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("WorldGenerator initialized successfully")

    @property
    def object_field_name(self):
        return "world"

    @property
    def entity_class(self):
        return World

    @property
    def object_class(self):
        return WorldObject
