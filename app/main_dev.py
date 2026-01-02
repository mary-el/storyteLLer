from app.agents.character_gen import CharacterGenerator
from app.agents.world_gen import WorldGenerator

import dotenv
from langgraph.store.memory import InMemoryStore
dotenv.load_dotenv()

# Namespace is now constructed dynamically from user_id in config
# When invoking the graph via API, include user_id in config:
# config = {"configurable": {"thread_id": "user_123", "user_id": "user_123"}}
character_generator = CharacterGenerator(langdev=True)
character_graph = character_generator.build_graph()
world_generator = WorldGenerator(langdev=True)
world_graph = world_generator.build_graph()
