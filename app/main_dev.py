from app.agents.character_gen import CharacterGenerator
from app.agents.world_gen import WorldGenerator

import dotenv
from langgraph.store.memory import InMemoryStore
dotenv.load_dotenv()

in_memory_store = InMemoryStore()
user_id = "1"
namespace = (user_id, "memories")

character_generator = CharacterGenerator(langdev=False, memory_store=in_memory_store, namespace=namespace)
character_graph = character_generator.build_graph()
world_generator = WorldGenerator(langdev=False, memory_store=in_memory_store, namespace=namespace)
world_graph = world_generator.build_graph()
