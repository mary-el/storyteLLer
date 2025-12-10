from app.agents.character_gen import CharacterGenerator
from app.agents.world_gen import WorldGenerator

character_generator = CharacterGenerator()
character_graph = character_generator.build_graph()
world_generator = WorldGenerator()
world_graph = world_generator.build_graph()
