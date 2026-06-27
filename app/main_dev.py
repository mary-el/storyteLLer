import dotenv
from langchain_openai import ChatOpenAI

from app.config import load_app_config
from app.graph import Storyteller

dotenv.load_dotenv()

# Namespace is now constructed dynamically from user_id in config
# When invoking the graph via API, include user_id in config:
# config = {"configurable": {"thread_id": "user_123", "user_id": "user_123"}}
_config = load_app_config()
_llm = ChatOpenAI(
    model=_config.llm.model,
    base_url=_config.llm.base_url,
    temperature=_config.llm.temperature,
)
storyteller = Storyteller(langdev=True, config=_config)
storyteller_graph = storyteller.build_graph()
