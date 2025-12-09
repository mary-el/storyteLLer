from app.state.schemas import GenerateCharacterState, Character
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import START, END, StateGraph
import dotenv
from trustcall import create_extractor

dotenv.load_dotenv()
# llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", base_url="http://127.0.0.1:1234/v1")
llm = ChatOpenAI(model="rag-runtime", base_url="http://127.0.0.1:3000/v1", temperature=0.3)

generation_instructions = """
You are a character generator helper. Follow these instructions:
1. Review the conversation;
2. If human is satisfied with the character, return only the word "done".
3. Otherwise, (re)write a character description based on the human feedback or ask for more feedback.
Be concise, use the same language as the user.
"""

character_instructions = """Extract the character from the following conversation.
Always use the same language as the user!
Don't add any information that is not in the conversation.
"""

# Create the extractor
trustcall_extractor = create_extractor(
    llm,
    tools=[Character],
    tool_choice="required", # Enforces use of the Character tool
)

def create_character_description(state: GenerateCharacterState):
    """ Create character description """
    # Generate character description
    messages = state.get('messages', [])
    answer = llm.invoke([SystemMessage(content=generation_instructions)] + messages)

    if answer.content.lower().strip() == "done":
        return {"status": "created"}

    return {"messages": [answer]}

def create_character(state: GenerateCharacterState):
    """
    Create the character from the conversation.
    """
    messages = state.get('messages', [])
    existing_character_obj = state.get('character')
    if isinstance(existing_character_obj, dict):
        existing_character = existing_character_obj
    else:
        existing_character = existing_character_obj.model_dump()
    # Handle both dict (from state) and Character object cases
    character = trustcall_extractor.invoke({
        "messages": [SystemMessage(content=character_instructions)] + messages,
        # "existing": {"Character": existing_character} if existing_character else {}
    })
    if len(character["responses"]) == 0:
        return {}
    # Extract character - handle both dict and Character object
    character_response = character["responses"][0]
    # Write the character to state
    return {"character": character_response}

def human_feedback(state: GenerateCharacterState):
    """ No-op node that should be interrupted on """
    return {}

def should_continue(state: GenerateCharacterState):
    """ Return the next node to execute """
    # Check if character is created
    status = state.get('status', 'in_progress')
    if status == "created":
        return "create_character"
    
    # Otherwise continue to human feedback
    return "human_feedback"

def build_graph():
    """ Build the graph """
    
    builder = StateGraph(GenerateCharacterState)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("create_character_description", create_character_description)
    builder.add_node("create_character", create_character)
    builder.add_edge(START, "human_feedback")
    builder.add_edge("human_feedback", "create_character_description")
    builder.add_conditional_edges("create_character_description", should_continue, ["create_character", "human_feedback"])
    builder.add_edge("create_character", END)

    # Compile
    graph = builder.compile(interrupt_before=['human_feedback'])

    return graph
