from app.state.schemas import GenerateCharacterState, Character
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import dotenv

dotenv.load_dotenv()
llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", base_url="http://127.0.0.1:1234/v1")

character_instructions = """
You are a character generator. Follow these instructions to generate a character:
1. Review the generation query: {query}
2. Review the human feedback: {human_character_feedback}
3. Generate a character based on the query and the human feedback.
"""

def create_character(state: GenerateCharacterState):
    """ Create character """
    
    query=state['query']
    human_character_feedback=state.get('human_character_feedback', '')
        
    # Enforce structured output
    structured_llm = llm.with_structured_output(Character)

    # System message
    system_message = character_instructions.format(query=query, human_character_feedback=human_character_feedback)

    # Generate character 
    character = structured_llm.invoke([SystemMessage(content=system_message)])
    
    # Write the character to state
    return {"character": character}

def human_feedback(state: GenerateCharacterState):
    """ No-op node that should be interrupted on """
    pass

def should_continue(state: GenerateCharacterState):
    """ Return the next node to execute """

    # Check if human feedback
    human_character_feedback=state.get('human_character_feedback', None)
    if human_character_feedback:
        return "create_character"
    
    # Otherwise end
    return END

def build_graph():
    """ Build the graph """
    
    builder = StateGraph(GenerateCharacterState)
    builder.add_node("create_character", create_character)
    builder.add_node("human_feedback", human_feedback)
    builder.add_edge(START, "create_character")
    builder.add_edge("create_character", "human_feedback")
    builder.add_conditional_edges("human_feedback", should_continue, ["create_character", END])

    # Compile
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=['human_feedback'])

    return graph
