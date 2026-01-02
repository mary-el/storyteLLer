import asyncio
import argparse
import dotenv
from langgraph.store.memory import InMemoryStore
from app.agents.character_gen import CharacterGenerator
from app.state.schemas import CharacterObject


def setup_character_generator(user_id: str, memory_store: InMemoryStore) -> tuple[CharacterGenerator, dict]:
    """
    Set up the character generator and configuration.
    
    Args:
        user_id: The user ID for the session
        memory_store: The memory store instance
        
    Returns:
        Tuple of (character_generator, config)
    """
    config = {"configurable": {"thread_id": user_id, "user_id": user_id}}
    character_generator = CharacterGenerator(memory_store=memory_store)
    return character_generator, config


async def process_single_message(character_generator: CharacterGenerator, message: str, config: dict) -> str:
    """
    Process a single message through the character generator and display the response.
    
    Args:
        character_generator: The character generator instance
        message: The message to process
        config: Configuration dictionary
        
    Returns:
        The status after processing ("created" or "in_progress")
    """
    try:
        result = await character_generator.send_feedback(message, config)
        last_message = None
        status = "in_progress"
        async for event in result:
            # Check status from the event state
            if event.get("status"):
                status = event["status"]
            if len(event.get("messages", [])) > 0:
                last_message = event["messages"][-1]
        
        if last_message:
            print("\n--- Model Response ---")
            last_message.pretty_print()
            print("--- End Response ---\n")
        else:
            print("No response received for message.")
        
        return status
    except Exception as e:
        print(f"Error processing message '{message}': {e}")
        raise


async def interactive_conversation(character_generator: CharacterGenerator, config: dict) -> None:
    """
    Run an interactive conversation loop with the character generator.
    
    Args:
        character_generator: The character generator instance
        config: Configuration dictionary
    """
    print("\n=== Interactive Character Generation ===")
    print("Enter your messages to refine the character.")
    print("Type 'exit', 'quit', or 'done' to finish and view the final character.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'done', 'q']:
                print("\nFinishing conversation...")
                break
            
            # Process the message and check status
            status = await process_single_message(character_generator, user_input, config)
            
            # Check if character generation is complete
            if status == "created":
                print("\nCharacter generation is complete!")
                break
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Finishing conversation...")
            break
        except EOFError:
            print("\n\nEnd of input. Finishing conversation...")
            break
        except Exception as e:
            print(f"Error in conversation: {e}")
            print("Continuing...")


def retrieve_character(memory_store: InMemoryStore, user_id: str, char_id: str) -> CharacterObject:
    """
    Retrieve a character from the memory store.
    
    Args:
        memory_store: The memory store instance
        user_id: The user ID
        char_id: The character ID
        
    Returns:
        The CharacterObject
        
    Raises:
        ValueError: If character is not found
    """
    namespace = (user_id, "memories")
    character = memory_store.get(namespace, char_id)
    if character is None:
        raise ValueError(f"Character with ID {char_id} not found in memory store")
    return character


async def main(user_id: str = "1") -> None:
    """
    Main orchestration function for interactive character generation.
    
    Args:
        user_id: The user ID for the session
    """
    in_memory_store = InMemoryStore()
    
    try:
        # Setup
        character_generator, config = setup_character_generator(user_id, in_memory_store)
        
        # Initialize the character and get its ID
        try:
            char_id = await character_generator.initialize_object(config)
            print(f"Character initialized with ID: {char_id}\n")
        except Exception as e:
            print(f"Error initializing character: {e}")
            raise
        
        # Run interactive conversation
        await interactive_conversation(character_generator, config)
        
        # Retrieve and display final character
        character = retrieve_character(in_memory_store, user_id, char_id)
        print("\n" + "="*50)
        print("Final Character:")
        print("="*50)
        print(character)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


def cli():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive character generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main
  python -m app.main --user-id user123
        """
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="1",
        help="User ID for the session (default: '1')"
    )
    
    args = parser.parse_args()
    asyncio.run(main(user_id=args.user_id))


if __name__ == "__main__":
    dotenv.load_dotenv()
    cli()
