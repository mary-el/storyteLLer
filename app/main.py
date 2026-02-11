import argparse
import asyncio
import json

import dotenv
from langgraph.store.memory import InMemoryStore

from app.graph import Storyteller


def _print_message(message: str) -> None:
    print("\n--- Model Response ---")
    print(message)
    print("--- End Response ---\n")


async def process_single_message(storyteller: Storyteller, message: str, user_id: str) -> bool:
    """
    Process a single message through the storyteller and display the response.

    Args:
        storyteller: The storyteller instance
        message: The message to process
    """
    try:
        response_text = await storyteller.tell(message, user_id)

        try:
            response_json = json.loads(response_text)
            message = response_json.get("response", "")
        except json.JSONDecodeError:
            message = response_text
        if message:
            _print_message(message)
        else:
            print("No response received for message.")
        return True
    except Exception as e:
        print(f"Error processing message '{message}': {e}")
        raise


async def interactive_conversation(storyteller: Storyteller, user_id: str) -> None:
    """
    Run an interactive conversation loop with the storyteller.

    Args:
        storyteller: The storyteller instance
    """
    print("\n=== Interactive Storyteller ===")
    print("Enter your messages to build the story.")
    print("Type 'exit', 'quit', or 'done' to finish.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "done", "q"]:
                print("\nFinishing conversation...")
                break

            # Process the message through the storyteller
            should_continue = await process_single_message(storyteller, user_input, user_id)
            if not should_continue:
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


async def main(user_id: str = "1") -> None:
    """
    Main orchestration function for interactive storytelling.

    Args:
        user_id: The user ID for the session
    """
    in_memory_store = InMemoryStore()

    # Setup
    storyteller = Storyteller(memory_store=in_memory_store)

    # Run interactive conversation
    await interactive_conversation(storyteller, user_id)


def cli():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive character generator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m app.main
  python -m app.main --user-id user123
        """,
    )
    parser.add_argument(
        "--user-id", type=str, default="1", help="User ID for the session (default: '1')"
    )

    args = parser.parse_args()
    asyncio.run(main(user_id=args.user_id))


if __name__ == "__main__":
    dotenv.load_dotenv()
    cli()
