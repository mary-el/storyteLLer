import argparse
import asyncio
import json
import traceback

import dotenv
from langgraph.store.memory import InMemoryStore

from app.graph import Storyteller

# Synthetic first user line so world gen runs immediately; avoids an extra turn before any draft.
_BOOTSTRAP_MESSAGE = "Begin world creation."


def _print_message(message: str) -> None:
    print("\n--- Model Response ---")
    print(message)
    print("--- End Response ---\n")


async def process_single_message(storyteller: Storyteller, user_message: str, user_id: str) -> bool:
    """
    Process a single message through the storyteller and display the response.

    Args:
        storyteller: The storyteller instance
        user_message: The user's input text
    """
    try:
        response_text = await storyteller.tell(user_message, user_id)

        try:
            response_json = json.loads(response_text)
            display_text = response_json.get("response", "") or ""
        except json.JSONDecodeError:
            display_text = response_text
        if display_text:
            _print_message(display_text)
        return True
    except Exception:
        traceback.print_exc()
        return True


async def interactive_conversation(storyteller: Storyteller, user_id: str) -> None:
    """
    Run an interactive conversation loop with the storyteller.

    Args:
        storyteller: The storyteller instance
    """
    print("\n=== Interactive Storyteller ===")
    print("Type 'exit', 'quit', or 'done' to finish.\n")
    _print_message(
        "I am a storyteller. Let's create a story together. "
        "We'll start by creating a world — the first assistant reply is triggered automatically."
    )
    await process_single_message(storyteller, _BOOTSTRAP_MESSAGE, user_id)

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
        except Exception:
            traceback.print_exc()


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
