import argparse
import asyncio
import json
import traceback
from datetime import datetime

import dotenv
from langgraph.store.memory import InMemoryStore

from app import persistence
from app.graph import Storyteller
from app.utils import split_thinking


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
        thinking, visible = split_thinking(response_text)

        try:
            response_json = json.loads(visible)
            display_text = response_json.get("response", "") or ""
        except json.JSONDecodeError:
            display_text = visible
        if display_text:
            _print_message(display_text)
        if thinking:
            print("--- Model Reasoning ---")
            print(thinking)
            print("--- End Reasoning ---\n")
        return True
    except Exception:
        traceback.print_exc()
        return True


async def _handle_load(storyteller: Storyteller, user_id: str) -> bool:
    """Prompt user to pick a save and load it. Returns True if a save was loaded."""
    saves_dir = storyteller.saves_dir
    saves = persistence.list_saves(saves_dir) if saves_dir else []
    if not saves:
        print("\n[No saved worlds found.]\n")
        return False
    print("\n=== Saved Worlds ===")
    for i, save in enumerate(saves, 1):
        saved_dt = save["saved_at"]
        try:
            saved_dt = datetime.fromisoformat(save["saved_at"]).strftime("%b %d %H:%M")
        except ValueError:
            pass
        print(f"  {i}. {save['title']}  [phase: {save['phase']}, turn {save['turn']}, {saved_dt}]")
    print("  0. Cancel")
    while True:
        try:
            choice = input("Pick a number: ").strip()
            idx = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if idx == 0:
            return False
        if 1 <= idx <= len(saves):
            selected = saves[idx - 1]
            break
        print(f"Enter a number between 0 and {len(saves)}.")

    print(f"\nLoading \"{selected['title']}\"…")
    save_data = persistence.load_story(selected["path"])
    await storyteller.load(save_data, user_id, user_id)
    print(f"[Loaded. Continuing from phase={selected['phase']}, turn={selected['turn']}]\n")
    return True


async def interactive_conversation(storyteller: Storyteller, user_id: str) -> None:
    """
    Run an interactive conversation loop with the storyteller.

    Args:
        storyteller: The storyteller instance
    """
    print("\n=== Interactive Storyteller ===")
    print("Type 'exit', 'quit', or 'done' to finish.")
    print("Type '/load' to resume a saved world.\n")
    greeting = await storyteller.init(user_id)
    if greeting:
        _print_message(greeting)

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "done", "q"]:
                print("\nFinishing conversation...")
                break

            # Handle /load command
            if user_input.lower() == "/load":
                await _handle_load(storyteller, user_id)
                continue

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
