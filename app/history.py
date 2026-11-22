"""Story history numbered by user-message position."""

from langchain_core.messages import HumanMessage

from app.state.schemas import Story


def user_messages(messages: list) -> list[HumanMessage]:
    return [message for message in messages if isinstance(message, HumanMessage)]


def rewound_story(story: Story, cut_turn: int) -> Story:
    """Keep the story prefix strictly before the selected user message."""
    kept = [item for item in story.turn_summaries if item.turn < cut_turn]
    previous = max(kept, key=lambda item: item.turn, default=None)
    return story.model_copy(
        update={
            "events": [event for event in story.events if event.turn < cut_turn],
            "turn_summaries": kept,
            "summary": previous.summary if previous else "None",
            "title": previous.title if previous else "Untitled",
        }
    )
