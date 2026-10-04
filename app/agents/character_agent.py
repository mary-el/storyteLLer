from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from trustcall import create_extractor

from app.config.schema import AppConfig
from app.state.schemas import Character, CharacterObject, StorytellerState
from app.utils import logger, message_text, trim_history


class CharacterAgent:
    """Creates and patches characters on demand, as they're mentioned during the story.

    The narrator only ever describes what happened in plain language (see
    `_pending_character_commands`); it's trustcall's own multi-instance/insert support
    that decides *which* character (if any) a description refers to — it sees every
    existing character's real id and current fields side by side with each command,
    and picks a PatchDoc (update) or a fresh Character (insert) accordingly.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        *,
        app_config: AppConfig,
    ) -> None:
        self.instructions = app_config.agents.character.instructions
        self.max_trim_tokens = app_config.story_update.world_patch_max_tokens
        self.extractor = create_extractor(
            llm, tools=[Character], tool_choice="required", enable_inserts=True
        )

        self.insert_only_extractor = create_extractor(
            llm,
            tools=[Character],
            tool_choice="required",
            enable_inserts=True,
            enable_updates=False,
        )

    def _trimmed_context(self, state: StorytellerState) -> list:
        messages = trim_history(state.get("messages", []), self.max_trim_tokens)
        return [
            AIMessage(content=message_text(m)) if isinstance(m, AIMessage) else m for m in messages
        ]

    async def _extract_one(self, messages: list, existing: list) -> tuple:
        """Run the main extractor; if it silently yields nothing (e.g. the model's PatchDoc
        targeted an id not in `existing`, which trustcall drops without retrying or raising),
        fall back to an insert-only extractor so the command still produces a character."""
        result = await self.extractor.ainvoke({"messages": messages, "existing": existing})
        if result["responses"]:
            return result["responses"][0], result["response_metadata"][0].get("json_doc_id")

        logger.warning(
            "CharacterAgent: main extractor returned no response (likely an unresolved "
            "PatchDoc target); retrying insert-only"
        )
        fallback = await self.insert_only_extractor.ainvoke(
            {"messages": messages, "existing": existing}
        )
        if not fallback["responses"]:
            return None, None
        return fallback["responses"][0], None

    async def run(self, state: StorytellerState) -> StorytellerState:
        """Node entrypoint: resolve every character command pending this turn.

        One trustcall call per command (not one batched call for the whole turn) —
        `tool_choice="any"` only guarantees *some* tool call is made, not one per
        command, so batching them risks the model quietly fulfilling only some of
        several pending commands. Processing sequentially also lets a later command
        in the same turn see a character a prior command in that turn just created.
        """
        story = state.get("story")
        commands: list[str] = state.get("_pending_character_commands") or []
        if not story or not commands:
            return {}

        characters = list(story.characters)
        context_messages = self._trimmed_context(state)
        changed = False

        for command in commands:
            existing = [(c.object_id, "Character", c.character.model_dump()) for c in characters]
            messages = context_messages + [
                SystemMessage(content=self.instructions.format(instruction=command))
            ]
            try:
                character, json_doc_id = await self._extract_one(messages, existing)
            except Exception as e:
                logger.error(f"CharacterAgent failed for {command!r}: {e}")
                continue
            if character is None:
                logger.error(f"CharacterAgent: no character resolved for {command!r}")
                continue

            target = next((c for c in characters if c.object_id == json_doc_id), None)
            if json_doc_id and target is not None:
                obj = target.model_copy(update={"character": character})
                characters = [obj if c.object_id == json_doc_id else c for c in characters]
                action = "patched"
            else:
                obj = CharacterObject(character=character)
                characters.append(obj)
                action = "created"
            changed = True
            logger.debug(f"CharacterAgent: {action} {obj.object_id} ({character.name!r})")

        if not changed:
            return {}
        return {"story": story.model_copy(update={"characters": characters})}
