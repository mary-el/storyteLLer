# StoryteLLer

![](media/storyteller_img.png)

Lightweight LangGraph-based storytelling assistant: build a **world**, add **characters**, then run the **story** with rolling memory, in-story character updates, and optional memory lookup.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` with your API key:

```env
OPENAI_API_KEY=your_key_here
```

Configuration lives in [`app/config/default.yaml`](app/config/default.yaml). Override the path with `APP_CONFIG_PATH`. Key sections: `llm`, `router`, `story_narrator`, `story_update`, `agents`, `memory_agent`, `saves_dir`.
## Run

**CLI**

```bash
python -m app.main
python -m app.main --user-id user123
```

Type `/load` to pick a saved world and resume. Type `exit`, `quit`, or `done` to finish.

**Streamlit UI**

```bash
streamlit run app/ui.py
```

The sidebar shows phase, world, characters, rolling summary, and events. Actions:

- **New Story** — start a fresh session on a new thread (old saves stay on disk).
- **Load a world** — resume a saved story; chat history is restored from the save file.

World creation starts automatically on first load (same bootstrap as the CLI).

## Saves

Stories are **auto-saved** after each turn once a world exists. Files go to `saves/<story_id>.json` (configurable via `saves_dir` in config). Each file holds the full graph state: story aggregate, messages, phase, turn, and thread id.

On load, checkpoint state and the in-memory store are rebuilt from the saved `Story` so memory lookup keeps working.

## Pipeline overview

1. **World (once)** — Until `story.world` is set, each new user turn starts at `generate_world`.
2. **Setup router** — With a world in place, `router` handles chat and may route to `generate_character` or `begin_story` when the user wants to start the narrative.
3. **Story** — After `enter_story`, `phase` is `story`. Each turn goes to the `story` narrator, which may route to:
   - **`dialogue`** — normal narrative; then `story_update` runs before the turn ends.
   - **`update_characters`** — patch a character whose state changed (inventory, relationships, etc.); then `story_update`.
   - **`memory_tool`** — list or fetch world/character/event data; ends the turn without `story_update`.

`finalize_object` merges finished `WorldObject` / `CharacterObject` into [`Story`](app/state/schemas.py) (`story` + `phase` on `StorytellerState`).

## Memory

Two layers:

| Layer | Where | What |
| --- | --- | --- |
| **Story aggregate** | `Story` in graph state | `world`, `characters`, `summary`, `events` |
| **Store** | `InMemoryStore` | `(user_id, "memories")` — world/character objects; `(user_id, "events")` — per-turn event records |

After each narrative turn, `story_update` refreshes `Story.summary`, appends a `StoryEvent`, and writes the event to the store. The narrator can call `memory_tool` proactively to recall past events or character details not in context.

During story play, `update_characters` uses the same trustcall extraction pattern as setup to patch character fields in place.

## Top-level graph (`StorytellerState`)

```mermaid
flowchart TD
  START --> nextStart[next_after_START]
  nextStart -->|no_world| generate_world
  nextStart -->|setup| router
  nextStart -->|phase_story| story
  generate_world --> finalize_object
  generate_character --> finalize_object
  finalize_object --> router
  router -->|generate_character| generate_character
  router -->|begin_story| enter_story
  router -->|dialogue| END
  enter_story --> story
  story -->|memory_tool| memory_tool
  story -->|dialogue| story_update
  story -->|update_characters| update_characters
  update_characters --> story_update
  story_update --> END
  memory_tool --> END
```

## Character / world subgraph (`ObjectGenerator`)

```mermaid
flowchart TD
  initialize_object --> generate_description
  human_feedback --> generate_description
  generate_description -->|should_extract| extract
  generate_description -->|human_feedback| human_feedback
  extract -->|created| END
  extract -->|continue| human_feedback
```
