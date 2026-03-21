# storyteLLer

Lightweight LangGraph-based storytelling assistant

## Run

- Install dependencies and set environment variables (for example in `.env`).
- Start the CLI:
  - `python -m app.main`
  - `python -m app.main --user-id user123`

## Current Pipeline Schema

### Top-Level Graph (`StorytellerState`)

```mermaid
flowchart TD
    START --> dialogue
    dialogue -->|node=generate_character| generate_character
    dialogue -->|node=generate_world| generate_world
    dialogue -->|node=dialogue| END
    generate_character --> finalize_object
    generate_world --> finalize_object
    finalize_object --> END
```

Routing is controlled by structured JSON from `dialogue`:
- `{"node":"generate_character","response":"..."}`
- `{"node":"generate_world","response":"..."}`
- `{"node":"dialogue","response":"..."}`

### Generator Subgraph (used by character/world nodes)

```mermaid
flowchart TD
    initialize_object --> generate_description
    human_feedback --> generate_description
    generate_description -->|should_extract=extract| extract
    generate_description -->|should_extract=human_feedback| human_feedback
    extract -->|status=created| END
    extract -->|otherwise| human_feedback
```
