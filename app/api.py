"""HTTP API for storyteLLer: threads, SSE-streamed turns, story edits, and saves.

Run with: uv run uvicorn app.api:app --reload
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator, Optional

import dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app import persistence
from app.graph import Storyteller
from app.state.schemas import Character, World, coerce_story

dotenv.load_dotenv()

_DEFAULT_CORS_ORIGINS = "http://localhost:5173"


class ThreadCreated(BaseModel):
    thread_id: str
    greeting: str


class TurnRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatMessage(BaseModel):
    role: str
    content: str


class ThreadState(BaseModel):
    phase: str
    turn: int
    story: Optional[dict[str, Any]]
    awaiting_feedback: bool
    pending_prompt: Optional[str]
    messages: list[ChatMessage]


class SaveInfo(BaseModel):
    story_id: str
    title: str
    saved_at: str
    phase: str
    turn: int


class ResumedThread(BaseModel):
    thread_id: str
    state: ThreadState


def _known_lock_fields(names: list[str], model: type[BaseModel]) -> list[str]:
    unknown = [name for name in names if name not in model.model_fields]
    if unknown:
        raise ValueError(f"Unknown lock fields: {', '.join(unknown)}")
    return names


class WorldUpdate(BaseModel):
    """Full world replacement, matching the editor's Save world action."""

    world: World
    world_locked_fields: list[str] = []

    @field_validator("world_locked_fields")
    @classmethod
    def locks_are_world_fields(cls, names: list[str]) -> list[str]:
        return _known_lock_fields(names, World)


class CharacterUpdate(BaseModel):
    """Full character replacement, matching the editor's Save character action."""

    character: Character
    locked_fields: list[str] = []

    @field_validator("locked_fields")
    @classmethod
    def locks_are_character_fields(cls, names: list[str]) -> list[str]:
        return _known_lock_fields(names, Character)


def get_storyteller(request: Request) -> Storyteller:
    return request.app.state.storyteller


StorytellerDep = Annotated[Storyteller, Depends(get_storyteller)]
UserIdHeader = Annotated[str, Header()]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(storyteller: Optional[Storyteller] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "storyteller", None) is None:
            app.state.storyteller = Storyteller()
        yield

    app = FastAPI(title="storyteLLer", lifespan=lifespan)
    app.state.storyteller = storyteller
    app.state.thread_locks = {}

    origins = os.getenv("STORYTELLER_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def thread_lock(thread_id: str) -> asyncio.Lock:
        return app.state.thread_locks.setdefault(thread_id, asyncio.Lock())

    async def require_state(st: Storyteller, user_id: str, thread_id: str) -> dict:
        state = await st.get_state(user_id, thread_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found")
        return state

    @asynccontextmanager
    async def editable_story(st: Storyteller, user_id: str, thread_id: str):
        """Hold the turn lock and yield the current story when manual edits are allowed."""
        await require_state(st, user_id, thread_id)
        lock = thread_lock(thread_id)
        if lock.locked():
            raise HTTPException(status_code=409, detail="A turn is already in progress")
        async with lock:
            state = await require_state(st, user_id, thread_id)
            if state["awaiting_feedback"]:
                raise HTTPException(
                    status_code=409,
                    detail="Finish the current step in Chat before editing.",
                )
            story = coerce_story(state["story"])
            if story is None or story.world is None:
                raise HTTPException(
                    status_code=404,
                    detail="No world yet. Create one in Chat first.",
                )
            yield story

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/threads", response_model=ThreadCreated, status_code=201)
    async def create_thread(
        st: StorytellerDep,
        x_user_id: UserIdHeader = "1",
    ) -> ThreadCreated:
        thread_id = str(uuid.uuid4())
        greeting = await st.init(x_user_id, thread_id)
        return ThreadCreated(thread_id=thread_id, greeting=greeting)

    @app.get("/threads/{thread_id}/state", response_model=ThreadState)
    async def read_state(
        thread_id: str,
        st: StorytellerDep,
        x_user_id: UserIdHeader = "1",
    ) -> dict:
        return await require_state(st, x_user_id, thread_id)

    @app.post("/threads/{thread_id}/turns")
    async def send_turn(
        thread_id: str,
        body: TurnRequest,
        st: StorytellerDep,
        x_user_id: UserIdHeader = "1",
    ) -> StreamingResponse:
        await require_state(st, x_user_id, thread_id)
        lock = thread_lock(thread_id)
        if lock.locked():
            raise HTTPException(status_code=409, detail="A turn is already in progress")
        # Held for the whole stream so concurrent turns on one thread are rejected, not queued.
        await lock.acquire()

        async def events() -> AsyncIterator[str]:
            try:
                async for event in st.tell_stream(body.message, x_user_id, thread_id):
                    yield _sse(event["type"], event)
                yield _sse("done", {})
            finally:
                lock.release()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.patch("/threads/{thread_id}/world", response_model=ThreadState)
    async def update_world(
        thread_id: str,
        body: WorldUpdate,
        st: StorytellerDep,
        x_user_id: UserIdHeader = "1",
    ) -> dict:
        async with editable_story(st, x_user_id, thread_id) as story:
            updated = story.model_copy(
                update={"world": body.world, "world_locked_fields": body.world_locked_fields}
            )
            await st.update_story(updated, x_user_id, thread_id)
            return await require_state(st, x_user_id, thread_id)

    @app.patch("/threads/{thread_id}/characters/{object_id}", response_model=ThreadState)
    async def update_character(
        thread_id: str,
        object_id: str,
        body: CharacterUpdate,
        st: StorytellerDep,
        x_user_id: UserIdHeader = "1",
    ) -> dict:
        async with editable_story(st, x_user_id, thread_id) as story:
            try:
                updated = story.update_character(object_id, body.character, body.locked_fields)
            except KeyError:
                raise HTTPException(
                    status_code=404, detail=f"Character '{object_id}' not found"
                ) from None
            await st.update_story(updated, x_user_id, thread_id)
            return await require_state(st, x_user_id, thread_id)

    @app.get("/saves", response_model=list[SaveInfo])
    async def list_saves(st: StorytellerDep) -> list[dict]:
        if not st.saves_dir:
            return []
        return persistence.list_saves(st.saves_dir)

    @app.post("/saves/{story_id}/resume", response_model=ResumedThread)
    async def resume_save(
        story_id: str,
        st: StorytellerDep,
        x_user_id: UserIdHeader = "1",
    ) -> dict:
        path = persistence.find_save(st.saves_dir, story_id) if st.saves_dir else None
        if path is None:
            raise HTTPException(status_code=404, detail=f"Save '{story_id}' not found")
        thread_id = str(uuid.uuid4())
        await st.load(persistence.load_story(path), x_user_id, thread_id)
        return {"thread_id": thread_id, "state": await require_state(st, x_user_id, thread_id)}

    return app


app = create_app()
