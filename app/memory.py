from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from langgraph.store.base import BaseStore, Item
from pydantic import BaseModel


def _is_jsonable_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _to_jsonable(value: Any) -> Any:
    if _is_jsonable_primitive(value):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _item_to_record(item: Item) -> dict[str, Any]:
    return {
        "id": item.key,
        "namespace": tuple(item.namespace),
        "value": _to_jsonable(item.value),
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat(),
    }


def list_memories(
    store: Optional[BaseStore], namespace: tuple[str, ...], *, limit: int = 100, offset: int = 0
) -> dict[str, dict[str, Any]]:
    """List memories under a namespace (exact namespace, keyed by id)."""
    if store is None:
        return {}
    results = store.search(namespace, limit=limit, offset=offset)
    out: dict[str, dict[str, Any]] = {}
    for item in results:
        # BaseStore.search returns SearchItem (subclass of Item)
        out[item.key] = _item_to_record(item)
    return out


def get_memory(
    store: Optional[BaseStore],
    namespace: tuple[str, ...],
    memory_id: str,
) -> dict[str, Any] | None:
    """Get a single memory by id within a namespace."""
    if store is None:
        return None
    item = store.get(namespace, memory_id)
    if item is None:
        return None
    return _item_to_record(item)
