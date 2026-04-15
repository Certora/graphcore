from typing import TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def result_type(m: type[T]) -> type[T]:
    import langgraph.checkpoint.serde._msgpack

    to_add = (m.__module__, m.__name__)

    copy = langgraph.checkpoint.serde._msgpack.SAFE_MSGPACK_TYPES.union([to_add])

    langgraph.checkpoint.serde._msgpack.SAFE_MSGPACK_TYPES = copy

    return m
