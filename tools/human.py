from typing import TypeVar, Callable, Literal, Annotated, get_args, get_origin, cast, Any

from pydantic import create_model, Field, BaseModel
from pydantic.fields import FieldInfo

from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

M = TypeVar("M")
S = TypeVar("S")

_injected_state_name = "graphcore_injected_state"

def _process_model_tydict(
    t: type[dict]
) -> tuple[dict[str, Any], str | None]:
    disc : str | None = None
    fields : dict[str, Any] = {}
    for (k, v) in t.__annotations__.items():
        if get_origin(v) is Literal:
            if k != "type":
                raise RuntimeError(f"Illegal type annotation: {v} on {k}")
            disc = get_args(v)[0]
            continue
        elif get_origin(v) is Annotated:
            a = get_args(v)
            if len(a) != 2 or not isinstance(a[1], str):
                raise RuntimeError(f"Illegal type annotation: {v} for {k}")
            fields[k] = (a[0], Field(description=a[1]))
        else:
            raise RuntimeError(f"Illegal type annotation: {v} for {k}")
    return (fields, disc)

def _process_model_basem(
    t: type[BaseModel]
) -> tuple[dict[str, Any], str | None]:
    disc : str | None = None
    fields : dict[str, Any] = {}
    for (k, v) in t.model_fields:
        assert isinstance(v, FieldInfo)
        ty = v.annotation
        assert isinstance(ty, type)
        if get_origin(ty) is Literal:
            if k != "type":
                raise RuntimeError(f"Illegal type annotation: {v} on {k}")
            disc = get_args(v)[0]
            continue
        fields[k] = (v.annotation, Field(description=v.description))
    return (fields, disc)


def human_interaction_tool(
    t: type[M],
    state: type[S],
    name: str,
    state_updater: Callable[[S, M, str], dict] = lambda x, y, z: {}
) -> BaseTool:
    assert issubclass(t, BaseModel) or issubclass(t, dict)
    fields : dict[str, Any]
    disc : str | None
    if issubclass(t, BaseModel):
        (fields, disc) = _process_model_basem(t)
    else:
        (fields, disc) = _process_model_tydict(t)
    
    fields[_injected_state_name] = (Annotated[state, InjectedState], Field())
    fields["tool_call_id"] = (Annotated[str, InjectedToolCallId], Field())

    model = create_model(
        t.__name__,
        __doc__ = t.__doc__,
        **cast(dict[str, Any], fields)
    )
    @tool(name, args_schema=model)
    def interaction_tool(
        **kwargs
    ) -> Command:
        dict_args = {
            k: v for (k, v) in kwargs.items() if k != "tool_call_id" and k != _injected_state_name
        }
        if disc is not None:
            dict_args["type"] = disc
        payload : Any
        if issubclass(t, BaseModel):
            payload = t.model_validate(dict_args)
        else:
            payload = t(**dict_args)
        response = interrupt(payload)
        state_update = state_updater(kwargs[_injected_state_name], payload, response)
        response_update = {
            "messages": [
                ToolMessage(content=response, tool_call_id=kwargs["tool_call_id"])
            ]
        }
        response_update.update(state_update)
        return Command(update=response_update)
    return interaction_tool
