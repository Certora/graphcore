from typing import Generic, TypeVar, Annotated, Any

from pydantic import BaseModel

from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import StructuredTool, BaseTool

ST = TypeVar("ST")

T_RES = TypeVar("T_RES", bound=str | Command)

class WithInjectedState(BaseModel, Generic[ST]):
    state: Annotated[ST, InjectedState]

class WithInjectedId(BaseModel):
    tool_call_id: Annotated[str, InjectedToolCallId]

class WithImplementation(BaseModel, Generic[T_RES]):
    def run(self) -> T_RES:
        """Override this method to implement the tool logic."""
        raise NotImplementedError("Subclasses must implement run()")
    
    @classmethod
    def as_tool(
        cls,
        name: str
    ) -> BaseTool:
        impl_method = getattr(cls, "run")
        
        # Simple wrapper - just accept kwargs, instantiate model, call method
        def wrapper(**kwargs: Any) -> Any:
            instance = cls(**kwargs)
            return impl_method(instance)
        
        return StructuredTool.from_function(
            func=wrapper,
            args_schema=cls,
            description=cls.__doc__,
            name=name,
        )

class InjectAll(WithInjectedState[ST], WithInjectedId):
    pass
