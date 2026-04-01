from typing import Generic, TypeVar, Annotated, Any, ClassVar, override, Iterator, cast

from contextlib import contextmanager
from contextvars import ContextVar

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
    
class WithAsyncImplementation(BaseModel, Generic[T_RES]):
    async def run(self) -> T_RES:
        """Override this method to implement the tool logic."""
        raise NotImplementedError("Subclasses must implement run()")
    
    @classmethod
    def as_tool(
        cls,
        name: str
    ) -> BaseTool:
        impl_method = getattr(cls, "run")
        
        # Simple wrapper - just accept kwargs, instantiate model, call method
        async def wrapper(**kwargs: Any) -> Any:
            instance = cls(**kwargs)
            d = await impl_method(instance)
            return d
        
        return StructuredTool.from_function(
            coroutine=wrapper,
            args_schema=cls,
            description=cls.__doc__,
            name=name,
        )

DEPS = TypeVar("DEPS")

DEPS_BOUND = TypeVar("DEPS_BOUND", bound="WithAsyncDependencies")

class ToolBuilder:
    def __init__(self, ty: type[DEPS_BOUND], deps: object):
        self._ty = ty
        self.deps = deps

    def as_tool(self, name: str) -> BaseTool:
        impl_method = self._ty.run
        
        # Simple wrapper - just accept kwargs, instantiate model, call method
        async def wrapper(**kwargs: Any) -> Any:
            instance = self._ty(**kwargs)
            tok = self._ty._dep_ctx.set(self.deps)
            try:
                d = await impl_method(instance)
                return d
            finally:
                self._ty._dep_ctx.reset(tok)
        
        return StructuredTool.from_function(
            coroutine=wrapper,
            args_schema=self._ty,
            description=self._ty.__doc__,
            name=name,
        )

class WithAsyncDependencies(BaseModel, Generic[T_RES, DEPS]):
    _dep_ctx: ClassVar[ContextVar[object | None]]

    @override
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        cls._dep_ctx = ContextVar(f"_{cls.__name__}_ctx")
        super().__pydantic_init_subclass__(**kwargs)

    async def run(self) -> T_RES:
        raise NotImplementedError("")
    
    @classmethod
    def bind(cls, deps: DEPS) -> ToolBuilder:
        return ToolBuilder(cls, deps)
    
    @contextmanager
    def tool_deps(self) -> Iterator[DEPS]:
        d = type(self)._dep_ctx.get()
        assert d is not None
        yield cast(DEPS, d)

class InjectAll(WithInjectedState[ST], WithInjectedId):
    pass
