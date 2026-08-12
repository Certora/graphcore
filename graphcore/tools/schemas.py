from typing import (
    Generic, TypeVar, Annotated, Any, ClassVar, override, Iterator, cast, Mapping, Callable, Never
)
import typing
import string
import re
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic import BaseModel, Field, create_model

from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import StructuredTool, BaseTool

ST = TypeVar("ST")

type BareResult = str | dict

T_RES = TypeVar("T_RES", bound=BareResult | list[BareResult] | Command)

class WithInjectedState(BaseModel, Generic[ST]):
    state: Annotated[ST, InjectedState]

class WithInjectedId(BaseModel):
    tool_call_id: Annotated[str, InjectedToolCallId]

class WithImplementation(BaseModel, Generic[T_RES]):
    def run(self) -> T_RES:
        """Override this method to implement the tool logic."""
        raise NotImplementedError(f"Subclasses ({type(self)}) must implement run()")
    
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
        raise NotImplementedError(f"Subclasses {type(self)} must implement run()")
    
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

@dataclass
class TemplatedTool[T: type[BaseModel], **P]:
    _staged: T

    def with_template(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        assert self._staged.__doc__ is not None
        new_doc = self._staged.__doc__.format(*args, **kwargs)
        assert issubclass(self._staged, BaseModel)
        new_fields : dict[str, Any] = {}
        for (k, v) in self._staged.model_fields.items():
            if not v.description:
                continue
            descr = v.asdict()
            new_attrs = {
                **descr["attributes"],
                "description": v.description.format(*args, **kwargs)
            }
            new_fields[k] = (Annotated[(v.annotation, *descr["metadata"], Field(**new_attrs))], None)
        return create_model(
            f"{self._staged.__name__}Templated",
            __doc__=new_doc,
            __base__=self._staged,
            **new_fields
        )

def _placeholders(fmt: str) -> set[str]:
    to_ret = set()
    for _, fn, _, _ in string.Formatter().parse(fmt):
        if not fn:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", fn) is None:
            raise ValueError("Cannot define non-simple placeholder")
        to_ret.add(fn)
    return to_ret


@typing.dataclass_transform(kw_only_default=True)
class ToolFamilyParams:
    def __new__(cls) -> Never:
        raise ValueError("These are phantom types and never meant to be instantiated")

def tool_family[T:
    WithAsyncDependencies | WithAsyncImplementation | WithImplementation,
    M: ToolFamilyParams,
    **P,
](
    m: Callable[P, M],
) -> Callable[[type[T]], TemplatedTool[type[T], P]]:
    def wrapper(t: type[T]):
        assert isinstance(m, type)
        assert issubclass(m, ToolFamilyParams) and issubclass(t, BaseModel)
        doc = t.__doc__
        assert doc is not None
        params = set()
        params |= _placeholders(doc)
        for (k, v) in t.model_fields.items():
            if not v.description:
                continue
            params |= _placeholders(v.description)
        annots = typing.get_type_hints(m)
        if not (params <= annots.keys()):
            missing = params - annots.keys()
            if missing:
                raise ValueError(f"Missing declared tool params: {missing}")
        return TemplatedTool(t)
    return wrapper
