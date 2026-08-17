from typing import (
    Generic, TypeVar, Annotated, Any, ClassVar,
    override, Iterator, cast, Callable, Never, get_args, get_origin
)
import types
import typing
import string
import re
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from functools import reduce
from operator import or_

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


@typing.dataclass_transform(kw_only_default=True)
class ToolFamilyParams:
    def __new__(cls) -> Never:
        raise ValueError("These are phantom types and never meant to be instantiated")

def _placeholders(fmt: str) -> set[str]:
    to_ret = set()
    for _, fn, _, _ in string.Formatter().parse(fmt):
        if not fn:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", fn) is None:
            raise ValueError("Cannot define non-simple placeholder")
        to_ret.add(fn)
    return to_ret


def map_type[T, U](t: Any, to_rewrite: type[T], f: Callable[[type[T]], type[U]]) -> Any:
    """Rebuild type expression `t`, replacing occurrences of `to_rewrite` with f(match)."""
    # A match rewrites and stops -- we don't descend into the matched type.
    if isinstance(t, type) and issubclass(t, to_rewrite):
        return f(t)

    origin = get_origin(t)
    if origin is None:
        return t  # leaf: plain class, None, Ellipsis, a Literal value, ...

    # Annotated[X, meta...]: walk X, leave metadata alone.
    if origin is Annotated:
        inner, *meta = get_args(t)
        return Annotated[tuple([map_type(inner, to_rewrite, f), *meta])]

    args = get_args(t)
    new_args = tuple(
        [map_type(x, to_rewrite, f) for x in a] if isinstance(a, list)  # Callable's [params]
        else map_type(a, to_rewrite, f)
        for a in args
    )
    if new_args == args:
        return t  # untouched subtree: hand back the original object

    if origin is types.UnionType:  # X | Y can't be rebuilt as origin[args]
        return reduce(or_, new_args)

    return origin[new_args]

class _TemplatedTool[T: type[BaseModel], M: ToolFamilyParams, **P](BaseModel):
    _wrapped: ClassVar[type[BaseModel]]
    _key_type: ClassVar[type[ToolFamilyParams]]

    @classmethod
    def with_template(cls, *args: P.args, **kwargs: P.kwargs) -> T:
        assert cls._wrapped.__doc__ is not None
        new_doc = cls._wrapped.__doc__.format(*args, **kwargs)
        assert issubclass(cls._wrapped, BaseModel)
        new_fields : dict[str, Any] = {}
        def type_mapper(
            t: type[_TemplatedTool]
        ) -> type[Any]:
            return t.with_template(*args, **kwargs)
        for (k, v) in cls._wrapped.model_fields.items():
            actual_type = v.annotation
            if v.annotation is not None:
                actual_type = map_type(v.annotation, _TemplatedTool, type_mapper)
            descr = v.asdict()
            new_attrs = {
                **descr["attributes"],
            }
            if v.description:
                new_attrs["description"] = v.description.format(*args, **kwargs)
            if descr["metadata"]:
                new_fields[k] = (Annotated[actual_type, *descr["metadata"]], Field(**new_attrs))
            else:
                new_fields[k] = (actual_type, Field(**new_attrs))
        return create_model(
            cls._wrapped.__name__,
            __doc__=new_doc,
            __base__=cast(T, cls._wrapped),
            **new_fields
        )

    @staticmethod
    def _for_type[X: BaseModel, K: ToolFamilyParams,  **R](t: type[X], m: type[K]) -> type["_TemplatedTool[type[X], K, R]"]:
        clone = create_model(
            f"{t.__name__}Template",
            __base__=(_TemplatedTool,)
        )
        clone._wrapped = t
        clone._key_type = m

        return clone

def tool_family[
    T: BaseModel,
    M: ToolFamilyParams,
    **P,
](
    m: Callable[P, M],
) -> Callable[[type[T]], type[_TemplatedTool[type[T], M, P]]]:
    def wrapper(t: type[T]):
        assert isinstance(m, type)
        assert issubclass(m, ToolFamilyParams) and issubclass(t, BaseModel)
        doc = t.__doc__
        assert doc is not None
        params = set()
        params |= _placeholders(doc)
        def check_key(nested: type[_TemplatedTool]) -> type[_TemplatedTool]:
            if nested._key_type is not m:
                raise ValueError(
                    f"Cannot use inconsistent key types: {m} vs {nested._key_type} (via {nested.__name__})"
                )
            return nested
        for (k, v) in t.model_fields.items():
            if v.annotation is not None:
                map_type(v.annotation, _TemplatedTool, check_key)
            if not v.description:
                continue
            params |= _placeholders(v.description)
        annots = typing.get_type_hints(m)
        if not (params <= annots.keys()):
            missing = params - annots.keys()
            if missing:
                raise ValueError(f"Missing declared tool params: {missing}")
        return _TemplatedTool._for_type(t, cast(type[M], m))
    return wrapper
