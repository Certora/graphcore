# pyright: reportInvalidTypeForm=false
import pytest

from pydantic import BaseModel, Field, create_model, ValidationError
from typing import Annotated, cast, Any, TypedDict, get_args, get_origin

from annotated_types import Gt, Ge, Le, Lt

from hypothesis import HealthCheck, given, settings, strategies as st, Phase

from graphcore.tools.schemas import (
    WithImplementation, WithInjectedState, ToolFamilyParams, tool_family,
)

class TemplateArgValues(TypedDict):
    verb: str
    thing: str

DOC_TEMPLATE = "Hello {verb} {thing}"
TEMPLATE_ARGS: TemplateArgValues = {"verb": "hello", "thing": "world"}

def maybe_bounded_int_type():
    def maybe_bounded_above():
        return st.integers(min_value=0, max_value=2).flatmap(
            lambda flg: \
            st.just(int if flg == 0 else Annotated[int, Lt(10)] if flg == 1 else Annotated[int, Le(10)])
        )
    def maybe_bounded_below():
        return st.integers(min_value=0, max_value=2).flatmap(
            lambda flg: \
            maybe_bounded_above().flatmap(
                lambda wrapped: \
                st.just(wrapped if flg == 0 else Annotated[wrapped, Gt(0)] if flg == 1 else Annotated[wrapped, Ge(0)])
            )
        )
    return maybe_bounded_below()

def base_type(
    include_list: bool
) -> st.SearchStrategy[type]:
    return st.one_of(
        *[
            maybe_bounded_int_type(),
            st.just(bool),
            st.just(str),
            *(
                [maybe_optional_type(include_list=False).map(lambda t: list[t])] if include_list else []
            )
        ]
    )

def maybe_optional_type(
    include_list: bool
):
    return st.booleans().flatmap(
        lambda optional_type: \
        base_type(include_list) if not optional_type else base_type(include_list).map(lambda ty: ty | None)
    )

def field_type():
    return maybe_optional_type(True)

def sample_fields(
    n_fields: int
) -> st.SearchStrategy[list[type]]:
    return cast(st.SearchStrategy[list[type]], st.lists(
        field_type(),
        min_size=n_fields,
        max_size=n_fields
    ))

vocab = ["red", "green", "blue", "cat", "{verb}", "{thing}", "jumps"]

class FamilyParams(ToolFamilyParams):
    verb: str
    thing: str

def field_info():
    phrases = st.lists(st.sampled_from(vocab), min_size=1, max_size=5).map(" ".join)
    return phrases.map(lambda s: Field(description=s))

def to_field_defs(l: list[type]) -> st.SearchStrategy[dict[str, Any]]:
    to_res = [
        field_info().map(
            lambda fi, i=i, t=t: (f"field_{i}", (t, fi))
        ) for (i, t) in enumerate(l)
    ]
    return st.tuples(*to_res).map(lambda d: {
        cast(str, k): cast(Any, v) for (k,v) in d
    })

type RawSchemaDef = dict[str, Any]

def random_schema_raw() -> st.SearchStrategy[dict[str, Any]]:
    return st.integers(1, 4).flatmap(
        sample_fields
    ).flatmap(to_field_defs)



def with_permissive_schema(x: RawSchemaDef) -> tuple[type[BaseModel], type[BaseModel], type[BaseModel]]:
    # get what pydantic thinks the types of these fields should be
    model = create_model(
        "TestModelBase",
        **x
    )

    # for each such field, create a shema which erases all bounds (for ints) and makes the fields optional
    permissive_fields : dict[str, Any] = {}
    for (k, v) in model.model_fields.items():
        field_ty = cast(type, v.asdict()["annotation"])
        permissive_fields[k] = field_ty | None
    permissive_model = create_model(
        "PermissiveModel",
        **permissive_fields
    )

    # create a schema inheriting from `WithImplementation` with the field information parsed via `TestModelBase`
    tool_family_able = create_model(
        "TemplatedModel",
        __doc__=DOC_TEMPLATE,
        __base__=(WithImplementation,),
        **x
    )

    # and create the templated version
    templated=tool_family(FamilyParams)(tool_family_able).with_template(**TEMPLATE_ARGS)

    return (permissive_model, tool_family_able, templated)


def validation_outcome(model: type[BaseModel], payload: dict[str, Any]) -> tuple[str, Any] | str:
    """Collapse validation into a comparable value: parsed dump on success,
    the (loc, type) error signature on failure."""
    try:
        return ("ok", model.model_validate(payload).model_dump())
    except ValidationError as e:
        return "err"
 

@given(st.data())
@settings(
    max_examples=1000,          # crank as desired; replaces `range(0, 1000)`
    deadline=None,             # 3x create_model per example blows the 200ms default
    suppress_health_check=[HealthCheck.too_slow],
)
def test_tool_family_preserves_validation(data: st.DataObject) -> None:
    raw = data.draw(random_schema_raw(), label="schema")

    raw_descriptions = {k: fi.description for k, (_ty, fi) in raw.items()}

    permissive_model, basic, templated = with_permissive_schema(raw)


    # Bonus property: {verb}/{thing} placeholders in the generated schema are substituted.
    assert templated.__doc__ == DOC_TEMPLATE.format(**TEMPLATE_ARGS), \
        f"__doc__ not templated: {templated.__doc__!r}"
    actual_desc = {k: templated.model_fields[k].description for k in raw_descriptions}
    expected_desc = {k: d.format(**TEMPLATE_ARGS) for k, d in raw_descriptions.items()}
    assert actual_desc == expected_desc, \
        f"descriptions not templated: {actual_desc} != {expected_desc}"


    # generate a dict representation of a sample from the permissive model
    # remember, representation (might) have fields outside of the declared bounds
    # or None (where `basic` is a non-none field)
    payload = data.draw(st.from_type(permissive_model), label="payload").model_dump()

    key_universe = sorted(payload)

    # Also exercise missing-field errors, which always-present-keys can't reach
    # delete some random keys from the source
    for k in data.draw(st.sets(st.sampled_from(key_universe)), label="dropped keys"):
        del payload[k]

    # now randomly overwriting some fields in the basic model with some arbitrarily chosen values
    for mut in data.draw(st.sets(st.sampled_from(key_universe), max_size=2), label="wrong keys"):
        random_value = data.draw(st.sampled_from([
            3, "hello", False, [3]
        ]))
        payload[mut] = random_value

    # now, check the validation behavior
    # if payload had legal mutations (deleted no fields, generated all values in range, etc.)
    # both validations should dump the same representation. Otherwise we should see
    # a difference in outcomes
    assert validation_outcome(basic, payload) == validation_outcome(templated, payload)


# ---------------------------------------------------------------------------
# Transitive templating: fields whose annotations mention another tool family
# are templated with the same arguments as the enclosing family.
# ---------------------------------------------------------------------------

class RecipeParams(ToolFamilyParams):
    dish: str


class Ingredient(BaseModel):
    """An ingredient of the {dish}"""
    name: str = Field(description="Name of the ingredient in the {dish}")
    amount: int = Field(description="How much of it to use")


IngredientFamily = tool_family(RecipeParams)(Ingredient)


class MakeRecipe(WithImplementation):
    """Write a recipe for the {dish}"""
    title: str = Field(description="Title of the {dish} recipe")
    main: IngredientFamily = Field(description="The main ingredient") #type: ignore[invalidTypeForm]
    extras: list[IngredientFamily] = Field(description="Additional ingredients")
    garnish: IngredientFamily | None = Field(default=None, description="Optional garnish")


RecipeFamily = tool_family(RecipeParams)(MakeRecipe)


def test_transitive_template_direct_field():
    recipe = RecipeFamily.with_template(dish="paella")

    assert recipe.__name__ == "MakeRecipe"
    assert recipe.__doc__ == "Write a recipe for the paella"
    assert recipe.model_fields["title"].annotation is str
    assert recipe.model_fields["title"].description == "Title of the paella recipe"

    main_ty = recipe.model_fields["main"].annotation
    assert isinstance(main_ty, type) and issubclass(main_ty, Ingredient)
    assert main_ty.__doc__ == "An ingredient of the paella"
    assert main_ty.model_fields["name"].description == "Name of the ingredient in the paella"


def test_transitive_template_inside_containers():
    recipe = RecipeFamily.with_template(dish="soup")

    extras_ty = recipe.model_fields["extras"].annotation
    assert get_origin(extras_ty) is list
    (elem_ty,) = get_args(extras_ty)
    assert issubclass(elem_ty, Ingredient)
    assert elem_ty.__doc__ == "An ingredient of the soup"

    garnish_ty = recipe.model_fields["garnish"].annotation
    garnish_args = get_args(garnish_ty)
    assert type(None) in garnish_args
    (inner_ty,) = [a for a in garnish_args if a is not type(None)]
    assert issubclass(inner_ty, Ingredient)
    assert inner_ty.__doc__ == "An ingredient of the soup"


def test_transitive_template_validation():
    recipe = RecipeFamily.with_template(dish="stew")

    parsed = recipe.model_validate({
        "title": "Beef stew",
        "main": {"name": "beef", "amount": 2},
        "extras": [{"name": "carrot", "amount": 3}],
        "garnish": None,
    })
    assert parsed.main.amount == 2
    assert parsed.extras[0].name == "carrot"

    with pytest.raises(ValidationError):
        recipe.model_validate({
            "title": "Beef stew",
            "main": {"name": "beef"},  # missing amount
            "extras": [],
            "garnish": None,
        })


def test_templated_instances_are_independent():
    soup = RecipeFamily.with_template(dish="soup")
    pie = RecipeFamily.with_template(dish="pie")

    soup_main = soup.model_fields["main"].annotation
    pie_main = pie.model_fields["main"].annotation
    assert soup_main is not pie_main
    assert soup_main.__doc__ == "An ingredient of the soup"
    assert pie_main.__doc__ == "An ingredient of the pie"


def test_transitive_template_without_field_description():
    class Pantry(WithImplementation):
        """Check the pantry for the {dish}"""
        staple: IngredientFamily

    pantry = tool_family(RecipeParams)(Pantry).with_template(dish="curry")

    staple_ty = pantry.model_fields["staple"].annotation
    assert isinstance(staple_ty, type) and issubclass(staple_ty, Ingredient)
    assert staple_ty.__doc__ == "An ingredient of the curry"


def test_inconsistent_key_types_rejected():
    class GardenParams(ToolFamilyParams):
        dish: str

    with pytest.raises(ValueError, match="inconsistent key types"):
        @tool_family(GardenParams)
        class BadRecipe(WithImplementation):
            """Write a recipe for the {dish}"""
            main: IngredientFamily = Field(description="The main ingredient")

    with pytest.raises(ValueError, match="inconsistent key types"):
        @tool_family(GardenParams)
        class BadRecipeNested(WithImplementation):
            """Write a recipe for the {dish}"""
            extras: list[IngredientFamily] = Field(description="Additional ingredients")

    with pytest.raises(ValueError, match="inconsistent key types"):
        @tool_family(GardenParams)
        class BadRecipeUndescribed(WithImplementation):
            """Write a recipe for the {dish}"""
            staple: IngredientFamily


def test_injected_state_subscription_copies_doc():
    class Slice(TypedDict):
        n: int

    class Tool[T](WithInjectedState[T]):
        """the description"""

    concrete = Tool[Slice]
    assert concrete.__doc__ == "the description"
    assert concrete.model_fields["state"].annotation is Slice
