#      The Certora Prover
#      Copyright (C) 2025  Certora Ltd.
#
#      This program is free software: you can redistribute it and/or modify
#      it under the terms of the GNU General Public License as published by
#      the Free Software Foundation, version 3 of the License.
#
#      This program is distributed in the hope that it will be useful,
#      but WITHOUT ANY WARRANTY; without even the implied warranty of
#      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#      GNU General Public License for more details.
#
#      You should have received a copy of the GNU General Public License
#      along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import TypedDict, Literal, List, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.runnables import Runnable


type TokenUsageKeysT = Literal[
    "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
]

_token_usage_keys : list[TokenUsageKeysT] = [
    "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
]

class TokenUsageDict(TypedDict):
    """Dictionary for accumulating token usage across LLM calls."""
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    model_name: str | None

def get_token_usage(m: AIMessage) -> TokenUsageDict:
    to_ret : TokenUsageDict = {
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "model_name": None
    }
    rm = m.response_metadata
    if "model_name" in rm and isinstance(rm['model_name'], str):
        to_ret["model_name"] = rm['model_name']
    if "usage" not in rm or not isinstance(rm["usage"], dict):
        return to_ret
    usage_meta = m.response_metadata["usage"]
    for k in _token_usage_keys:
        tok = usage_meta.get(k, 0)
        if not isinstance(tok, int):
            continue # be cool
        to_ret[k] = to_ret[k] + tok
    return to_ret

class NormalizedTokenUsage(TypedDict):
    total_input_tokens: int
    total_output_tokens: int

    cache_read_tokens: int
    cache_write_tokens: int
    thinking_tokens: int

    model_name: str | None

def get_normalized_token_usage(m: AIMessage) -> NormalizedTokenUsage:
    to_ret : NormalizedTokenUsage = {
        "total_input_tokens": 0,
        "model_name": m.response_metadata.get("model_name"),
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "thinking_tokens": 0,
        "total_output_tokens": 0
    }

    if not (usage := m.usage_metadata):
        return to_ret
    
    to_ret["total_input_tokens"] = usage["input_tokens"]
    to_ret["total_output_tokens"] = usage["output_tokens"]

    if "output_token_details" in usage:
        out_details = usage["output_token_details"]
        to_ret["thinking_tokens"] = out_details.get("reasoning", 0)
    if "input_token_details" in usage:
        in_details = usage["input_token_details"]
        to_ret["cache_read_tokens"] = in_details.get("cache_read", 0)
        
        cache_write = in_details.get("cache_creation", 0)
        if not cache_write:
            # thanks langchain
            for t in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                cache_write += in_details.get(t, 0)
        to_ret["cache_write_tokens"] = cache_write
    
    return to_ret

def current_prompt_tokens(messages: List[AnyMessage]) -> int:
    """
    Effective context size of the most recent LLM call, used to decide when to summarize.

    Returns input + cache-read + cache-creation tokens from the latest AIMessage. ToolMessages
    appended after that AIMessage are not counted (router fires after TOOLS_NODE) and the
    summarizer's own AIMessage is discarded before reaching state. Both are small enough that
    the threshold should be set with headroom anyway.
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            usage = get_normalized_token_usage(m)
            return usage["total_input_tokens"]
    return 0


def default_max_prompt_tokens(model_name: str | None) -> int:
    """
    Prompt-token threshold at which to compact history. Keep this conservatively below the model's
    context window to leave room for output, thinking budget, and the next batch of tool results.
    Add a new case here when introducing a new model.
    """
    match model_name:
        case "claude-opus-4-6":
            return 500_000   # 1M context window
        case "claude-sonnet-4-6":
            return 500_000   # 1M context window
        case "claude-opus-4-7":
            return 500_000   # 1M context window
        case _:
            return 100_000   # fallback for unknown models


# ---------------------------------------------------------------------------
# Content normalization for LLM invocation
#
# OpenAI's Chat Completions API rejects bare strings inside a list-shaped
# ``content`` — every list element must be a content-part dict with a
# ``type`` key. Anthropic's Messages API is more permissive and tolerates
# ``list[str | dict]``, but it also accepts the strict ``list[dict]``
# form, so we normalize everything to ``list[dict]`` unconditionally
# before invoking. ``invoke`` / ``ainvoke`` are the wrappers every
# workflow LLM call should go through.
# ---------------------------------------------------------------------------


def _normalize_content(content: str | list[str | dict]) -> str | list[dict]:
    """Promote bare strings inside a list-content to ``{"type": "text",
    "text": s}`` dicts. ``str`` content (the single-text form) is
    passed through unchanged."""
    if not isinstance(content, list):
        return content
    out: list[dict] = []
    for item in content:
        if isinstance(item, str):
            out.append({"type": "text", "text": item})
        else:
            out.append(item)
    return out


def _normalize_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Return a list of messages whose ``content`` (where list-shaped)
    has every bare string promoted to a text content-part dict. Each
    affected message is copied; messages that already conform are
    passed through unchanged so we don't churn references that the
    caller may still hold."""
    out: list[BaseMessage] = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            normalized = _normalize_content(content)
            if normalized is not content:
                m = m.model_copy(update={"content": normalized})
        out.append(m)
    return out


def invoke(
    llm: BaseChatModel | Runnable,
    messages: Sequence[BaseMessage],
    **kwargs,
) -> BaseMessage:
    """Synchronous LLM invocation wrapper that normalizes message
    content shapes before calling ``llm.invoke``. Use this in place of
    ``llm.invoke(messages)`` everywhere a workflow talks to the model
    — the normalization keeps OpenAI's Chat Completions happy and
    leaves Anthropic's Messages API behavior unchanged."""
    return llm.invoke(_normalize_messages(messages), **kwargs)


async def ainvoke(
    llm: BaseChatModel | Runnable,
    messages: Sequence[BaseMessage],
    **kwargs,
) -> BaseMessage:
    """Async counterpart to :func:`invoke`."""
    return await llm.ainvoke(_normalize_messages(messages), **kwargs)
