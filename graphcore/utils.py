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

import logging
from typing import TypedDict, List, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


class TokenUsageDict(TypedDict):
    """Dictionary for accumulating token usage across LLM calls."""
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    model_name: str | None

def get_token_usage(m: AIMessage) -> TokenUsageDict:
    """Per-bucket raw counts for one response, read from ``usage_metadata`` — the
    one field every provider and request surface populates.

    The three input buckets are disjoint, and each bills at its own rate.
    ``usage_metadata`` reports an inclusive input total, so the cache buckets are
    subtracted back out to leave the fresh input ``input_tokens`` means here."""
    normalized = get_normalized_token_usage(m)
    cache_read = normalized["cache_read_tokens"]
    cache_write = normalized["cache_write_tokens"]

    # A gateway normalizing usage from an arbitrary upstream can report a bucket
    # bigger than the total it belongs to. Unclamped that lands as a negative count,
    # which subtracts from the run's totals and bills below zero instead of failing.
    fresh_input = normalized["total_input_tokens"] - cache_read - cache_write
    if fresh_input < 0:
        logger.warning(
            "%s reported cache buckets (read %d, write %d) exceeding its %d-token "
            "input total; counting fresh input as 0.",
            normalized["model_name"], cache_read, cache_write,
            normalized["total_input_tokens"],
        )
        fresh_input = 0

    return {
        "input_tokens": fresh_input,
        "output_tokens": normalized["total_output_tokens"],
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
        "model_name": normalized["model_name"],
    }

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
            if len(item) == 0: # some providers crash on empty text blocks, even in the context of non-empty messages
                continue
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
