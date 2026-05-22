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

from typing import TypedDict, Literal
from langchain_core.messages import AIMessage


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
