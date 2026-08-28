"""llmcall: one primitive for headless codex -> cc -> claude text-judgment calls.

    from llmcall import call
    r = call("...")            # -> Result; codex -> cc -> claude
    r.text, r.provider         # the answer + who gave it
    str(r), bool(r)            # str-coercible + truthy (drop-in for the old str | None)

`call_chain` is a back-compat shim for the legacy call_chain(prompt, chain, ...) signature (str|None).
"""
from .core import DEFAULT_CHAIN, Attempt, Result, active_chain, call, call_many, refine

__all__ = ["call", "call_many", "refine", "call_chain", "Result", "Attempt", "DEFAULT_CHAIN",
           "active_chain"]


def call_chain(prompt, chain=None, providers=None, timeout=180, log=None):
    # timeout is passed straight to call(), so it means what it means there: the budget for the
    # WHOLE chain, not for each provider in it. Callers of this wrapper inherited that change
    # without touching their own code, which is why it is written here too.
    """Back-compat: same signature/return (str | None) as the original llm_chain.call_chain. The
    `providers` per-provider overrides are accepted and ignored (model/effort now resolve from one
    source); pass model=/effort= to llmcall.call directly if you need an override."""
    # None, not DEFAULT_CHAIN: naming the constant here would pin the ladder at this layer and make
    # every caller that goes through the shim ignore LLMCALL_CHAIN. Letting call() resolve it is what
    # keeps the override effective for the legacy signature too.
    r = call(prompt, chain=tuple(chain) if chain else None, timeout=timeout)
    if log:
        for a in r.attempts:
            log("llmcall: %s %s" % (a.provider, "answered" if a.ok else "unavailable/empty, trying next"))
    return r.text if r else None
