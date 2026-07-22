"""llmcall: one primitive for headless codex -> cc -> claude text-judgment calls.

    from llmcall import call
    r = call("...")            # -> Result; codex -> cc -> claude
    r.text, r.provider         # the answer + who gave it
    str(r), bool(r)            # str-coercible + truthy (drop-in for the old str | None)

`call_chain` is a back-compat shim for the legacy call_chain(prompt, chain, ...) signature (str|None).
"""
from .core import DEFAULT_CHAIN, Attempt, Result, call, refine

__all__ = ["call", "refine", "call_chain", "Result", "Attempt", "DEFAULT_CHAIN"]


def call_chain(prompt, chain=None, providers=None, timeout=180, log=None):
    """Back-compat: same signature/return (str | None) as the original llm_chain.call_chain. The
    `providers` per-provider overrides are accepted and ignored (model/effort now resolve from one
    source); pass model=/effort= to llmcall.call directly if you need an override."""
    r = call(prompt, chain=tuple(chain) if chain else DEFAULT_CHAIN, timeout=timeout)
    if log:
        for a in r.attempts:
            log("llmcall: %s %s" % (a.provider, "answered" if a.ok else "unavailable/empty, trying next"))
    return r.text if r else None
