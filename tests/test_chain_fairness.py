# -*- coding: utf-8 -*-
"""A provider that hangs must not be able to starve the ones behind it.

The whole-chain budget fixed an unbounded worst case, and introduced this: the first provider gets
`remaining`, which on a hang is everything, so the chain returns "budget exhausted" without ever
having tried the fallback. Measured against a codex launcher that accepts the call and then hangs:

    total 90.9s against a 90s budget
    codex  ok=False  ms=90872  err=timeout after 90s
    cc     ok=False  ms=0      err=chain budget exhausted     <- never attempted

That is not a fallback ladder, it is a single provider with extra steps. The point of ordering
providers is that the next one is tried when the previous cannot answer, and "cannot answer"
includes "sat there".

So an attempt is capped at its FAIR SHARE of what is left: remaining / providers-still-to-try. It
stays adaptive rather than a fixed slice, because a provider that answers quickly should hand its
unused time to the ones behind it, not have it deducted.
"""
import time

import pytest

from llmcall import call, core


@pytest.fixture(autouse=True)
def _tiny_floor(monkeypatch):
    monkeypatch.setattr(core, "_MIN_PROVIDER_BUDGET", 0.01)


def test_a_hanging_first_provider_does_not_starve_the_rest(monkeypatch):
    """The headline. A hang must cost its share, not the whole ladder."""
    seen = []

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        seen.append(name)
        if name == "codex":
            time.sleep(max(timeout, 0))          # accepts, then sits there
            return None, "timeout after %.3fs" % timeout
        return "PONG", None

    monkeypatch.setattr(core, "_invoke", fake)
    t0 = time.time()
    r = call("q", chain=["codex", "cc", "claude"], timeout=0.30)
    elapsed = time.time() - t0

    assert "cc" in seen, (
        "the hang consumed the whole budget and the fallback was never tried: %r" % (seen,))
    assert r.text == "PONG", "the chain did not recover onto a working provider: %r" % (r.error,)
    assert elapsed <= 0.30 + core._MIN_PROVIDER_BUDGET * 2, (
        "recovering must not cost more than the budget: %.3fs" % elapsed)


def test_a_fast_first_provider_hands_its_unused_time_on(monkeypatch):
    """Negative control for over-correction.

    A fixed slice would cap every provider at budget/N whether or not the ones before it were quick.
    The share is of what is LEFT, so provider two here must be offered clearly more than a third.
    """
    handed = []

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        handed.append((name, timeout))
        if name == "codex":
            return None, "unavailable"           # fails instantly, spends nothing
        return "PONG", None

    monkeypatch.setattr(core, "_invoke", fake)
    call("q", chain=["codex", "cc", "claude"], timeout=0.30)

    by = dict(handed)
    assert "cc" in by, "the second provider was never reached: %r" % (handed,)
    assert by["cc"] > 0.30 / 3 * 1.2, (
        "a provider that failed instantly still cost the next one its share: %r" % (handed,))


def test_the_total_is_still_bounded(monkeypatch):
    """The property the whole-chain budget exists for must survive the fairness cap."""
    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        time.sleep(max(timeout, 0))
        return None, "hung"

    monkeypatch.setattr(core, "_invoke", fake)
    t0 = time.time()
    call("q", chain=["codex", "cc", "claude"], timeout=0.30)
    elapsed = time.time() - t0
    assert elapsed <= 0.30 + core._MIN_PROVIDER_BUDGET * 2, (
        "three hanging providers overran the budget: %.3fs" % elapsed)
