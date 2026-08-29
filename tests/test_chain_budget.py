# -*- coding: utf-8 -*-
"""`timeout` bounds the whole chain, not each provider in it.

The bug this pins down: the same `timeout` was handed to every provider, so a caller asking for
40 minutes could wait 40 minutes PER provider. The caller cannot compensate for that, because it
does not choose the chain: active_chain() reads LLMCALL_CHAIN, so an operator adding one provider
silently doubles the worst case of every caller on the machine.

Measured in the wild before the fix: a daily job passed timeout=2400 with a two provider chain and
spent 80 minutes discovering that both were hanging (08:07 to 08:47, then 08:47 to 09:27). Its
scheduled task has a two hour limit, so it was killed while writing output it had already produced.
An 80 minute wait is not a slow call, it is a caller that has lost control of its own clock.

Time is real here rather than mocked: the property under test IS wall clock, and a mocked clock
would let a regression that actually blocks pass green. The budgets are kept in tens of
milliseconds so the file stays fast.
"""
import time

import pytest

from llmcall import call, core


@pytest.fixture(autouse=True)
def _tiny_floor(monkeypatch):
    """Scale the per provider floor down to match the millisecond budgets used here.

    Without this the floor (15s, sized for a real model call) exceeds every budget in the file and
    every provider is skipped, which would make the tests pass for a reason that has nothing to do
    with the behaviour they claim to check.
    """
    monkeypatch.setattr(core, "_MIN_PROVIDER_BUDGET", 0.01)


def _hanging(record):
    """A provider that burns exactly the budget it was handed, then reports unavailable.

    This is the shape that matters: a provider which is not refusing quickly but sitting there.
    Fast failures never expose the bug, because the chain gets through them either way.
    """
    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        record.append((name, timeout))
        time.sleep(max(timeout, 0))
        return None, "timeout after %.3fs" % timeout
    return fake


def test_a_chain_of_hanging_providers_still_returns_within_the_budget(monkeypatch):
    """The headline property. Three dead providers, one budget, one budget's worth of waiting.

    Against the old per provider code this took ~3x the budget and failed on the elapsed assert.
    """
    seen = []
    monkeypatch.setattr(core, "_invoke", _hanging(seen))

    t0 = time.time()
    r = call("x", chain=["codex", "cc", "claude"], timeout=0.30)
    elapsed = time.time() - t0

    assert not r, "all three providers were dead, the call must not report success"
    assert elapsed < 0.30 * 1.6, (
        "chain took %.3fs on a 0.30s budget; %d providers were tried with budgets %r"
        % (elapsed, len(seen), [round(b, 3) for b in (t for _, t in seen)]))


def test_each_provider_is_offered_only_what_is_left(monkeypatch):
    """The mechanism behind the property, asserted directly.

    Elapsed time alone would also be satisfied by a chain that gave up after one provider. This
    pins that the budget is being SHARED: every attempt gets strictly less than the one before,
    and no attempt is ever offered more than the caller asked for in total.
    """
    seen = []
    # NOT _hanging: a provider that burns its WHOLE budget leaves nothing for the next one, so only
    # the first is ever attempted and every pairwise assertion below compares an empty sequence.
    # Measured 2026-08-28: against correct code this test saw exactly [('codex', 0.3)], so
    # `all(b < a for a, b in zip(...))` was ranging over zero pairs and passing vacuously, and
    # reverting the whole change to per-provider budgets did not redden it. The comment that used to
    # be here claimed a poisoning run produced [0.2, 0.2, 0.2]; with a fully-burning provider that
    # cannot happen, so that verification was never of this test.
    # A provider that burns a FRACTION is what makes sharing observable.
    def partial(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        seen.append((name, timeout))
        time.sleep(max(timeout, 0) * 0.3)
        return None, "unavailable"

    monkeypatch.setattr(core, "_invoke", partial)
    call("x", chain=["codex", "cc", "claude"], timeout=0.30)

    budgets = [b for _, b in seen]
    assert len(budgets) >= 2, (
        "fewer than two providers were attempted, so every pairwise assertion below is vacuous "
        "and this test would pass with the sharing removed entirely: %r" % (seen,))
    assert all(b <= 0.30 + 1e-9 for b in budgets), \
        "a provider was offered more than the whole budget: %r" % (budgets,)
    # Strictly decreasing, not merely non increasing. `sorted(reverse=True)` is satisfied by a list
    # of identical values, which is exactly the broken behaviour: three providers each handed the
    # full timeout passes that check while sharing nothing. Re-verified by poisoning on 2026-08-28
    # with the fractional provider above: restoring the per-provider budget gives [0.3, 0.3, 0.3]
    # and this assertion goes red, which it did not do before that provider was changed.
    assert all(b < a for a, b in zip(budgets, budgets[1:])), \
        "each attempt must get strictly less than the one before, got %r" % (budgets,)


def test_a_provider_reached_with_nothing_left_is_recorded_not_silently_dropped(monkeypatch):
    """Running out of time must be legible.

    A chain that exhausted its budget and a chain whose providers were all genuinely unavailable
    are different problems with different fixes (raise the budget vs repair the provider). If the
    skipped providers just vanish from the attempt list, the next reader cannot tell which one
    they are looking at.
    """
    seen = []
    monkeypatch.setattr(core, "_invoke", _hanging(seen))

    # With a budget the chain can actually divide, the reserve now guarantees every provider a turn
    # even when the first one hangs, so nothing is skipped and nothing says "exhausted". That is the
    # newer guarantee, asserted here so the two states cannot be confused for each other.
    r = call("x", chain=["codex", "cc", "claude"], timeout=0.30)
    assert [a.provider for a in r.attempts] == ["codex", "cc", "claude"], \
        "every provider in the chain must appear in the attempt list"
    assert len(seen) == 3, \
        "the reserve guarantees each provider a turn, but only %r were invoked" % (seen,)

    # The original point of this test stands and still needs a case of its own: a budget that CANNOT
    # cover the chain must SAY so, rather than looking like three broken providers. Below one floor
    # per provider there is genuinely nothing to hand out, and the skip and its wording are what tell
    # a reader to raise the budget instead of going off to repair providers that are fine.
    seen2 = []
    monkeypatch.setattr(core, "_invoke", _hanging(seen2))
    r2 = call("x", chain=["codex", "cc", "claude"], timeout=core._MIN_PROVIDER_BUDGET * 1.2)
    assert [a.provider for a in r2.attempts] == ["codex", "cc", "claude"], \
        "a skipped provider must still appear, or a reader cannot see that it was skipped"
    exhausted = [a for a in r2.attempts if a.error and "budget" in a.error]
    assert exhausted, "a budget too small for the chain said nothing about the budget: %r" \
        % ([a.error for a in r2.attempts],)
    assert len(seen2) < 3, "a provider was invoked despite there being no budget left"


def test_a_single_slow_provider_still_gets_the_whole_budget(monkeypatch):
    """Negative control for over correction.

    Splitting the budget up front (timeout / len(chain)) would also bound the total, and would
    also pass the headline test. It would be wrong: the common case is that the FIRST provider
    answers, and it must get the caller's full patience, not a third of it.
    """
    seen = []

    def slow_but_fine(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        seen.append((name, timeout))
        time.sleep(0.05)
        return "answer from %s" % name, None

    monkeypatch.setattr(core, "_invoke", slow_but_fine)
    r = call("x", chain=["codex", "cc", "claude"], timeout=0.30)

    assert r.provider == "codex" and r.text == "answer from codex"
    assert len(seen) == 1, "the chain kept going after a provider answered"
    assert seen[0][1] > 0.30 * 0.9, \
        "the only provider tried got %.3fs of a 0.30s budget: the budget was pre-divided" \
        % (seen[0][1],)


def test_a_healthy_chain_is_unaffected(monkeypatch):
    """Negative control for scope. Fast providers must behave exactly as before: fall through the
    dead one, answer from the next, no budget bookkeeping visible in the outcome."""
    def mixed(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        if name == "codex":
            return None, "not configured"
        return "hi from %s" % name, None

    monkeypatch.setattr(core, "_invoke", mixed)
    r = call("x", chain=["codex", "cc", "claude"], timeout=0.30)

    assert r.provider == "cc" and r.text == "hi from cc"
    assert [a.provider for a in r.attempts] == ["codex", "cc"]
    assert not any("budget" in (a.error or "") for a in r.attempts), \
        "a chain that never ran short of time must not mention the budget"


def test_the_floor_is_what_stops_the_loop_not_zero(monkeypatch):
    """Why there is a floor at all.

    With a bare `remaining > 0` check, a provider reached with 40ms left would be launched, and
    for a real model call that is a subprocess spawned only to be killed. This asserts the loop
    stops at the declared floor rather than grinding out sub-second attempts.
    """
    seen = []
    monkeypatch.setattr(core, "_MIN_PROVIDER_BUDGET", 0.12)
    monkeypatch.setattr(core, "_invoke", _hanging(seen))

    call("x", chain=["codex", "cc", "claude"], timeout=0.20)

    assert len(seen) == 1, \
        "expected the second provider to be below the 0.12s floor, got budgets %r" % (seen,)
