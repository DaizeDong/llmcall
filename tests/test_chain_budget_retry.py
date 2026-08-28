# -*- coding: utf-8 -*-
"""Does the nudge retry respect the chain budget, or does the floor let it past the deadline?

`call()` hands _extract_or_retry `max(deadline - now, _MIN_PROVIDER_BUDGET)`. Past the deadline that
is the floor, not zero, so every provider that needs a retry can spend the floor again after the
budget is gone. The headline test cannot see this: its providers carry no schema, so the retry path
never runs. A schema is the common case for the caller this change was written for.
"""
import time
import pytest
from llmcall import call, core


@pytest.fixture(autouse=True)
def _tiny_floor(monkeypatch):
    monkeypatch.setattr(core, "_MIN_PROVIDER_BUDGET", 0.05)


def test_the_nudge_retry_overruns_by_at_most_one_floor(monkeypatch):
    spent = []

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        spent.append((name, timeout))
        time.sleep(max(timeout, 0))
        return "not json at all", None          # forces _apply to miss -> nudge retry

    monkeypatch.setattr(core, "_invoke", fake)
    monkeypatch.setattr(core, "active_chain", lambda: ("p1", "p2", "p3"))

    budget = 0.30
    t0 = time.time()
    call("q", schema={"type": "object"}, timeout=budget)
    elapsed = time.time() - t0

    print("\n  budget   = %.3fs" % budget)
    print("  elapsed  = %.3fs" % elapsed)
    print("  overrun  = %+.3fs" % (elapsed - budget))
    print("  handed out:")
    for n, t in spent:
        print("    %-4s %.3fs" % (n, t))
    # The floor is granted to the retry even past the deadline, deliberately: handing a nudge zero
    # seconds is not a retry, it is a slower way of failing. So the honest bound is the budget plus
    # ONE floor, not the budget, and not the budget plus one floor per provider -- once the deadline
    # is gone the loop skips everything left. Measured here rather than asserted from the code.
    assert elapsed <= budget + core._MIN_PROVIDER_BUDGET * 1.6, (
        "overran by more than one floor: %.3fs spent against %.3fs + %.3fs"
        % (elapsed, budget, core._MIN_PROVIDER_BUDGET))
    # Assert the FLOOR, not the elapsed time. Inferring it from the clock was too weak to notice the
    # floor being removed: with the retry handed zero, elapsed is still a hair over budget from
    # ordinary overhead, so `elapsed > budget` stayed true and the mutation survived.
    assert len(spent) >= 2, "the nudge retry never ran, so nothing here is about the retry: %r" % (spent,)
    retry_budget = spent[1][1]
    assert retry_budget >= core._MIN_PROVIDER_BUDGET, (
        "the retry was handed %.4fs, below the floor: a nudge with no time is not a retry, it is a "
        "slower way of failing, and call()'s docstring promises the floor" % retry_budget)
    # And the providers after the one that overran must NOT have been invoked: the overrun is one
    # floor total, not one per provider.
    invoked = {n for n, _ in spent}
    assert invoked == {"p1"}, (
        "the budget leaked past the first provider: %r" % (sorted(invoked),))
