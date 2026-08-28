# -*- coding: utf-8 -*-
"""Make the suite hermetic with respect to the developer's own environment.

Why this file exists. `active_chain()` reads LLMCALL_CHAIN at call time, deliberately, so a
provider can be routed around without restarting long-lived callers. The side effect is that a
machine which has that variable exported runs the tests against ITS chain instead of the built-in
ladder the tests assert on. Measured on a real machine with LLMCALL_CHAIN='cc,claude' exported at
user scope: 11 of 67 tests failed, none of them because of the code.

That is the worst shape a test suite can take: green in CI, red on the machine of the person
actually changing the code. It teaches you that a red local run means nothing, which is exactly
when a real regression walks past.

So every variable that steers behaviour is cleared for the whole session. A test that WANTS one of
them sets it itself with monkeypatch, which is also the only way a reader can tell that the
variable is part of what is being tested.
"""
import pytest

# Every environment variable core.py consults. Keep this list in step with the module: an entry
# missing here is a way for the host machine to change a test result without saying so.
STEERING_VARS = (
    "LLMCALL_CHAIN",          # which providers, in which order
    "LLMCALL_AGENT_RUNNER",   # the external agent runner for the cc/claude agentic path
    "LLMCALL_RELAY",          # notification relay
    "LLMCALL_LEDGER",         # ledger destination
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    for name in STEERING_VARS:
        monkeypatch.delenv(name, raising=False)
