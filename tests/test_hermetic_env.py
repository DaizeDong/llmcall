# -*- coding: utf-8 -*-
"""Controls for the hermetic-environment fixture in conftest.py.

These live here, not in conftest.py, because pytest does not collect tests from conftest: a
control written there never runs, which is the same dead-check problem it exists to prevent.
"""
import os
import re

import llmcall.core as core
from steering_vars import STEERING_VARS


def test__conftest_actually_clears(monkeypatch):
    """This file is itself a control, so it has to be able to fail.

    If the fixture ever stops being autouse, or a new steering variable is added to core.py
    without being added above, this catches it rather than letting the suite quietly start
    reading the host machine again.
    """
    for name in STEERING_VARS:
        assert os.environ.get(name) is None, "%s leaked into the test session" % name

    src = open(core.__file__, encoding="utf-8").read()
    used = set(re.findall(r'os\.environ\.get\("(LLMCALL_[A-Z_]+)"', src))
    used |= set(re.findall(r'os\.getenv\("(LLMCALL_[A-Z_]+)"', src))
    missing = used - set(STEERING_VARS)
    assert not missing, (
        "core.py reads %s, which this conftest does not clear, so the host machine can still "
        "change test results silently" % sorted(missing))
