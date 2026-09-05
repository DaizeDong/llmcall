# -*- coding: utf-8 -*-
"""The default model for each provider must be an ALIAS, never a pinned version id.

WHY THIS TEST EXISTS. The Claude default was pinned to claude-opus-4-8. It kept working, which is
the problem: both a pin and an alias resolve and answer, so the only way to notice the fleet had
fallen a model behind was to read modelUsage out of the CLI's JSON envelope. Measured 2026-09-05,
--model opus resolved to claude-opus-5 while the pinned id resolved to claude-opus-4-8, and no
caller anywhere passes model= explicitly, so every headless call that fell through to a Claude
provider ran the older one.

A pin does not fail. It quietly stops keeping up, and that is exactly the failure mode this fleet
keeps having to rediscover.
"""
import re

from llmcall.core import _resolve_model

# A version-pinned id looks like a family name followed by digits: claude-opus-4-8, gpt-5.6-sol,
# gemini-3-pro-preview. An alias is a bare family word: opus, sonnet, fable.
_PINNED = re.compile(r"[0-9]")


def test_claude_default_is_an_alias_not_a_pin():
    model, _ = _resolve_model("claude", None, None)
    assert not _PINNED.search(model), (
        "the Claude default is pinned to %r. Use the CLI alias (opus / sonnet) so it follows the "
        "model line; a pin keeps answering while silently running an older model, which is how "
        "this went unnoticed." % model)


def test_cc_default_is_an_alias_not_a_pin():
    model, _ = _resolve_model("cc", None, None)
    assert not _PINNED.search(model), (
        "the cc default is pinned to %r; see the note in test_claude_default_is_an_alias_not_a_pin" % model)


def test_an_explicit_model_still_wins():
    """The alias is a DEFAULT, not a policy. A caller that names a model gets it.

    Without this, a fix that hardcoded the alias everywhere would pass the two tests above while
    removing the ability to pin deliberately, which some callers legitimately need.
    """
    model, _ = _resolve_model("claude", "claude-opus-4-8", None)
    assert model == "claude-opus-4-8"
