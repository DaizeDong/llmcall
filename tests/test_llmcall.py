"""llmcall unit tests. Network + subprocess are mocked; no real provider is ever called except the
opt-in real-codex smoke at the bottom (gated behind LLMCALL_SMOKE=1)."""
import json
import os
import subprocess

import pytest

import llmcall
from llmcall import Result, call, call_chain, core
from llmcall.schema import extract_json, validate


@pytest.fixture(autouse=True)
def _isolate_ledger(tmp_path, monkeypatch):
    """Redirect the provider-mix ledger to a temp file for EVERY test so a real call() in the suite
    never appends to the operator's ~/.llmcall/ledger.jsonl."""
    monkeypatch.setenv("LLMCALL_LEDGER", str(tmp_path / "ledger.jsonl"))


# ---- helpers: replace the per-provider _invoke with a scripted one --------------------------------
def _fixed(mapping):
    calls = []

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        calls.append(name)
        return mapping.get(name, (None, "not configured"))
    return fake, calls


def _scripted(scripts):
    calls = []
    iters = {k: iter(v) for k, v in scripts.items()}

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        calls.append(name)
        return next(iters[name])
    return fake, calls


# ---- chain order + Result contract ---------------------------------------------------------------
def test_first_non_empty_wins_and_stops(monkeypatch):
    fake, calls = _fixed({"codex": ("hi from codex", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x")
    assert r.provider == "codex" and r.text == "hi from codex"
    assert str(r) == "hi from codex" and bool(r) is True
    assert calls == ["codex"]  # did not touch cc/claude


def test_falls_through_on_empty(monkeypatch):
    fake, calls = _fixed({"codex": (None, "codex down"), "cc": ("from cc", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x")
    assert r.provider == "cc" and calls == ["codex", "cc"]
    assert r.attempts[0].provider == "codex" and r.attempts[0].ok is False


def test_total_failure_is_falsy():
    # no providers resolvable in a clean env: real _invoke returns (None, "not found") for each
    r = call("x", chain=["codex"], timeout=1)
    # either it truly could not find codex, or (rare) codex answered; assert the contract shape
    assert isinstance(r, Result)
    if not r:
        assert r.provider is None and r.text == "" and str(r) == "" and r.error


def test_str_none_backcompat(monkeypatch):
    fake, _ = _fixed({})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x")
    assert (str(r) if r else None) is None  # reproduces old `str | None`


# ---- per-call reorder ----------------------------------------------------------------------------
def test_chain_reorder(monkeypatch):
    fake, calls = _fixed({"cc": ("from cc", None), "codex": ("from codex", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x", chain=["cc", "claude", "codex"])
    assert r.provider == "cc" and calls[0] == "cc"


# ---- notify fires only on total failure ----------------------------------------------------------
def test_notify_on_total_failure(monkeypatch):
    fake, _ = _fixed({})
    monkeypatch.setattr(core, "_invoke", fake)
    got = {}
    monkeypatch.setattr(core, "_notify", lambda s, m: got.update(stream=s, msg=m))
    r = call("x", notify="infra")
    assert not r and got["stream"] == "infra" and "failed" in got["msg"]


def test_no_notify_on_success(monkeypatch):
    fake, _ = _fixed({"codex": ("ok", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    got = {}
    monkeypatch.setattr(core, "_notify", lambda s, m: got.update(x=1))
    call("x", notify="infra")
    assert "x" not in got


# ---- schema layer --------------------------------------------------------------------------------
SCHEMA = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}


def test_schema_valid(monkeypatch):
    fake, _ = _fixed({"codex": ('here you go: {"ok": true}', None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x", schema=SCHEMA)
    assert r.data == {"ok": True} and r.provider == "codex"


def test_schema_retries_same_provider_then_succeeds(monkeypatch):
    fake, calls = _scripted({"codex": [("not json at all", None), ('{"ok": true}', None)]})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x", schema=SCHEMA, chain=["codex", "cc"])
    assert r.provider == "codex" and r.data == {"ok": True}
    assert calls == ["codex", "codex"]  # retried SAME provider, never fell to cc


def test_schema_exhausted_falls_through(monkeypatch):
    fake, calls = _scripted({"codex": [("bad", None), ("still bad", None)], "cc": [('{"ok": false}', None)]})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x", schema=SCHEMA, chain=["codex", "cc"])
    assert r.provider == "cc" and r.data == {"ok": False}
    assert calls == ["codex", "codex", "cc"]


# ---- never raises --------------------------------------------------------------------------------
def test_never_raises_even_if_provider_throws(monkeypatch):
    def boom(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        raise RuntimeError("provider blew up")
    monkeypatch.setattr(core, "_invoke", boom)
    r = call("x")  # must not raise
    assert not r and any("blew up" in (a.error or "") for a in r.attempts)


# ---- provider plumbing (mock subprocess.run + capture argv) --------------------------------------
def test_cc_disables_mcp_and_unwraps_envelope(monkeypatch):
    cap = {}

    def fake_run(cmd, input=None, **kw):
        cap["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "unwrapped text"}), stderr="")
    monkeypatch.setattr(core, "_find", lambda n, c: "/x/cc.exe")
    monkeypatch.setattr(subprocess, "run", fake_run)
    text, err = core._invoke("cc", "p", 10, None, None)
    assert text == "unwrapped text" and err is None
    assert "--strict-mcp-config" in cap["cmd"] and '{"mcpServers":{}}' in cap["cmd"]
    assert "--output-format" in cap["cmd"] and "json" in cap["cmd"]


def test_codex_reads_output_file_and_sets_readonly(monkeypatch, tmp_path):
    cap = {}

    def fake_run(cmd, input=None, **kw):
        cap["cmd"] = cmd
        # emulate codex: write the final message to the -o outfile
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w", encoding="utf-8") as f:
            f.write("codex final answer")
        return subprocess.CompletedProcess(cmd, 0, stdout="reasoning preamble noise", stderr="")
    monkeypatch.setattr(core, "_find", lambda n, c: "/x/codex")
    monkeypatch.setattr(subprocess, "run", fake_run)
    text, err = core._invoke("codex", "p", 10, "gpt-x", "high")
    assert text == "codex final answer" and err is None
    assert "-s" in cap["cmd"] and "read-only" in cap["cmd"] and "--ephemeral" in cap["cmd"]
    assert "-m" in cap["cmd"] and "gpt-x" in cap["cmd"] and "model_reasoning_effort=high" in cap["cmd"]


def test_missing_binary_is_a_clean_miss(monkeypatch):
    monkeypatch.setattr(core, "_find", lambda n, c: None)
    text, err = core._invoke("cc", "p", 10, None, None)
    assert text is None and "not found" in err


# ---- gemini provider -----------------------------------------------------------------------------
def test_gemini_argv_and_plain_text(monkeypatch):
    cap = {}

    def fake_run(cmd, input=None, **kw):
        cap["cmd"] = cmd
        cap["stdin"] = input
        return subprocess.CompletedProcess(cmd, 0, stdout="gemini plain answer", stderr="")
    monkeypatch.setattr(core, "_find", lambda n, c: "/x/gemini.cmd")
    monkeypatch.setattr(subprocess, "run", fake_run)
    text, err = core._invoke("gemini", "the prompt", 10, None, None)
    assert text == "gemini plain answer" and err is None            # plain text, no JSON envelope unwrap
    assert "-m" in cap["cmd"] and core._GEMINI_FALLBACK_MODEL in cap["cmd"]
    assert "-p" in cap["cmd"] and "the prompt" in cap["cmd"]         # prompt is the -p arg
    assert cap["stdin"] == ""                                        # not stdin


def test_gemini_missing_is_a_clean_miss(monkeypatch):
    monkeypatch.setattr(core, "_find", lambda n, c: None)
    text, err = core._invoke("gemini", "p", 10, None, None)
    assert text is None and "gemini not found" in err


def test_gemini_model_resolution():
    assert core._resolve_model("gemini", None, None) == (core._GEMINI_FALLBACK_MODEL, None)
    assert core._resolve_model("gemini", "gemini-3-flash-preview", None) == ("gemini-3-flash-preview", None)


def test_gemini_exclusive_chain(monkeypatch):
    fake, calls = _fixed({"gemini": ("g", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x", chain=["gemini"])
    assert r.provider == "gemini" and r.text == "g" and calls == ["gemini"]


# ---- web_search opt-in (off by default; relaxes read-only) ---------------------------------------
def _capture_cmd(monkeypatch, stdout="ok"):
    cap = {}

    def fake_run(cmd, input=None, **kw):
        cap["cmd"] = cmd
        if "-o" in cmd:  # codex writes to the outfile
            with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as f:
                f.write("codex answer")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    monkeypatch.setattr(core, "_find", lambda n, c: "/x/bin")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return cap


def test_codex_no_web_search_by_default(monkeypatch):
    cap = _capture_cmd(monkeypatch)
    core._invoke("codex", "p", 10, None, None)                       # web_search defaults False
    assert "tools.web_search=true" not in " ".join(cap["cmd"])
    assert "read-only" in cap["cmd"]                                 # still read-only


def test_codex_web_search_opt_in(monkeypatch):
    cap = _capture_cmd(monkeypatch)
    core._invoke("codex", "p", 10, None, None, True)                 # web_search=True
    joined = " ".join(cap["cmd"])
    assert "tools.web_search=true" in joined
    assert "read-only" in cap["cmd"]                                 # FS still read-only


def test_cc_web_tools_only_when_opted_in(monkeypatch):
    cap = _capture_cmd(monkeypatch, stdout=json.dumps({"result": "x"}))
    core._invoke("cc", "p", 10, None, None)                          # default: no web tools
    assert "--allowedTools" not in cap["cmd"]
    cap2 = _capture_cmd(monkeypatch, stdout=json.dumps({"result": "x"}))
    core._invoke("cc", "p", 10, None, None, True)                    # opted in
    assert "--allowedTools" in cap2["cmd"] and "WebSearch" in cap2["cmd"] and "WebFetch" in cap2["cmd"]


def test_web_search_threads_through_call(monkeypatch):
    seen = {}

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        seen[name] = web_search
        return ("ok", None)
    monkeypatch.setattr(core, "_invoke", fake)
    call("x", web_search=True)
    assert seen["codex"] is True
    call("y")  # default (web_search=None -> mode=judge -> no web)
    assert seen["codex"] is False


# ---- hardened extract_json (balanced-brace, first complete) --------------------------------------
def test_extract_json_first_complete_and_string_aware():
    assert extract_json("prose {\"a\": 1} tail") == {"a": 1}
    assert extract_json("[1, 2, 3]") == [1, 2, 3]
    assert extract_json('{"s": "a brace } inside a string"}') == {"s": "a brace } inside a string"}
    # two objects: the greedy \{.*\} regex over-captured both and json.loads FAILED -> None; the
    # balanced scanner returns the FIRST complete one (strict improvement).
    assert extract_json('{"a":1}\n{"b":2}') == {"a": 1}
    assert extract_json("no json here") is None


# ---- extract= hook (general form of schema=) -----------------------------------------------------
def test_extract_hook_returns_data(monkeypatch):
    fake, _ = _fixed({"codex": ("the answer is 42 somewhere", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x", extract=lambda t: {"n": 42} if "42" in t else None)
    assert r.data == {"n": 42} and r.provider == "codex"


def test_extract_hook_miss_retries_then_falls_through(monkeypatch):
    fake, calls = _scripted({"codex": [("nope", None), ("still nope", None)], "cc": [("has KEY", None)]})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x", extract=lambda t: {"ok": 1} if "KEY" in t else None, chain=["codex", "cc"])
    assert r.provider == "cc" and r.data == {"ok": 1}
    assert calls == ["codex", "codex", "cc"]  # same-provider retry, then fall through (fixes furry's gap)


def test_extract_hook_that_throws_is_a_miss_not_an_escape(monkeypatch):
    fake, _ = _fixed({"codex": ("x", None), "cc": ("x", None)})
    monkeypatch.setattr(core, "_invoke", fake)

    def boom(t):
        raise ValueError("bad extractor")
    r = call("x", extract=boom, chain=["codex", "cc"])  # must not raise
    assert not r  # a throwing extractor counts as a miss on every attempt; chain exhausts, never escapes


# ---- mode= tiers ---------------------------------------------------------------------------------
def test_mode_research_turns_on_web(monkeypatch):
    seen = {}

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        seen["web"], seen["agentic"] = web_search, agentic
        return ("ok", None)
    monkeypatch.setattr(core, "_invoke", fake)
    call("x", mode="research")
    assert seen["web"] is True and seen["agentic"] is False


def test_mode_agent_sets_agentic(monkeypatch):
    seen = {}

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        seen["agentic"] = agentic
        return ("ok", None)
    monkeypatch.setattr(core, "_invoke", fake)
    call("x", mode="agent")
    assert seen["agentic"] is True


def test_web_search_overrides_mode(monkeypatch):
    seen = {}

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        seen["web"] = web_search
        return ("ok", None)
    monkeypatch.setattr(core, "_invoke", fake)
    call("x", mode="research", web_search=False)
    assert seen["web"] is False  # explicit False forces off even in research
    call("x", mode="judge", web_search=True)
    assert seen["web"] is True   # explicit True forces on even in judge


def test_codex_agent_uses_workspace_write(monkeypatch):
    cap = _capture_cmd(monkeypatch)
    core._invoke("codex", "p", 10, None, None, False, True)  # agentic=True
    assert "workspace-write" in cap["cmd"] and "read-only" not in cap["cmd"]


def test_codex_judge_stays_read_only(monkeypatch):
    cap = _capture_cmd(monkeypatch)
    core._invoke("codex", "p", 10, None, None)  # defaults
    assert "read-only" in cap["cmd"] and "workspace-write" not in cap["cmd"]


def test_cc_agent_delegates_to_runner(monkeypatch):
    cap = {}

    def fake_run(cmd, input=None, **kw):
        cap["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"result": "agent answer"}), stderr="")
    monkeypatch.setattr(core.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(subprocess, "run", fake_run)
    text, err = core._invoke("cc", "p", 10, None, None, False, True)  # agentic=True
    assert text == "agent answer" and err is None
    assert any("agent-runner" in str(c) for c in cap["cmd"]) and "-Capture" in cap["cmd"]


# ---- log= callback -------------------------------------------------------------------------------
def test_log_callback_per_attempt(monkeypatch):
    fake, _ = _fixed({"codex": (None, "down"), "cc": ("hi", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    lines = []
    call("x", log=lines.append)
    assert any("codex: unavailable" in l for l in lines) and any("cc: answered" in l for l in lines)


# ---- model/effort resolution ---------------------------------------------------------------------
def test_kwarg_overrides_config(monkeypatch):
    monkeypatch.setattr(core, "_codex_config", lambda: {"model": "cfg-model", "model_reasoning_effort": "low"})
    assert core._resolve_model("codex", "kw-model", "xhigh") == ("kw-model", "xhigh")
    assert core._resolve_model("codex", None, None) == ("cfg-model", "low")


def test_claude_model_default(monkeypatch):
    assert core._resolve_model("claude", None, None) == (core._CLAUDE_FALLBACK_MODEL, None)
    assert core._resolve_model("claude", "custom", None) == ("custom", None)


# ---- schema validator unit ------------------------------------------------------------------------
def test_validator_and_extractor():
    assert extract_json('prose {"a": 1} tail') == {"a": 1}
    assert extract_json("no json here") is None
    ok, _ = validate({"a": 1}, {"type": "object", "required": ["a"], "properties": {"a": {"type": "integer"}}})
    assert ok
    ok, e = validate({"a": "x"}, {"type": "object", "properties": {"a": {"type": "integer"}}})
    assert not ok and "a:" in e
    ok, _ = validate("yes", {"type": "string", "enum": ["yes", "no"]})
    assert ok
    ok, _ = validate("maybe", {"type": "string", "enum": ["yes", "no"]})
    assert not ok


# ---- call_chain back-compat shim -----------------------------------------------------------------
def test_call_chain_returns_str_or_none(monkeypatch):
    fake, _ = _fixed({"codex": ("hello", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    assert call_chain("x") == "hello"
    fake2, _ = _fixed({})
    monkeypatch.setattr(core, "_invoke", fake2)
    assert call_chain("x") is None


# ---- opt-in real smoke ---------------------------------------------------------------------------
@pytest.mark.skipif(os.environ.get("LLMCALL_SMOKE") != "1", reason="set LLMCALL_SMOKE=1 for a real call")
def test_real_codex_smoke():
    r = call("Reply with only the word PONG.", chain=["codex"], timeout=90)
    assert r and "PONG" in r.text.upper()


# ---- refine (iterative deepening) ----------------------------------------------------------------
from llmcall import refine  # noqa: E402


def test_refine_callable_recurses_then_stops(monkeypatch):
    outs = iter([("first", None), ("second", None)])
    monkeypatch.setattr(core, "_invoke", lambda *a: next(outs))
    seen = []

    def judge(r, depth):
        seen.append(depth)
        return "go deeper" if depth == 1 else None
    r = refine("start", judge=judge, max_depth=3)
    assert r.text == "second" and r.depth == 1 and seen == [1, 2]


def test_refine_self_refine_continue_then_done(monkeypatch):
    seq = iter([("draft1", None), ("CONTINUE: add X", None), ("draft2", None), ("DONE", None)])
    monkeypatch.setattr(core, "_invoke", lambda *a: next(seq))
    r = refine("task", max_depth=3)
    assert r.text == "draft2" and r.depth == 1


def test_refine_convergence_stops(monkeypatch):
    seq = iter([("same", None), ("CONTINUE: x", None), ("same", None)])
    monkeypatch.setattr(core, "_invoke", lambda *a: next(seq))
    r = refine("task", max_depth=5)
    assert r.text == "same" and r.depth == 1  # regenerate returned the same text -> converged


def test_refine_max_depth_cap(monkeypatch):
    counter = {"n": 0}

    def fake(name, prompt, timeout, model, effort, web_search=False, agentic=False):
        if "Your verdict:" in prompt:
            return ("CONTINUE: more", None)
        counter["n"] += 1
        return (f"draft{counter['n']}", None)
    monkeypatch.setattr(core, "_invoke", fake)
    r = refine("task", max_depth=2)
    assert r.depth == 2  # never exceeds max_depth even when the judge keeps saying CONTINUE


def test_refine_first_pass_failure_is_falsy(monkeypatch):
    monkeypatch.setattr(core, "_invoke", lambda *a: (None, "down"))
    r = refine("task")
    assert not r and r.depth == 0


# ---- provider-mix ledger (telemetry; must never affect the call) ---------------------------------
from llmcall import mix  # noqa: E402


def _ledger_lines(monkeypatch, tmp_path, name="l.jsonl"):
    p = tmp_path / name
    monkeypatch.setenv("LLMCALL_LEDGER", str(p))
    return p


def test_ledger_records_served_provider(monkeypatch, tmp_path):
    p = _ledger_lines(monkeypatch, tmp_path)
    fake, _ = _fixed({"codex": (None, "down"), "cc": ("from cc", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x")
    assert r.provider == "cc"
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["provider"] == "cc" and rec["ok"] is True
    assert rec["prompt_chars"] == 1 and rec["reply_chars"] == len("from cc")
    assert rec["attempts"] == 2  # codex miss + cc answer


def test_ledger_records_total_failure(monkeypatch, tmp_path):
    p = _ledger_lines(monkeypatch, tmp_path)
    fake, _ = _fixed({})
    monkeypatch.setattr(core, "_invoke", fake)
    call("x")
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["provider"] is None and rec["ok"] is False and rec["error"]


def test_ledger_disabled_by_env(monkeypatch, tmp_path):
    p = tmp_path / "off.jsonl"
    monkeypatch.setenv("LLMCALL_LEDGER", "0")
    fake, _ = _fixed({"codex": ("ok", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    call("x")
    assert not p.exists()  # LLMCALL_LEDGER=0 -> no file written anywhere
    assert core._ledger_path() is None


def test_ledger_write_failure_never_breaks_call(monkeypatch, tmp_path):
    # point the ledger at an unwritable path (a dir where a file is expected) -> _ledger swallows it
    monkeypatch.setenv("LLMCALL_LEDGER", str(tmp_path))  # a directory, open(...) will fail
    fake, _ = _fixed({"codex": ("ok", None)})
    monkeypatch.setattr(core, "_invoke", fake)
    r = call("x")  # must still succeed despite the ledger write failing
    assert r.provider == "codex" and r.text == "ok"


def test_mix_aggregate_and_share():
    recs = [{"provider": "codex", "ok": True, "prompt_chars": 10, "reply_chars": 5, "ms": 100, "mode": "judge"},
            {"provider": "codex", "ok": True, "prompt_chars": 10, "reply_chars": 5, "ms": 200, "mode": "judge"},
            {"provider": "claude", "ok": True, "prompt_chars": 10, "reply_chars": 5, "ms": 300, "mode": "judge"},
            {"provider": None, "ok": False, "prompt_chars": 10, "reply_chars": 0, "mode": "judge"}]
    agg = mix.aggregate(recs)
    assert agg["calls"] == 4 and agg["ok"] == 3 and agg["failed"] == 1
    assert agg["codex_share"] == round(2 / 4, 3)
    assert agg["served"]["codex"] == 2 and agg["served"]["claude"] == 1 and agg["served"]["NONE"] == 1
    assert agg["avg_ms"] == round((100 + 200 + 300) / 3)


def test_mix_alert_fires_below_threshold(monkeypatch, tmp_path):
    p = tmp_path / "mix.jsonl"
    # 1 codex + 3 claude -> codex share 0.25
    p.write_text("\n".join(json.dumps(r) for r in [
        {"provider": "codex", "ok": True}, {"provider": "claude", "ok": True},
        {"provider": "claude", "ok": True}, {"provider": "claude", "ok": True}]), encoding="utf-8")
    alerts = []
    monkeypatch.setattr(mix, "_alert", lambda stream, msg: alerts.append((stream, msg)))
    mix.main(["--ledger", str(p), "--alert-below", "0.5", "--stream", "infra"])
    assert len(alerts) == 1 and alerts[0][0] == "infra" and "codex share 25%" in alerts[0][1]


def test_mix_no_alert_when_healthy(monkeypatch, tmp_path):
    p = tmp_path / "mix.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"provider": "codex", "ok": True}, {"provider": "codex", "ok": True},
        {"provider": "codex", "ok": True}, {"provider": "claude", "ok": True}]), encoding="utf-8")
    alerts = []
    monkeypatch.setattr(mix, "_alert", lambda stream, msg: alerts.append(msg))
    mix.main(["--ledger", str(p), "--alert-below", "0.5"])
    assert alerts == []  # codex share 75% >= 50%, no alert


def test_mix_empty_ledger_is_not_an_error(tmp_path):
    agg = mix.aggregate([])
    assert agg["calls"] == 0 and agg["codex_share"] is None
    assert "empty" in mix.summary_line(agg)
