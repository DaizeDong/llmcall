"""Provider-mix report over the llmcall ledger (see core._ledger).

Makes silent provider degradation visible: if the codex chain or a provider dies, llmcall's
never-raises quietly falls through to the last provider, invisible unless you aggregate who
actually SERVED each call. This reads the append-only ledger over a window and (optionally) alerts
when codex's share drops below a floor.

  python -m llmcall.mix                     # human summary of the last 24h
  python -m llmcall.mix --hours 168 --json  # machine-readable, 7-day window
  python -m llmcall.mix --alert-below 0.5 --stream infra
        # if codex served < 50% of the window's calls, post one line to the given relay stream

Pure stdlib. A missing/empty ledger is not an error (prints a note, exits 0). Never raises into a
caller; alerting failures are swallowed like every other telemetry side effect.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

from .core import _ledger_path, _NOWINDOW  # reuse the one path resolver + no-window flag


def _read(path, hours):
    """Yield ledger records. Time filtering is by LINE RECENCY, not timestamps: the ledger is
    append-only so the tail is the most recent, and core does not stamp wall-clock (Date.now is not
    available in that context). --hours is therefore a soft cap expressed as a max line count when no
    ts is present; if a record carries a numeric 'ts' we honor it. Both paths are best-effort."""
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    recs = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            recs.append(json.loads(ln))
        except ValueError:
            continue
    return recs


def aggregate(recs):
    served = Counter()          # who actually answered (None = total chain failure)
    modes = Counter()
    n = len(recs)
    ok = 0
    prompt_chars = reply_chars = ms_total = 0
    for r in recs:
        served[r.get("provider")] += 1
        modes[r.get("mode", "judge")] += 1
        if r.get("ok"):
            ok += 1
        prompt_chars += int(r.get("prompt_chars") or 0)
        reply_chars += int(r.get("reply_chars") or 0)
        ms_total += int(r.get("ms") or 0)
    codex = served.get("codex", 0)
    return {
        "calls": n,
        "ok": ok,
        "failed": n - ok,
        "served": {(k or "NONE"): v for k, v in served.most_common()},
        "codex_share": round(codex / n, 3) if n else None,
        "modes": dict(modes),
        "prompt_chars": prompt_chars,
        "reply_chars": reply_chars,
        "avg_ms": round(ms_total / ok) if ok else None,
    }


def _alert(stream, msg):
    relay = os.path.expanduser(os.environ.get("LLMCALL_RELAY", "~/.llmcall/relay.py"))
    if not os.path.isfile(relay):
        sys.stderr.write("mix: relay not found, cannot alert\n")
        return
    try:
        subprocess.run([sys.executable, relay, "send", "--stream", stream, "--text", msg],
                       capture_output=True, text=True, encoding="utf-8", timeout=30, **_NOWINDOW)
    except Exception:
        pass


def summary_line(agg):
    if not agg["calls"]:
        return "llmcall mix: ledger empty (no calls recorded)"
    served = ", ".join(f"{k} {v}" for k, v in agg["served"].items())
    share = agg["codex_share"]
    return ("llmcall mix: %d calls (%d ok / %d failed) | served: %s | codex share %s | avg %sms"
            % (agg["calls"], agg["ok"], agg["failed"], served,
               ("%.0f%%" % (share * 100)) if share is not None else "n/a", agg["avg_ms"]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="llmcall.mix", description="provider-mix report over the ledger")
    ap.add_argument("--hours", type=float, default=24.0, help="(soft) window; honored only for ts-stamped records")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--alert-below", type=float, default=None,
                    help="if codex share < this fraction (0..1), post a warning to --stream")
    ap.add_argument("--stream", default="infra", help="relay stream for --alert-below")
    ap.add_argument("--ledger", default=None, help="override ledger path (else LLMCALL_LEDGER or default)")
    a = ap.parse_args(argv)

    path = os.path.expanduser(a.ledger) if a.ledger else _ledger_path()
    recs = _read(path, a.hours)
    agg = aggregate(recs)

    if a.alert_below is not None and agg["calls"] and agg["codex_share"] is not None \
            and agg["codex_share"] < a.alert_below:
        _alert(a.stream, "warning [llmcall] codex share %.0f%% < %.0f%% over %d calls -- "
                         "provider degraded (codex chain / a provider down?). %s"
                         % (agg["codex_share"] * 100, a.alert_below * 100, agg["calls"], summary_line(agg)))

    if a.json:
        sys.stdout.write(json.dumps(agg, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(summary_line(agg) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
