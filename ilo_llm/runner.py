"""Runner: a named run = fixed settings over a list of maps, n samples each. Resumable; nothing is sent twice.

A copy of a2_llm/runner.py (same cache design, retries, usage-limit handling, --workers, --until-done and
--export), with this module's prompt and schema. Every map, training or dev, is sent in test format with no
links. Answers live in cache/ilo_llm/answers/<key>.json, where key = a hash of the exact request (model,
effort, system prompt, user message, schema, sample number). Test maps are not supported (D090: dev only).

    python -m ilo_llm.runner --run=smoke            # send what is missing, then summarise
    python -m ilo_llm.runner --run=pilot-medium --export   # also results/ilo_llm/<run>/responses.jsonl.gz (committed)
    python -m ilo_llm.runner --run=dev --export --workers=2 --until-done
"""
import gzip
import hashlib
import json
import re
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from common import RESULTS, ROOT
from common.aif import load_map, strip_to_input, view
from common.serialise import serialise
from common.split import load_split
from common.timing import timed
from a2_llm import claude_cli
from a2_llm.prompt import Pool
from ilo_llm.prompt import example, system_file, user_message
from ilo_llm.schema import NONE, problems, schema

CACHE = ROOT / "cache" / "ilo_llm" / "answers"
OUT = RESULTS / "ilo_llm"
MAX_ATTEMPTS = 3
SMOKE = ["nodeset18266", "nodeset18274", "nodeset23907"]     # A2's smoke maps: 8, 15 and 28 propositions
RUNS = {
    "smoke": {"split": "train", "maps": SMOKE, "samples": 1, "k": 4, "effort": "medium", "prompt": "v0"},
    "pilot-medium": {"split": "train", "maps": "pilot", "exclude": "pilot", "samples": 1, "k": 4, "effort": "medium", "prompt": "v0"},
    "pilot-xhigh": {"split": "train", "maps": "pilot", "exclude": "pilot", "samples": 1, "k": 4, "effort": "xhigh", "prompt": "v0"},
    # Frozen after the pilot (D095, D096): effort medium (tie rule D094), prompt v0 (no revision).
    "dev": {"split": "dev", "maps": "dev225", "samples": 3, "k": 4, "effort": "medium", "prompt": "v0"},
}


def resolve(cfg) -> dict:
    """Named map lists: 'dev225' = the 225 dev maps (at most GIANT propositions; the labeller's dev maps too);
    'pilot' in maps / exclude = A2's 20-map pilot and its 210 excluded pool maps (results/a2/pilot_maps.json)."""
    if cfg.get("maps") == "dev225":
        from a2_llm.prompt import GIANT
        return {**cfg, "maps": [p.stem for p in load_split("dev") if len(view(load_map(p), p.stem).props) <= GIANT]}
    if cfg.get("maps") != "pilot" and cfg.get("exclude") != "pilot":     # (maps may be a list: no set here)
        return cfg
    pilot = json.loads((RESULTS / "a2" / "pilot_maps.json").read_text())
    maps = pilot["maps"] if cfg["maps"] == "pilot" else cfg["maps"]
    return {**cfg, "maps": maps, "exclude": pilot["pool_exclude"] if cfg.get("exclude") == "pilot" else cfg.get("exclude", [])}


def evaluate(res: dict, s) -> str | None:
    """None if the stored result is an acceptable answer, else why not. The answer must be Sonnet 5.5's:
    any other model producing more than a few side tokens (e.g. a session title) is rejected."""
    if res.get("_error") and not isinstance(res.get("structured_output"), dict):
        return res["_error"]                              # the call itself failed (stale verdicts on usable output are ignored)
    usage = res.get("modelUsage") or {}
    if not usage.get(claude_cli.MODEL, {}).get("outputTokens"):
        return f"not answered by {claude_cli.MODEL}: {sorted(usage)}"
    side = {m: u.get("outputTokens", 0) for m, u in usage.items() if m != claude_cli.MODEL}
    if any(n > 50 for n in side.values()):
        return f"other model wrote {side}"
    if not isinstance(res.get("structured_output"), dict):
        return "no structured_output"
    bad = problems(res["structured_output"], s)
    return f"schema: {bad[:3]}" if bad else None


def retryable(error: str) -> bool:
    return not error.startswith(("not answered by", "other model"))


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def targets(cfg) -> list[tuple[str, dict]]:
    """(name, test-format input). No links on any split: the model links the turns itself (D090)."""
    if cfg["split"] not in ("train", "dev"):
        raise ValueError(f"split {cfg['split']!r}: this experiment is dev only (D090)")
    paths = {p.stem: p for p in load_split(cfg["split"])}
    return [(n, strip_to_input(load_map(paths[n]))) for n in cfg["maps"]]


def jobs(cfg):
    """One job per (map, sample): everything needed to send it, plus its cache key."""
    cfg = resolve(cfg)
    system = system_file(cfg["prompt"]).read_text(encoding="utf-8")
    pool = Pool(set(cfg.get("exclude", ()))) if cfg["k"] else None
    for name, inp in targets(cfg):
        s = serialise(inp, name, [])
        ex = pool.nearest(inp, name, cfg["k"]) if pool else []        # never the map itself
        user = user_message(s, [example(e, pool.maps[e]) for e in ex])
        sch = schema(s)
        for i in range(cfg["samples"]):
            key = hashlib.sha256(json.dumps([claude_cli.MODEL, cfg["effort"], system, user, sch, i]).encode()).hexdigest()
            yield {"key": key, "map": name, "sample": i, "examples": ex, "s": s, "user": user, "schema": sch,
                   "chars": len(system) + len(user)}


def load(key, s=None):
    """The stored record; with `s`, re-judged: `ok` and `answer` (the first acceptable attempt) recomputed."""
    path = CACHE / f"{key}.json"
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if s is not None:
        errors = [evaluate(a["result"], s) for a in rec["attempts"]]
        rec["ok"] = None in errors
        rec["answer"] = errors.index(None) if rec["ok"] else None
        rec["errors"] = errors
    return rec


def save(rec):
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{rec['key']}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, ensure_ascii=False)


def pending(rec) -> bool:
    """Still worth a call: no acceptable answer, attempts left, and the last failure was transient."""
    if rec is None:
        return True
    if rec["ok"] or len(rec["attempts"]) >= MAX_ATTEMPTS:
        return False
    return retryable(rec["errors"][-1])


def send(job, cfg, run, cli):
    """Call until the answer is acceptable or no retry is warranted; the record is saved after every attempt."""
    rec = load(job["key"], job["s"]) or {"key": job["key"], "run": run, "map": job["map"], "sample": job["sample"],
                                         "settings": {k: cfg[k] for k in ("k", "effort", "prompt")},
                                         "examples": job["examples"], "chars": job["chars"], "attempts": []}
    while pending(rec if rec["attempts"] else None):
        t0 = time.time()
        res = claude_cli.call(system_file(cfg["prompt"]), job["user"], job["schema"], cfg["effort"])
        rec["attempts"].append({"when": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0, 1),
                                "cli": cli, "result": res})
        save({k: v for k, v in rec.items() if k not in ("ok", "answer", "errors")})
        rec = load(job["key"], job["s"])
        print(f"  {job['map']} s{job['sample']}: {'ok' if rec['ok'] else rec['errors'][-1][:120]} "
              f"({rec['attempts'][-1]['seconds']} s)", flush=True)
    return rec


def summary(recs: list[dict]) -> dict:
    """Run totals (every call made, failed ones included) and per-answer statistics (accepted answers)."""
    calls = [a for r in recs for a in r["attempts"]]
    ok = [r for r in recs if r["ok"]]
    used = [r["attempts"][r["answer"]] for r in ok]
    u = lambda a, f: (a["result"].get("usage") or {}).get(f, 0)
    total_in = [u(a, "input_tokens") + u(a, "cache_creation_input_tokens") + u(a, "cache_read_input_tokens") for a in used]
    thinking = [((a["result"].get("usage") or {}).get("output_tokens_details") or {}).get("thinking_tokens", 0) for a in used]
    side = {}
    for a in calls:
        for m, mu in (a["result"].get("modelUsage") or {}).items():
            if m != claude_cli.MODEL:
                side[m] = side.get(m, 0) + 1
    out = {"jobs": len(recs), "ok": len(ok), "failed": len(recs) - len(ok), "calls_made": len(calls),
           "api_equivalent_usd": round(sum(claude_cli.api_usd(a["result"]) for a in calls), 4),
           "claude_code_reported_usd": round(sum(a["result"].get("total_cost_usd", 0) for a in calls), 4),
           "calls_with_side_models": side}
    if used:
        med = lambda xs: st.median(xs)
        out["per_answer"] = {
            "seconds": {"p50": med([a["seconds"] for a in used]), "max": max(a["seconds"] for a in used)},
            "input_tokens_total": {"p50": med(total_in), "max": max(total_in)},
            "output_tokens": {"p50": med([u(a, "output_tokens") for a in used]), "max": max(u(a, "output_tokens") for a in used)},
            "thinking_tokens": {"p50": med(thinking), "max": max(thinking)},
            "cache_read_tokens_total": sum(u(a, "cache_read_input_tokens") for a in used),
            "num_turns": sorted({a["result"].get("num_turns") for a in used}),
            "chars_per_input_token": round(sum(r["chars"] for r in ok) / max(sum(total_in), 1), 2)}
        ill = lambda a: a["result"]["structured_output"]["illocutions"]
        out["answers"] = [{"map": r["map"], "sample": r["sample"], "attempt": r["answer"], "chars": r["chars"], "input_total": t,
                           "cache_read": u(a, "cache_read_input_tokens"), "output": u(a, "output_tokens"), "thinking": th,
                           "seconds": a["seconds"], "usd": claude_cli.api_usd(a["result"]),
                           "illocutions": len(ill(a)), "linked": sum(it["proposition"] != NONE for it in ill(a))}
                          for r, a, t, th in zip(ok, used, total_in, thinking)]
    return out


RESET = re.compile(r"resets\s+(?:(?P<mon>[A-Z][a-z]{2,8})\s+(?P<day>\d{1,2}),?\s+(?:at\s+)?)?"
                   r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>[ap]m)\s*(?:\((?P<tz>[^)]+)\))?", re.I)


def seconds_to_reset(message: str, margin: int = 300) -> float:
    """Seconds until the reset named in a usage-limit message ("resets 3:10am (Europe/London)", or with a date
    for a weekly limit), plus a margin. Unparseable messages give 30 minutes: the next try costs nothing."""
    m = RESET.search(message)
    if not m:
        return 1800.0
    tz = ZoneInfo(m["tz"]) if m["tz"] else ZoneInfo("Europe/London")
    now = datetime.now(tz)
    hour = int(m["h"]) % 12 + (12 if m["ap"].lower() == "pm" else 0)
    at = now.replace(hour=hour, minute=int(m["m"] or 0), second=0, microsecond=0)
    if m["mon"]:
        at = at.replace(month=datetime.strptime(m["mon"][:3], "%b").month, day=int(m["day"]))
        if at < now - timedelta(days=1):
            at = at.replace(year=at.year + 1)
    elif at <= now:
        at += timedelta(days=1)
    return max((at - now).total_seconds(), 0) + margin


def main():
    run = arg("run", "smoke")
    cfg = RUNS[run]
    if not cfg["effort"] or not cfg["prompt"]:
        sys.exit(f"run {run}: effort and prompt are not set yet (chosen on the pilot, step 4 of the brief)")
    cli = claude_cli.version()
    js = list(jobs(cfg))
    workers = int(arg("workers", 1))
    until_done = "--until-done" in sys.argv           # sleep through usage limits and carry on (D060)
    while True:
        todo = [j for j in js if pending(load(j["key"], j["s"]))]
        print(f"run {run}: {len(js)} jobs, {len(js) - len(todo)} settled, {len(todo)} to send, {workers} at a time "
              f"({cli}, {datetime.now().isoformat(timespec='minutes')})", flush=True)
        halt, limits = threading.Event(), []

        def work(job):
            if halt.is_set():                             # a usage limit was hit: send nothing more
                return
            try:
                send(job, cfg, run, cli)
            except claude_cli.UsageLimit as e:
                limits.append(str(e))
                halt.set()

        with timed("ilo_llm", f"run:{run}", jobs=len(js), to_send=len(todo), workers=workers) as row:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(work, todo))                # re-raises any other error
            stopped = limits[0] if limits else None
            recs = [r for r in (load(j["key"], j["s"]) for j in js) if r]
            rep = {"run": run, "settings": cfg, "cli": cli, "stopped_at_usage_limit": stopped, **summary(recs)}
            row.update({k: rep[k] for k in ("ok", "failed", "calls_made", "api_equivalent_usd")})
        (OUT / run).mkdir(parents=True, exist_ok=True)
        with open(OUT / run / "summary.json", "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rep, indent=1))
        if "--export" in sys.argv:
            with gzip.open(OUT / run / "responses.jsonl.gz", "wt", encoding="utf-8") as f:
                for r in sorted(recs, key=lambda r: (r["map"], r["sample"])):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not stopped or not until_done:
            break
        until = datetime.now() + timedelta(seconds=seconds_to_reset(stopped))
        print(f"Usage limit: {stopped}\nSleeping until {until.isoformat(timespec='minutes')}", flush=True)
        while datetime.now() < until:                     # wall clock, in short steps: a single long sleep
            time.sleep(60)                                # stretches while the PC (and so WSL) is suspended
    print(json.dumps(rep, indent=1))
    if stopped:
        print(f"\nStopped at the usage limit: {stopped}\nRe-run the same command to continue; answered jobs are kept.")
        sys.exit(2)


if __name__ == "__main__":
    main()
