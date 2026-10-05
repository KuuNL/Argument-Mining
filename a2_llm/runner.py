"""A2 runner: a named run = fixed settings over a list of maps, n samples each. Resumable; nothing is sent twice.

Answers live in a content-addressed cache, cache/a2/answers/<key>.json, where key = a hash of the exact
request (model, effort, system prompt, user message, schema, sample number). Any answer already there is
reused, so a stopped run resumes exactly. Whether an answer is acceptable is judged from the stored result
(`evaluate`), so a corrected rule re-judges old answers without new calls. Only transient failures (no or
unusable output, an answer that breaks the schema) are retried, up to MAX_ATTEMPTS calls; every attempt is
kept. At a usage limit the runner stops and says so: re-run the same command to continue (D052).

    python -m a2_llm.runner --run=smoke            # send what is missing, then summarise
    python -m a2_llm.runner --run=smoke --export   # also results/a2/<run>/responses.jsonl.gz (committed, D052)
    python -m a2_llm.runner --run=dev-v1 --export --workers=2   # 2 calls at a time (each answer is its own file)
    ... --until-done      # at a usage limit, sleep until the stated reset (+5 min) and carry on (D060)
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
from common.aif import load_map, strip_to_input, test_paths, view
from common.serialise import serialise
from common.split import load_split
from common.timing import timed
from a2_llm import claude_cli
from a2_llm.prompt import Pool, system_file, user_message
from a2_llm.schema import problems, schema

CACHE = ROOT / "cache" / "a2" / "answers"
OUT = RESULTS / "a2"
MAX_ATTEMPTS = 3
SMOKE = ["nodeset18266", "nodeset18274", "nodeset23907"]     # 8, 15 and 28 propositions; the last two have a Conflict
RUNS = {
    "smoke": {"split": "train", "maps": SMOKE, "samples": 1, "k": 2, "effort": "medium", "prompt": "v0"},
    "smoke-repeat": {"split": "train", "maps": SMOKE[:1], "samples": 2, "k": 2, "effort": "medium", "prompt": "v0"},
    "smoke-high": {"split": "train", "maps": SMOKE[1:2], "samples": 1, "k": 2, "effort": "high", "prompt": "v0"},
    "smoke-verify": {"split": "train", "maps": SMOKE[2:], "samples": 2, "k": 2, "effort": "medium", "prompt": "v0"},
    # Reasoning levers (D056): does anything make Sonnet 5.5 think before answering? Baseline = "smoke" (0 thinking).
    "reason-think": {"split": "train", "maps": SMOKE, "samples": 1, "k": 2, "effort": "medium", "prompt": "v0think"},
    "reason-ultra": {"split": "train", "maps": SMOKE, "samples": 1, "k": 2, "effort": "medium", "prompt": "v0",
                     "user_suffix": "\n\nultrathink"},
    "reason-xhigh": {"split": "train", "maps": SMOKE, "samples": 1, "k": 2, "effort": "xhigh", "prompt": "v0"},
    # A2.2 sweep A: worked examples k on the 50 pilot maps, 1 sample, effort medium (no thinking, D056).
    **{f"pilot-k{k}": {"split": "train", "maps": "pilot50", "exclude": "pilot", "samples": 1, "k": k,
                       "effort": "medium", "prompt": "v0"} for k in (0, 2, 4, 8)},
    # A2.2 sweep B on the 20-map pilot (D057): k = 4 (sweep A) at medium (answers reused) vs xhigh (thinks, D056).
    "pilot20-k4": {"split": "train", "maps": "pilot", "exclude": "pilot", "samples": 1, "k": 4, "effort": "medium", "prompt": "v0"},
    "pilot20-k4-xhigh": {"split": "train", "maps": "pilot", "exclude": "pilot", "samples": 1, "k": 4, "effort": "xhigh", "prompt": "v0"},
    # A2.2 revision 1: sharper Rephrase / Inference boundary and the measured sparsity (prompts/system_v1.md).
    "pilot20-k4-xhigh-v1": {"split": "train", "maps": "pilot", "exclude": "pilot", "samples": 1, "k": 4, "effort": "xhigh", "prompt": "v1"},
    # Frozen configuration (D059): prompt v1, k = 4, effort xhigh, 3 samples. The pool is all test-like training maps.
    "dev-v1": {"split": "dev", "maps": "dev225", "samples": 3, "k": 4, "effort": "xhigh", "prompt": "v1"},
    "test-v1": {"split": "test", "maps": "test11", "samples": 3, "k": 4, "effort": "xhigh", "prompt": "v1"},
    # Reasoning ablation (D053, D056): the effort not chosen (medium: no thinking), 1 sample; vs sample 0 of dev-v1.
    "dev-medium": {"split": "dev", "maps": "dev225", "samples": 1, "k": 4, "effort": "medium", "prompt": "v1"},
}


def resolve(cfg) -> dict:
    """Named map lists: 'dev225' = the 225 dev maps A1 also uses (at most GIANT propositions), 'test11' = the
    test inputs; 'pilot' / 'pilot50' in maps, and 'pilot' in exclude, mean the lists in results/a2/pilot_maps.json
    (a2_llm.pilot --choose): the 20-map pilot (D057), its 50-map parent (sweep A), the excluded pool maps."""
    if cfg.get("maps") == "dev225":
        from a2_llm.prompt import GIANT
        return {**cfg, "maps": [p.stem for p in load_split("dev") if len(view(load_map(p), p.stem).props) <= GIANT]}
    if cfg.get("maps") == "test11":
        return {**cfg, "maps": [name for name, _, _ in test_paths()]}
    if not {"pilot", "pilot50"} & {cfg.get("maps"), cfg.get("exclude")}:
        return cfg
    pilot = json.loads((OUT / "pilot_maps.json").read_text())
    maps = {"pilot": pilot["maps"], "pilot50": pilot["maps_50"]}.get(cfg["maps"], cfg["maps"])
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


def targets(cfg) -> list[tuple[str, dict, list]]:
    """(name, test-format input, links): gold links on training maps, the labeller's on dev and test (D040)."""
    split = cfg["split"]
    if split == "test":
        links = json.loads((ROOT / "cache" / "ilo" / "links_test.json").read_text())
        maps = {n: load_map(p) for n, p, _ in test_paths()}
        return [(n, maps[n], links.get(n, [])) for n in cfg["maps"]]
    paths = {p.stem: p for p in load_split(split)}
    links = json.loads((ROOT / "cache" / "ilo" / "links_dev.json").read_text()) if split == "dev" else {}
    out = []
    for n in cfg["maps"]:
        gold = load_map(paths[n])
        out.append((n, strip_to_input(gold), view(gold, n).anchors() if split == "train" else links.get(n, [])))
    return out


def jobs(cfg):
    """One job per (map, sample): everything needed to send it, plus its cache key."""
    cfg = resolve(cfg)
    system = system_file(cfg["prompt"]).read_text(encoding="utf-8")
    pool = Pool(set(cfg.get("exclude", ()))) if cfg["k"] else None
    for name, inp, links in targets(cfg):
        s = serialise(inp, name, links)
        ex = pool.nearest(inp, name, cfg["k"]) if pool else []
        user = user_message(s, [pool.example(e) for e in ex]) + cfg.get("user_suffix", "")
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
        out["answers"] = [{"map": r["map"], "sample": r["sample"], "attempt": r["answer"], "chars": r["chars"], "input_total": t,
                           "cache_read": u(a, "cache_read_input_tokens"), "output": u(a, "output_tokens"), "thinking": th,
                           "seconds": a["seconds"], "usd": claude_cli.api_usd(a["result"]),
                           "relations": len(a["result"]["structured_output"]["relations"])}
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

        with timed("a2", f"run:{run}", jobs=len(js), to_send=len(todo), workers=workers) as row:
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
        wait = seconds_to_reset(stopped)
        print(f"Usage limit: {stopped}\nSleeping {wait / 60:.0f} min, until {(datetime.now() + timedelta(seconds=wait)).isoformat(timespec='minutes')}",
              flush=True)
        time.sleep(wait)
    print(json.dumps(rep, indent=1))
    if stopped:
        print(f"\nStopped at the usage limit: {stopped}\nRe-run the same command to continue; answered jobs are kept.")
        sys.exit(2)


if __name__ == "__main__":
    main()
