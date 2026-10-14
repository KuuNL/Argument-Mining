"""Runner: a named run = fixed question settings over a list of maps. Resumable; nothing is sent twice.

After a4_jev/runner.py: each map's questions go out as one or more requests; answers live in a content-addressed
cache, cache/ilo_jev/answers/<key>.json (key = a hash of model, state and questions); a refusal (no credit, no
access, a malformed request) stops the run; every request's gateway cost is recorded (D075). Maps are sent in
test format with no links; test maps are not supported (D099: dev only).

    python -m ilo_jev.runner --run=smoke                 # send what is missing, then summarise
    python -m ilo_jev.runner --run=pilot-k0 --export     # also results/ilo_jev/<run>/responses.jsonl.gz (committed)
    ... --workers=4                                      # requests in parallel (default 4)
Run under `bash -lc` in WSL, so AI_GATEWAY_API_KEY from ~/.profile is set.
"""
import gzip
import hashlib
import json
import statistics as st
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache

from common import RESULTS, ROOT
from common.timing import timed
from a4_jev import client
from ilo_jev.parse import problems
from ilo_jev.questions import build
from ilo_llm.runner import SMOKE, resolve, targets

CACHE = ROOT / "cache" / "ilo_jev" / "answers"
OUT = RESULTS / "ilo_jev"
RUNS = {
    "smoke": {"split": "train", "maps": SMOKE, "k": 0},
    # Pilot (D099): the 20 pilot maps; worked examples from the pool without the pilot episodes.
    "pilot-k0": {"split": "train", "maps": "pilot", "exclude": "pilot", "k": 0},
    "pilot-k4": {"split": "train", "maps": "pilot", "exclude": "pilot", "k": 4},
    # Frozen from the pilot (D100): k = 4 by the D099 rule.
    "dev": {"split": "dev", "maps": "dev225", "k": 4},
}


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def config(run: str) -> dict:
    return resolve(dict(RUNS[run]))


@lru_cache
def _pool(exclude: tuple):
    from a2_llm.prompt import Pool
    return Pool(set(exclude))


@lru_cache
def _builds(run: str, k: int) -> dict:
    cfg = config(run)
    pool = _pool(tuple(cfg.get("exclude", ()))) if k else None
    return {name: build(inp, name, k=k, pool=pool) for name, inp in targets(cfg)}


def builds(run: str, k: int | None = None) -> dict:
    """map -> Built, rebuilt deterministically from the run's settings (k may be overridden, e.g. by check.py)."""
    return _builds(run, RUNS[run]["k"] if k is None else k)


def key_of(request: dict) -> str:
    return hashlib.sha256(json.dumps([client.MODEL, request["state"], request["questions"]], ensure_ascii=False).encode()).hexdigest()


def jobs(run: str, k: int | None = None):
    for name, b in builds(run, k).items():
        for part, req in enumerate(b.requests):
            yield {"key": key_of(req), "map": name, "part": part, "parts": len(b.requests), "request": req,
                   "chars": len(json.dumps(req, ensure_ascii=False))}


def load(key: str):
    path = CACHE / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save(rec: dict):
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"{rec['key']}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, ensure_ascii=False)


def send(job: dict, run: str) -> dict:
    req = job["request"]
    res = client.evaluate(req["state"], req["questions"])
    rec = {k: job[k] for k in ("key", "map", "part", "parts", "chars")}
    rec.update(run=run, n_questions=len(req["questions"]), when=datetime.now().isoformat(timespec="seconds"), result=res)
    save(rec)
    resp = res["response"]
    print(f"  {job['map']} part {job['part'] + 1}/{job['parts']}: {len(req['questions'])} questions, "
          f"{client.input_tokens(resp)} tokens, ${client.cost(resp):.6f}, {res['seconds']} s", flush=True)
    return rec


def answers(run: str) -> dict:
    """map -> {qid: answer}, merged over a map's requests (complete maps only)."""
    out = {}
    js = list(jobs(run))
    for name in dict.fromkeys(j["map"] for j in js):
        recs = [load(j["key"]) for j in js if j["map"] == name]
        if all(recs):
            merged = {}
            for rec in recs:
                merged.update(rec["result"]["response"].get("answers") or {})
            out[name] = merged
    return out


def summary(run: str) -> dict:
    js = list(jobs(run))
    recs = [r for r in (load(j["key"]) for j in js) if r]
    resp = [r["result"]["response"] for r in recs]
    toks = [client.input_tokens(x) for x in resp]
    secs = [r["result"]["seconds"] for r in recs]
    bs = builds(run)
    ans = answers(run)
    bad = {m: problems(bs[m], a)[:3] for m, a in ans.items() if problems(bs[m], a)}
    out = {"requests": len(js), "answered": len(recs), "maps": len(bs), "maps_complete": len(ans), "maps_with_problems": bad,
           "usd": round(sum(client.cost(x) for x in resp), 6), "input_tokens": sum(toks),
           "models": sorted({x.get("model", "?") for x in resp})}
    if recs:
        out["per_request"] = {"seconds_p50": st.median(secs), "seconds_max": max(secs), "tokens_p50": st.median(toks),
                              "tokens_max": max(toks), "retries": sum(r["result"]["attempts"] - 1 for r in recs),
                              "chars_per_token": round(sum(r["chars"] for r in recs) / max(sum(toks), 1), 3)}
        per_map = {}
        for r in recs:
            d = per_map.setdefault(r["map"], {"requests": 0, "questions": 0, "tokens": 0, "usd": 0.0, "seconds": 0.0})
            d["requests"] += 1
            d["questions"] += r["n_questions"]
            d["tokens"] += client.input_tokens(r["result"]["response"])
            d["usd"] += client.cost(r["result"]["response"])
            d["seconds"] += r["result"]["seconds"]
        out["per_map"] = [{"map": m, **{k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()}} for m, d in per_map.items()]
    return out


def main():
    run = arg("run", "smoke")
    if RUNS[run]["k"] is None:
        sys.exit(f"run {run}: k is not set yet (chosen on the pilot, D099)")
    workers = int(arg("workers", 4))
    js = list(jobs(run))
    todo = [j for j in js if load(j["key"]) is None]
    print(f"run {run}: {len(js)} requests, {len(js) - len(todo)} cached, {len(todo)} to send, {workers} at a time", flush=True)
    halt, refused = threading.Event(), []

    def work(job):
        if halt.is_set():
            return
        try:
            send(job, run)
        except client.Refused as e:
            refused.append(str(e))
            halt.set()

    with timed("ilo_jev", f"run:{run}", requests=len(js), to_send=len(todo), workers=workers) as row:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(work, todo))
        rep = {"run": run, "settings": {k: v for k, v in config(run).items() if k != "exclude"},
               "refused": refused[:1], **summary(run)}
        row.update({k: rep[k] for k in ("answered", "usd", "input_tokens")})
    (OUT / run).mkdir(parents=True, exist_ok=True)
    with open(OUT / run / "summary.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rep, indent=1))
    if "--export" in sys.argv:
        with gzip.open(OUT / run / "responses.jsonl.gz", "wt", encoding="utf-8") as f:
            for j in js:
                rec = load(j["key"])
                if rec:
                    f.write(json.dumps({**rec, "request": j["request"]}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "per_map"}, indent=1))
    if refused:
        print(f"\nStopped: the gateway refused a request.\n{refused[0]}")
        sys.exit(2)


if __name__ == "__main__":
    main()
