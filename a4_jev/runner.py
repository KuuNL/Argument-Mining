"""A4 runner: a named run = fixed question settings over a list of maps. Resumable; nothing is sent twice.

Each map's questions go out as one or more requests (questions.split). Answers live in a content-addressed cache,
cache/a4/answers/<key>.json, where key = a hash of the exact request (model, state, questions, repeat number), so
a stopped run resumes exactly and identical requests in different runs share one answer. `repeat` > 0 re-sends an
identical request on purpose (the repeatability probe). A refusal (no credit, no access, a malformed request)
stops the run; rate limits are retried inside the client. Every request's gateway cost is recorded (D075).

    python -m a4_jev.runner --run=probe                 # send what is missing, then summarise
    python -m a4_jev.runner --run=probe --export        # also results/a4/<run>/responses.jsonl.gz (committed)
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
from common.aif import load_map, view
from common.split import load_split
from common.timing import timed
from a2_llm.runner import resolve, targets
from a4_jev import client
from a4_jev.parse import problems
from a4_jev.questions import REQUEST_BUDGET, build

CACHE = ROOT / "cache" / "a4" / "answers"
OUT = RESULTS / "a4"
SMOKE = ["nodeset18266", "nodeset18274", "nodeset23907"]     # A2's smoke maps: 8, 15 and 28 propositions
RUNS = {
    # A4.1 live check (D073): A2's smoke maps + the two largest test-like training maps outside the pilot episodes.
    # (They ran at a 56k request budget, before D076 lowered the default to 50k; kept so their cache keys hold.)
    "probe": {"split": "train", "maps": "probe", "defs": "short", "k": 0, "repeats": 1, "budget": 56_000},
    "probe-repeat": {"split": "train", "maps": SMOKE[1:2], "defs": "short", "k": 0, "repeats": 2, "budget": 56_000},
    # A4.2 pilot (D077): A2's 50 pilot maps (training split, gold links); worked-example pool without the pilot episodes.
    "pilot-k0": {"split": "train", "maps": "pilot50", "exclude": "pilot", "defs": "short", "k": 0},
    "pilot-k1": {"split": "train", "maps": "pilot50", "exclude": "pilot", "defs": "short", "k": 1},
    # A4.2 wording fix (D078): source questions ask what P supports / restates / attacks, not what it "bears on".
    "pilot-k0-w2": {"split": "train", "maps": "pilot50", "exclude": "pilot", "defs": "short", "k": 0, "wording": "v2"},
    # Frozen by A4.2 (D078): guide v1, short options, k = 0, wording v1, pair + source questions, both orders.
    "dev": {"split": "dev", "maps": "dev225", "defs": "short", "k": 0},
    "test": {"split": "test", "maps": "test11", "defs": "short", "k": 0},
}


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


@lru_cache
def probe_maps() -> tuple[str, ...]:
    from a2_llm.prompt import GIANT
    pilot = json.loads((RESULTS / "a2" / "pilot_maps.json").read_text())
    away = set(pilot["pool_exclude"]) | set(SMOKE)
    sizes = []
    for p in load_split("train"):
        if p.stem in away:
            continue
        v = view(load_map(p), p.stem)
        if len(v.props) <= GIANT and v.anchors():
            sizes.append((len(v.props), p.stem))
    return tuple(SMOKE + [n for _, n in sorted(sizes, reverse=True)[:2]])


def config(run: str) -> dict:
    cfg = dict(RUNS[run])
    if cfg["maps"] == "probe":
        cfg["maps"] = list(probe_maps())
    return resolve(cfg) if isinstance(cfg["maps"], str) else cfg     # named lists: dev225, test11, pilot50 (A2's)


@lru_cache
def _pool(exclude: tuple):
    from a2_llm.prompt import Pool
    return Pool(set(exclude))


def builds(run: str) -> dict:
    """map -> Built, rebuilt deterministically from the run's settings."""
    cfg = config(run)
    pool = _pool(tuple(cfg.get("exclude", ()))) if cfg["k"] else None
    return {name: build(inp, name, links, defs=cfg["defs"], k=cfg["k"], pool=pool, version=cfg.get("guide", "v1"),
                        orders=tuple(cfg.get("orders", ("f", "r"))), budget=cfg.get("budget", REQUEST_BUDGET),
                        wording=cfg.get("wording", "v1"))
            for name, inp, links in targets(cfg)}


def key_of(request: dict, repeat: int) -> str:
    return hashlib.sha256(json.dumps([client.MODEL, request["state"], request["questions"], repeat],
                                     ensure_ascii=False).encode()).hexdigest()


def jobs(run: str):
    cfg = config(run)
    for name, b in builds(run).items():
        for part, req in enumerate(b.requests):
            for r in range(cfg.get("repeats", 1)):
                yield {"key": key_of(req, r), "map": name, "part": part, "parts": len(b.requests), "repeat": r,
                       "request": req, "chars": len(json.dumps(req, ensure_ascii=False))}


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
    rec = {k: job[k] for k in ("key", "map", "part", "parts", "repeat", "chars")}
    rec.update(run=run, n_questions=len(req["questions"]), when=datetime.now().isoformat(timespec="seconds"), result=res)
    save(rec)
    resp = res["response"]
    print(f"  {job['map']} part {job['part'] + 1}/{job['parts']} r{job['repeat']}: {len(req['questions'])} questions, "
          f"{client.input_tokens(resp)} tokens, ${client.cost(resp):.6f}, {res['seconds']} s", flush=True)
    return rec


def answers(run: str) -> dict:
    """map -> repeat -> {qid: answer}, merged over a map's requests (only complete repeats)."""
    out = {}
    js = list(jobs(run))
    for name in dict.fromkeys(j["map"] for j in js):
        mine = [j for j in js if j["map"] == name]
        for r in sorted({j["repeat"] for j in mine}):
            recs = [load(j["key"]) for j in mine if j["repeat"] == r]
            if all(recs):
                merged = {}
                for rec in recs:
                    merged.update(rec["result"]["response"].get("answers") or {})
                out.setdefault(name, {})[r] = merged
    return out


def summary(run: str) -> dict:
    js = list(jobs(run))
    recs = [r for r in (load(j["key"]) for j in js) if r]
    resp = [r["result"]["response"] for r in recs]
    toks = [client.input_tokens(x) for x in resp]
    secs = [r["result"]["seconds"] for r in recs]
    bs = builds(run)
    ans = answers(run)
    bad = {m: problems(bs[m], a)[:3] for m, reps in ans.items() for a in reps.values() if problems(bs[m], a)}
    per_map = {}
    for r in recs:
        d = per_map.setdefault((r["map"], r["repeat"]), {"requests": 0, "questions": 0, "tokens": 0, "usd": 0.0, "seconds": 0.0})
        d["requests"] += 1
        d["questions"] += r["n_questions"]
        d["tokens"] += client.input_tokens(r["result"]["response"])
        d["usd"] += client.cost(r["result"]["response"])
        d["seconds"] += r["result"]["seconds"]
    out = {"requests": len(js), "answered": len(recs), "maps_complete": sum(len(v) for v in ans.values()),
           "maps_with_problems": bad, "usd": round(sum(client.cost(x) for x in resp), 6), "input_tokens": sum(toks),
           "models": sorted({x.get("model", "?") for x in resp})}
    if recs:
        out["per_request"] = {"seconds_p50": st.median(secs), "seconds_max": max(secs), "tokens_p50": st.median(toks),
                              "tokens_max": max(toks), "retries": sum(r["result"]["attempts"] - 1 for r in recs),
                              "chars_per_token": round(sum(r["chars"] for r in recs) / max(sum(toks), 1), 3)}
        out["per_map"] = [{"map": m, "repeat": rp, **{k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()}}
                          for (m, rp), d in per_map.items()]
    return out


def main():
    run = arg("run", "probe")
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

    with timed("a4", f"run:{run}", requests=len(js), to_send=len(todo), workers=workers) as row:
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
