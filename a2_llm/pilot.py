"""A2.2 pilot: the training maps the prompt and settings are developed on, and how each run scores there.

Choosing (--choose): 3 training episodes picked with a fixed seed; their test-like maps (at most 40
propositions, illocution layer annotated) sorted by size and sampled evenly down to 50 (`maps_50`,
used by sweep A), then evenly again down to 20 (`maps`, every run from sweep B on; D057). The example
pool then drops every training map from those episodes or sharing a specific speaker with them (not
the host, not audience members), so a pilot map never gets worked examples from its own debate.
    -> results/a2/pilot_maps.json

Scoring (--score=run1,run2,...): 1 sample per map, compared with gold:
- ARI general / focused (mean per map, exact official metric), keeping every listed relation, and
  keeping only medium / high-confidence ones;
- relations listed per map against gold; Inference direction right among gold Inferences found;
- turn-type accuracy on the gold links, overall and for the non-Asserting ones;
- seconds, tokens and API-equivalent $ per answer.
    -> results/a2/pilot_table.json (one entry per run, replaced when re-scored)
"""
import json
import random
import statistics as st
import sys

from common import RESULTS
from common.aif import load_map, view
from common.score import ari_labels
from common.split import episodes, is_specific, load_split, map_speakers
from a1_cross_encoder.decide import ari_f1, decode
from a2_llm import claude_cli
from a2_llm.parse import pair_scores
from a2_llm.prompt import GIANT
from a2_llm.runner import RUNS, jobs, load
from a2_llm.schema import link_id

OUT = RESULTS / "a2"
SEED, EPISODES, MAPS, SMALL = 13, 3, 50, 20     # SMALL: the pilot from sweep B on (user, 2026-10-04: D057)
KEEP = {"all listed": {"Inference": 0.01, "Conflict": 0.01, "Rephrase": 0.01},
        "medium+": {"Inference": 0.5, "Conflict": 0.5, "Rephrase": 0.5}}


def choose():
    speakers = map_speakers()
    ep_of, *_ = episodes(speakers)
    train = {int(p.stem.removeprefix("nodeset")): p for p in load_split("train")}
    eps = sorted({ep_of[i] for i in train if i in ep_of})
    chosen = sorted(random.Random(SEED).sample(eps, EPISODES))
    cands = []
    for i, p in train.items():
        if ep_of.get(i) in chosen:
            v = view(load_map(p), p.stem)
            if len(v.props) <= GIANT and v.anchors():
                cands.append((len(v.props), p.stem))
    cands.sort()
    picks = sorted({cands[round(j * (len(cands) - 1) / (MAPS - 1))][1] for j in range(MAPS)})
    voices = {s for i in train if ep_of.get(i) in chosen for s in speakers[i] if is_specific(s)}
    exclude = sorted(p.stem for i, p in train.items() if ep_of.get(i) in chosen or speakers[i] & voices)
    sizes = sorted(n for n, name in cands if name in set(picks))
    by_size = [name for _, name in cands if name in set(picks)]           # the 50, smallest first
    small = sorted({by_size[round(j * (MAPS - 1) / (SMALL - 1))] for j in range(SMALL)})
    out = {"seed": SEED, "episodes": chosen, "episode_maps_test_like": len(cands), "maps_50": picks, "maps": small,
           "sizes": {"min": sizes[0], "median": sizes[len(sizes) // 2], "max": sizes[-1]},
           "pool_exclude": exclude, "pool_excluded": len(exclude)}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pilot_maps.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("maps", "pool_exclude")}, indent=1))


def score(run: str) -> dict:
    cfg = RUNS[run]
    gen, foc = {k: [] for k in KEEP}, {k: [] for k in KEEP}
    listed, gold_n, dir_ok, dir_n, typed, typed_ok, rare, rare_ok = [], [], 0, 0, 0, 0, 0, 0
    used, missing = [], []
    for job in jobs(cfg):
        rec = load(job["key"], job["s"])
        if not rec or not rec["ok"]:
            missing.append(job["map"])
            continue
        s, att = job["s"], rec["attempts"][rec["answer"]]
        used.append(att)
        answer = att["result"]["structured_output"]
        g = load_map(next(p for p in load_split(cfg["split"]) if p.stem == job["map"]))
        props = set(view(g, job["map"]).props)
        gold = {k: c for k, c in ari_labels(g, props).items() if k[1] in props and k[0] != k[1]}
        n_pairs = len(props) * (len(props) - 1)
        rows, scores = pair_scores(s, [answer])
        for name, th in KEEP.items():
            pred = decode(rows, scores, th)
            general, focused = ari_f1(gold, pred, n_pairs)
            gen[name].append(100 * general[2])
            if focused is not None:
                foc[name].append(100 * focused[2])
            if name == "all listed":
                listed.append(len(pred))
                for (a, b), c in gold.items():
                    if c == 0 and (pred.get((a, b)) == 0 or pred.get((b, a)) == 0):
                        dir_n += 1
                        dir_ok += pred.get((a, b)) == 0
        gold_n.append(len(gold))
        types = {}
        for t, p, y in view(g, job["map"]).anchors():
            types.setdefault((t, p), y)
        said = {it["link"]: it["type"] for it in reversed(answer["illocutions"])}
        for t, p in s.links:
            y = types.get((s.turn[t], s.prop[p]))
            if y is None:
                continue
            typed += 1
            typed_ok += said.get(link_id(t, p)) == y
            if y != "Asserting":
                rare += 1
                rare_ok += said.get(link_id(t, p)) == y
    u = lambda a, f: (a["result"].get("usage") or {}).get(f, 0)
    think = lambda a: ((a["result"].get("usage") or {}).get("output_tokens_details") or {}).get("thinking_tokens", 0)
    mean = lambda xs: round(st.mean(xs), 2) if xs else None
    return {"run": run, "settings": {k: cfg.get(k) for k in ("k", "effort", "prompt", "samples")},
            "maps_scored": len(used), "missing": missing,
            **{f"ari_{name}": {"general": mean(gen[name]), "focused": mean(foc[name])} for name in KEEP},
            "relations_per_map": {"listed": mean(listed), "gold": mean(gold_n)},
            "inference_direction_right": round(dir_ok / dir_n, 3) if dir_n else None,
            "turn_types": {"accuracy": round(typed_ok / typed, 3) if typed else None,
                           "non_asserting_right": f"{rare_ok}/{rare}"},
            "per_answer": {"seconds": mean([a["seconds"] for a in used]),
                           "input_tokens": mean([u(a, "input_tokens") + u(a, "cache_creation_input_tokens")
                                                 + u(a, "cache_read_input_tokens") for a in used]),
                           "output_tokens": mean([u(a, "output_tokens") for a in used]),
                           "thinking_tokens": mean([think(a) for a in used]),
                           "api_usd": round(st.mean(claude_cli.api_usd(a["result"]) for a in used), 4) if used else None}}


NAMES = {0: "Inference", 1: "Conflict", 2: "Rephrase"}


def errors(run: str, show: int = 10) -> dict:
    """Error kinds over all pilot maps (every listed relation kept), and `show` maps written out in
    full, gold next to predicted -> results/a2/pilot_errors_<run>.md (for reading, A2.2 step 6)."""
    from collections import Counter
    from common.aif import load_map
    cfg = RUNS[run]
    kinds, dist = Counter(), {"missed": [], "invented": [], "right": []}
    pages = []
    for n, job in enumerate(jobs(cfg)):
        rec = load(job["key"], job["s"])
        if not rec or not rec["ok"]:
            continue
        s = job["s"]
        answer = rec["attempts"][rec["answer"]]["result"]["structured_output"]
        g = load_map(next(p for p in load_split(cfg["split"]) if p.stem == job["map"]))
        props = set(view(g, job["map"]).props)
        gold = {k: c for k, c in ari_labels(g, props).items() if k[1] in props and k[0] != k[1]}
        rows, scores = pair_scores(s, [answer])
        pred = decode(rows, scores, KEEP["all listed"])
        gu = {frozenset(k): (k, c) for k, c in gold.items()}
        pu = {frozenset(k): (k, c) for k, c in pred.items()}
        def d(pair):
            a, b = tuple(pair)
            return abs(s.pos[a] - s.pos[b]) if a in s.pos and b in s.pos else None
        lines = []
        for pair in sorted(gu.keys() | pu.keys(), key=lambda q: sorted(int(s.pid[x][1:]) for x in q)):
            (gk, gc), (pk, pc) = gu.get(pair, (None, None)), pu.get(pair, (None, None))
            if gc is None:
                kind = f"invented {NAMES[pc]}"
                dist["invented"].append(d(pair))
            elif pc is None:
                kind = f"missed {NAMES[gc]}"
                dist["missed"].append(d(pair))
            elif gc != pc:
                kind = f"{NAMES[gc]} read as {NAMES[pc]}"
            elif gc == 0 and gk != pk:
                kind = "Inference, wrong direction"
            else:
                kind = f"right {NAMES[gc]}"
                dist["right"].append(d(pair))
            kinds[kind] += 1
            fmt = lambda k, c: f"{s.pid[k[0]]} -> {s.pid[k[1]]} {NAMES[c]}" if k else "-"
            lines.append(f"| {fmt(gk, gc)} | {fmt(pk, pc)} | {kind} |")
        if n < show:
            pages.append(f"## {job['map']} ({len(props)} propositions)\n```text\n{s.text}\n```\n"
                         f"| Gold | Predicted | Verdict |\n|---|---|---|\n" + "\n".join(lines))
    med = lambda xs: st.median([x for x in xs if x is not None]) if any(x is not None for x in xs) else None
    out = {"run": run, "error_kinds": dict(kinds.most_common()),
           "median_turn_distance": {k: med(v) for k, v in dist.items()}}
    with open(OUT / f"pilot_errors_{run}.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Pilot errors: {run}\n\n```json\n{json.dumps(out, indent=1)}\n```\n\n" + "\n\n".join(pages) + "\n")
    print(json.dumps(out, indent=1))
    return out


def main():
    if "--choose" in sys.argv:
        choose()
        return
    if any(a.startswith("--errors=") for a in sys.argv):
        errors(next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--errors=")))
        return
    runs = next(a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--score=")).split(",")
    path = OUT / "pilot_table.json"
    table = json.loads(path.read_text()) if path.exists() else {}
    for run in runs:
        table[run] = score(run)
        print(json.dumps(table[run], indent=1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
