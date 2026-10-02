"""A1.5: complete AIF maps from A1's relations + the ILO labeller's illocutions, scored on dev.

Each output map = the test-format input
  + one S-node per predicted relation (RA / CA / MA, decide.py with the tuned thresholds)
  + one YA per predicted L->I link (ILO labeller, cache/ilo/links_<split>.json)
  + one unscored TA-anchored YA per relation (Inference -> Arguing, Rephrase -> Restating,
    Conflict -> Disagreeing; D024), anchored on the transition into the later proposition's turn.
Every map must pass validate() with zero problems.

    python -m a1_cross_encoder.predict --run=v1 --split=dev    -> results/a1/<run>/dev_global.json
    python -m a1_cross_encoder.predict --run=v1 --split=test   -> results/a1/<run>/test_global.json and
        test_predictions/*.json (the final run: test inputs only until every map is saved, D049)
"""
import json
from collections import defaultdict

import numpy as np

from common import ROOT
from common.aif import load_map, strip_to_input, test_paths, view
from common.score import aggregate, format_table, score_map
from common.split import load_split
from common.validate import validate
from a1_cross_encoder.assemble import S_NODE
from a1_cross_encoder.decide import CODE, decode, read
from a1_cross_encoder.train import RUNS, arg, dirs

TA_YA = {"Inference": "Arguing", "Rephrase": "Restating", "Conflict": "Disagreeing"}
NAMES = {v: k for k, v in CODE.items()}


def build_map(inp: dict, name: str, relations: dict, links: list, pos: dict) -> dict:
    """relations: {(src, tgt): code}; links: [(turn, prop, label)]; pos: prop -> turn index."""
    m = {"nodes": [dict(n) for n in inp["nodes"]], "edges": [dict(e) for e in inp["edges"]],
         "locutions": inp.get("locutions", [])}
    v = view(inp, name)
    into = {}                                              # L-node -> TA whose edge enters it
    types = {n["nodeID"]: n["type"] for n in inp["nodes"]}
    for e in inp["edges"]:
        if types.get(e["fromID"]) == "TA" and e["toID"] in v.locs:
            into.setdefault(e["toID"], e["fromID"])

    def add(node, src, dst):
        m["nodes"].append(node)
        m["edges"] += [{"edgeID": node["nodeID"] + "_in", "fromID": src, "toID": node["nodeID"], "formEdgeID": None},
                       {"edgeID": node["nodeID"] + "_out", "fromID": node["nodeID"], "toID": dst, "formEdgeID": None}]

    for k, (t, p, y) in enumerate(links):
        add({"nodeID": f"a1_ya_{k}", "type": "YA", "text": y}, t, p)
    for k, ((src, tgt), code) in enumerate(sorted(relations.items())):
        kind = NAMES[code]
        t, text = S_NODE[kind]
        sid = f"a1_s_{k}"
        add({"nodeID": sid, "type": t, "text": text}, src, tgt)
        later = max((pos[x] for x in (src, tgt) if x in pos), default=None)
        ta = into.get(v.dialogue[later]) if later is not None else None
        if ta:                                             # unscored anchor of the relation (D024)
            add({"nodeID": f"a1_ta_ya_{k}", "type": "YA", "text": TA_YA[kind]}, ta, sid)
    return m


def test_probs(run: str, rows: list[dict], path) -> tuple[np.ndarray, float]:
    """A1 probabilities for the test pairs from the run's best checkpoint (computed once, then cached)."""
    import time
    from transformers import AutoTokenizer
    from common.finetune import Config, load_model, predict, tokenize
    ckpt, _ = dirs(run)
    cfg = Config(num_labels=5, out_dir=ckpt, log_path=ckpt / "unused.jsonl", max_len=1024)
    pad = AutoTokenizer.from_pretrained(cfg.model).pad_token_id
    ids = tokenize(cfg.model, [r["text_a"] for r in rows], [r["text_b"] for r in rows], cfg.max_len)
    model = load_model(cfg, str(ckpt))
    t0 = time.perf_counter()
    probs = predict(model, ids, pad, cfg.eval_max_tokens)
    secs = time.perf_counter() - t0
    np.save(path, probs)
    return probs, secs


def relation_diagnostics(golds: dict, preds: dict) -> dict:
    """Pooled per-type precision / recall / F1 over ordered pairs, Inference direction, invented types."""
    from collections import Counter
    from common.score import ari_labels
    tp, fp, fn, spurious = Counter(), Counter(), Counter(), Counter()
    right = found = 0
    for name, g in golds.items():
        props = {n["nodeID"] for n in g["nodes"] if n["type"] == "I"}
        gold = {k: c for k, c in ari_labels(g, props).items() if k[1] in props and k[0] != k[1]}
        pred = {k: c for k, c in ari_labels(preds[name], props).items() if k[1] in props}
        for k in gold.keys() | pred.keys():
            a, b = gold.get(k), pred.get(k)
            if a == b:
                tp[a] += 1
            else:
                if b is not None: fp[b] += 1
                if a is not None: fn[a] += 1
        for (s, t), c in gold.items():
            if c == 0 and (pred.get((s, t)) == 0 or pred.get((t, s)) == 0):
                found += 1
                right += pred.get((s, t)) == 0
        for c in set(pred.values()) - set(gold.values()):
            spurious[NAMES[c]] += 1
    out = {}
    for c in (0, 1, 2):
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        out[NAMES[c]] = {"gold": tp[c] + fn[c], "predicted": tp[c] + fp[c], "precision": round(p, 3),
                         "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}
    return {"per_type": out, "inference_direction_correct": round(right / max(found, 1), 3),
            "inference_pairs_found": found, "maps_with_invented_type": dict(spurious)}


def main():
    run, split = arg("run", "v1"), arg("split", "dev")
    cap = int(arg("cap", 0)) or None
    ckpt, out = dirs(run)
    th = json.loads((out / ("thresholds.json" if cap is None else f"thresholds_cap{cap}.json")).read_text())
    rows = read(split, RUNS[run].get("data"))
    probs_path, infer_secs = ckpt / f"{split}_probs.npy", None
    if split == "test" and not probs_path.exists():
        probs, infer_secs = test_probs(run, rows, probs_path)
    else:
        probs = np.load(probs_path)
    z = np.log(np.clip(probs, 1e-12, 1.0)) / th["temperature"]
    z -= z.max(1, keepdims=True)
    cal = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    links = json.loads((ROOT / "cache" / "ilo" / f"links_{split}.json").read_text())
    by_map = defaultdict(list)
    for k, r in enumerate(rows):
        by_map[r["map"]].append(k)

    if split == "test":   # inputs now; the gold (data/evaluation) is read only after all maps are saved (D049)
        inputs = {name: load_map(inp) for name, inp, _ in test_paths()}
        gold_paths = {name: g for name, _, g in test_paths()}
        pred_dir = out / "test_predictions"
    else:
        gold = {p.stem: load_map(p) for p in load_split("dev") if p.stem in by_map}
        inputs = {n: strip_to_input(g) for n, g in gold.items()}
        pred_dir = ROOT / "cache" / "a1" / f"pred_{split}_{run}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    preds, problems, counts = {}, {}, defaultdict(int)
    for name, idx in sorted(by_map.items()):
        rs = [rows[k] for k in idx]
        relations = decode(rs, cal[idx], th["thresholds"], cap)
        pos = {}
        for r in rs:
            for p, q in ((r["a"], r["a_pos"]), (r["b"], r["b_pos"])):
                if q is not None:
                    pos[p] = q
        m = build_map(inputs[name], name, relations, links.get(name, []), pos)
        bad = validate(m)
        if bad:
            problems[name] = bad[:3]
        for c in relations.values():
            counts[NAMES[c]] += 1
        with open(pred_dir / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f, indent=1 if split == "test" else None)
        preds[name] = m
    if split == "test":
        gold = {name: load_map(gold_paths[name]) for name in preds}
    per_map = [score_map(gold[name], preds[name]) for name in sorted(preds)]
    agg = aggregate(per_map)
    report = {"run": run, "split": split, "cap": cap, "maps": len(per_map), "invalid_maps": len(problems),
              "problems": dict(list(problems.items())[:5]), "relations_predicted": dict(counts),
              "table": format_table(agg), **{k: agg[k] for k in ("ari", "ilo", "global", "skipped_maps")},
              "per_map_f1": {name: {"ari": [round(100 * s["ari"][0][2], 2), round(100 * s["ari"][1][2], 2) if s["ari"][1] else None],
                                    "ilo": [round(100 * s["ilo"][0][2], 2), round(100 * s["ilo"][1][2], 2) if s["ilo"][1] else None]}
                             for name, s in zip(sorted(preds), per_map)},
              "diagnostics": relation_diagnostics(gold, preds)}
    if infer_secs is not None:
        report["a1_inference_seconds"] = {"all_maps": round(infer_secs, 1), "per_map": round(infer_secs / len(preds), 2)}
    with open(out / (f"{split}_global.json" if cap is None else f"{split}_global_cap{cap}.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(report["table"])
    print(json.dumps({k: report[k] for k in ("maps", "invalid_maps", "relations_predicted", "skipped_maps")}))


if __name__ == "__main__":
    main()
