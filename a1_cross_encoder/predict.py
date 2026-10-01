"""A1.5: complete AIF maps from A1's relations + the ILO labeller's illocutions, scored on dev.

Each output map = the test-format input
  + one S-node per predicted relation (RA / CA / MA, decide.py with the tuned thresholds)
  + one YA per predicted L->I link (ILO labeller, cache/ilo/links_<split>.json)
  + one unscored TA-anchored YA per relation (Inference -> Arguing, Rephrase -> Restating,
    Conflict -> Disagreeing; D024), anchored on the transition into the later proposition's turn.
Every map must pass validate() with zero problems.

    python -m a1_cross_encoder.predict --run=v1 --split=dev   -> results/a1/<run>/dev_global.json
"""
import json
from collections import defaultdict

import numpy as np

from common import ROOT
from common.aif import load_map, strip_to_input, view
from common.score import aggregate, format_table, score_map
from common.split import load_split
from common.validate import validate
from a1_cross_encoder.assemble import S_NODE
from a1_cross_encoder.decide import CODE, decode, read
from a1_cross_encoder.train import arg, dirs

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


def main():
    run, split = arg("run", "v1"), arg("split", "dev")
    cap = int(arg("cap", 0)) or None
    ckpt, out = dirs(run)
    th = json.loads((out / ("thresholds.json" if cap is None else f"thresholds_cap{cap}.json")).read_text())
    rows = read(split)
    probs = np.load(ckpt / f"{split}_probs.npy")
    z = np.log(np.clip(probs, 1e-12, 1.0)) / th["temperature"]
    z -= z.max(1, keepdims=True)
    cal = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    links = json.loads((ROOT / "cache" / "ilo" / f"links_{split}.json").read_text())
    by_map = defaultdict(list)
    for k, r in enumerate(rows):
        by_map[r["map"]].append(k)

    gold = {p.stem: load_map(p) for p in load_split("dev") if p.stem in by_map}
    per_map, problems, counts = [], {}, defaultdict(int)
    pred_dir = ROOT / "cache" / "a1" / f"pred_{split}_{run}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in sorted(by_map.items()):
        rs = [rows[k] for k in idx]
        relations = decode(rs, cal[idx], th["thresholds"], cap)
        pos = {}
        for r in rs:
            for p, q in ((r["a"], r["a_pos"]), (r["b"], r["b_pos"])):
                if q is not None:
                    pos[p] = q
        inp = strip_to_input(gold[name])
        m = build_map(inp, name, relations, links.get(name, []), pos)
        bad = validate(m)
        if bad:
            problems[name] = bad[:3]
        for c in relations.values():
            counts[NAMES[c]] += 1
        with open(pred_dir / f"{name}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(m, f)
        per_map.append(score_map(gold[name], m))
    agg = aggregate(per_map)
    report = {"run": run, "split": split, "cap": cap, "maps": len(per_map), "invalid_maps": len(problems),
              "problems": dict(list(problems.items())[:5]), "relations_predicted": dict(counts),
              "table": format_table(agg), **{k: agg[k] for k in ("ari", "ilo", "global", "skipped_maps")}}
    with open(out / (f"{split}_global.json" if cap is None else f"{split}_global_cap{cap}.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(report["table"])
    print(json.dumps({k: report[k] for k in ("maps", "invalid_maps", "relations_predicted", "skipped_maps")}))


if __name__ == "__main__":
    main()
