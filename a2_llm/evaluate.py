"""A2.4: complete maps, the LLM's turn types, and the ablations (dev).

1. Turn types (Task B). The ILO labeller's links stay (cache/ilo/links_dev.json); the LLM's most-voted type
   for a link replaces the labeller's when its vote share (1/3, 2/3 or 1) clears that type's threshold.
   Thresholds by coordinate ascent on mean per-map ILO general F1 (ilo/tune.ilo_f1, exact), cross-fitted by
   dev episode. Adopted only if the honest ILO beats the labeller alone on the same links.
2. Complete maps: A2's relations (A2.3 thresholds) + links + types + unscored anchors (A1's build_map), all
   validated and scored with the official port -> dev ARI / ILO / Global, tuned on all dev and cross-fitted.
3. Ablations (ARI): answers 1 / 2 / 3; plain votes vs confidence-weighted; majority vote (>= 2 of 3) vs tuned
   thresholds; the model's own Conflict / Rephrase direction vs the order rule; relations more than 10 turns
   apart dropped; reasoning on / off (effort medium vs xhigh, 1 answer each).

    python -m a2_llm.evaluate     -> results/a2/dev-v1/a24_report.json
"""
import json
import sys
from collections import Counter, defaultdict

from common import RESULTS, ROOT
from common.aif import load_map, strip_to_input
from common.score import aggregate, ilo_labels, score_map
from common.split import episodes, map_speakers
from common.validate import validate
from a1_cross_encoder.decide import ari_f1, mean_f1, tune
from a1_cross_encoder.predict import build_map
from a1_cross_encoder.tune import diagnostics
from a2_llm.parse import type_votes
from a2_llm.schema import TURN_TYPES
from a2_llm.tune import Scores, both, cross_fitted
from ilo.tune import ilo_f1

OUT = RESULTS / "a2"
RUN = "dev-v1"
NEVER = 1.01
GRID = [0.33, 0.66, 0.99, NEVER]                   # vote shares are 1/3, 2/3, 1
MAJORITY = {"Inference": 0.6, "Conflict": 0.6, "Rephrase": 0.6}


def folds_of(names):
    ep_of, *_ = episodes(map_speakers())
    by_ep = defaultdict(list)
    for n in names:
        by_ep[ep_of[int(n.removeprefix("nodeset"))]].append(n)
    return dict(sorted(by_ep.items()))


# ---------------------------------------------------------------- 1. turn types

class Types:
    def __init__(self, dev: Scores):
        self.links = json.loads((ROOT / "cache" / "ilo" / "links_dev.json").read_text())
        self.names = dev.names
        self.votes = {n: type_votes(dev.s[n], dev.answer_lists[n]) for n in self.names}
        self.gold, self.n_pairs = {}, {}
        for n in self.names:
            g = dev.gold_aif[n]
            props = {x["nodeID"] for x in g["nodes"] if x["type"] == "I"}
            locs = {x["nodeID"] for x in g["nodes"] if x["type"] == "L"}
            self.gold[n] = {k: y for k, y in ilo_labels(g, props | locs).items() if k[0] in locs and k[1] in props}
            self.n_pairs[n] = len(locs) * len(props)

    def typed(self, name, th) -> list[list]:
        """The labeller's links, with the LLM's type where it clears the threshold."""
        out = []
        for t, p, y in self.links.get(name, []):
            v = self.votes[name].get((t, p), {})
            best = max(v, key=lambda k: (v[k], k == y)) if v else y          # ties keep the labeller's type
            out.append([t, p, best if best != y and v[best] >= th.get(best, NEVER) else y])
        return out

    def scores(self, th, names=None):
        return {n: ilo_f1(self.gold[n], {(t, p): y for t, p, y in self.typed(n, th)}, self.n_pairs[n])
                for n in (names or self.names)}

    def tune(self, names):
        th = {y: NEVER for y in TURN_TYPES}
        best = mean_f1(self.scores(th, names))
        for _ in range(2):
            for y in TURN_TYPES:
                for v in GRID:
                    f = mean_f1(self.scores({**th, y: v}, names))
                    if f > best + 1e-9:
                        best, th = f, {**th, y: v}
        return th


def turn_types(dev: Scores):
    """(report, thresholds fitted on all dev, thresholds per held-out episode, the Types object)."""
    ty = Types(dev)
    labeller = ty.scores({})
    final = ty.tune(ty.names)
    held, per_fold = {}, {}
    for ep, names in folds_of(ty.names).items():
        th = ty.tune([n for n in ty.names if n not in names])
        held.update(ty.scores(th, names))
        per_fold[ep] = th
    changed = Counter()
    for n in ty.names:
        for (t, p, y0), (_, _, y1) in zip(ty.links.get(n, []), ty.typed(n, final)):
            if y0 != y1:
                right = ty.gold[n].get((t, p))
                changed[f"{y0} -> {y1} ({'right' if right == y1 else 'was right' if right == y0 else 'both wrong'})"] += 1
    rep = {"labeller alone": both(labeller), "llm types, tuned on all dev": both(ty.scores(final)),
           "llm types, cross-fitted (honest)": both(held), "thresholds": final, "changes_on_dev": dict(changed.most_common())}
    rep["adopted"] = rep["llm types, cross-fitted (honest)"]["general"] > rep["labeller alone"]["general"]
    return rep, final, per_fold, ty


# ---------------------------------------------------------------- 2. complete maps

def complete_maps(dev: Scores, ty: Types, rel_th, type_th, per_fold=None) -> dict:
    """Official scores of full maps. per_fold: {episode: (relation thresholds, type thresholds)} for honest scores."""
    fold = {n: ep for ep, names in folds_of(dev.names).items() for n in names}
    per_map, invalid = [], 0
    for n in dev.names:
        rt, tt = per_fold[fold[n]] if per_fold else (rel_th, type_th)
        inp = strip_to_input(dev.gold_aif[n])
        m = build_map(inp, n, dev.predicted(dev.probs, n, rt), ty.typed(n, tt), dev.s[n].pos)
        invalid += bool(validate(m))
        per_map.append(score_map(dev.gold_aif[n], m))
    agg = aggregate(per_map)
    return {"invalid_maps": invalid, **{task: {s: round(agg[task][s][2], 2) for s in ("general", "focused")}
                                        for task in ("ari", "ilo", "global")}}


# ---------------------------------------------------------------- 3. ablations

def honest(dev: Scores) -> dict:
    held, _ = cross_fitted(dev)
    return both(held)


def own_direction(dev: Scores, th) -> dict:
    """Same decisions, but each Conflict / Rephrase takes the direction most answers gave it."""
    rule, own = {}, {}
    for n in dev.names:
        pred = dev.predicted(dev.probs, n, th)
        s, flipped = dev.s[n], dict(pred)
        for (src, tgt), code in pred.items():
            if code == 0:
                continue
            kind = {1: "Conflict", 2: "Rephrase"}[code]
            fwd = back = 0
            for ans in dev.answer_lists[n]:
                for r in ans["relations"]:
                    if r["type"] == kind:
                        a, b = s.prop[r["from"]], s.prop[r["to"]]
                        fwd += (a, b) == (src, tgt)
                        back += (a, b) == (tgt, src)
            if back > fwd:
                del flipped[(src, tgt)]
                flipped[(tgt, src)] = code
        rule[n] = ari_f1(dev.gold[n], pred, dev.n_pairs[n])
        own[n] = ari_f1(dev.gold[n], flipped, dev.n_pairs[n])
    return {"order rule": both(rule), "model's own direction": both(own)}


def reach(dev: Scores, th, far: int = 10) -> dict:
    kept, n_far, right_far = {}, 0, 0
    for n in dev.names:
        pred, pos = dev.predicted(dev.probs, n, th), dev.s[n].pos
        near = {k: c for k, c in pred.items() if not (k[0] in pos and k[1] in pos and abs(pos[k[0]] - pos[k[1]]) > far)}
        n_far += len(pred) - len(near)
        right_far += sum(dev.gold[n].get(k) == c for k, c in pred.items() if k not in near)
        kept[n] = ari_f1(dev.gold[n], near, dev.n_pairs[n])
    return {"all relations": both(dev.scores(dev.probs, th)), f"relations over {far} turns dropped": both(kept),
            f"relations over {far} turns apart": n_far, "of which right": right_far}


def reasoning() -> dict:
    out = {}
    for name, run in (("xhigh (reasons), answer 1", RUN), ("medium (no reasoning), 1 answer", "dev-medium")):
        dev = Scores(run, use={0})
        final = tune(dev, dev.probs, dev.names)
        dg = diagnostics(dev, dev.probs, final["th"])
        out[name] = {"honest": honest(dev), "thresholds": final["th"],
                     "per_type": {k: {f: v[f] for f in ("precision", "recall", "f1")} for k, v in dg["per_type"].items()},
                     "inference_direction_right": dg["inference_direction_correct"],
                     "maps_with_invented_type": dg["maps_with_spurious_type"]}
    return out


def main():
    dev = Scores(RUN)
    dev.gold_aif = {n: load_map(ROOT / "data" / "training" / f"{n}.json") for n in dev.names}
    rel = json.loads((OUT / RUN / "thresholds.json").read_text())["thresholds"]
    report = {"run": RUN}

    types, type_th, type_folds, ty = turn_types(dev)
    report["turn_types"] = types
    _, rel_folds = cross_fitted(dev)
    use_types = type_th if types["adopted"] else {}
    folds = {ep: ({k: v for k, v in rel_folds[str(ep)].items() if k in ("Inference", "Conflict", "Rephrase")},
                  type_folds[ep] if types["adopted"] else {}) for ep in folds_of(dev.names)}
    report["complete_maps"] = {
        "A2 relations + labeller types, tuned on all dev": complete_maps(dev, ty, rel, {}),
        "A2 relations + chosen types, tuned on all dev": complete_maps(dev, ty, rel, use_types),
        "A2 relations + chosen types, cross-fitted (honest)": complete_maps(dev, ty, rel, use_types, folds),
        "A1 reference (D045)": {"tuned on all dev": [70.53, 57.29], "honest (approx.)": [70.3, 56.6]},
    }
    ablations = {}
    for k in (1, 2):
        ablations[f"{k} answer(s)"] = honest(Scores(RUN, use=set(range(k))))
    ablations["3 answers (main)"] = honest(dev)
    flat = Scores(RUN, flat=True)
    ablations["plain votes, tuned (confidence ignored)"] = honest(flat)
    ablations["majority vote (>= 2 of 3), no tuning"] = both(flat.scores(flat.probs, MAJORITY))
    ablations["direction"] = own_direction(dev, rel)
    ablations["reach"] = reach(dev, rel)
    if "--no-reasoning" not in sys.argv:              # needs the dev-medium run complete
        ablations["reasoning"] = reasoning()
    report["ablations_ari"] = ablations
    path = OUT / RUN / "a24_report.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
