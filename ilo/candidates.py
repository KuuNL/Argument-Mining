"""Step 3.1: candidate (turn, proposition) pairs for the ILO labeller (D028).

The model only ever scores candidate pairs; everything else is None. A turn's candidates are its
own OWN most text-similar propositions, plus the PREV most similar to the previous turn and the
NEXT most similar to the next turn. Borrowing from neighbours catches short turns whose
proposition was rebuilt from context ("Yes" -> the claim being agreed with), which similarity to
the turn itself can't find. Node IDs and node order are never used (D029).

    python -m ilo.candidates        # recall of gold links on the train split, overall and per label
"""
from difflib import SequenceMatcher

from common.aif import MapView
from common.text import norm, similarity  # noqa: F401  (similarity re-exported for older callers)

OWN, PREV, NEXT = 5, 5, 3


def ranked(v: MapView) -> dict[str, list[tuple[str, float]]]:
    """For each dialogue turn: all propositions of the map, most similar first."""
    props = {i: norm(t) for i, t in v.props.items()}
    out = {}
    for lid in v.dialogue:
        body = norm(v.locs[lid].body)
        sims = [(i, SequenceMatcher(None, body, p, autojunk=False).ratio()) for i, p in props.items()]
        out[lid] = sorted(sims, key=lambda x: (-x[1], props[x[0]]))   # ties broken by text, not node order (D029)
    return out


def candidates(v: MapView, own=OWN, prev=PREV, nxt=NEXT, ranking=None) -> list[dict]:
    """Candidate pairs: {turn, prop, sim (to the turn itself), rank (by that similarity), via}."""
    ranking = ranking or ranked(v)
    order = v.dialogue
    out = []
    for i, lid in enumerate(order):
        sim_rank = {pid: (sim, r) for r, (pid, sim) in enumerate(ranking[lid])}
        via = {}
        for pid, _ in ranking[lid][:own]:
            via.setdefault(pid, "own")
        if i > 0:
            for pid, _ in ranking[order[i - 1]][:prev]:
                via.setdefault(pid, "prev")
        if i + 1 < len(order):
            for pid, _ in ranking[order[i + 1]][:nxt]:
                via.setdefault(pid, "next")
        out += [{"turn": lid, "prop": pid, "sim": round(sim_rank[pid][0], 4), "rank": sim_rank[pid][1], "via": how}
                for pid, how in via.items()]
    return out


def main():
    import json
    from collections import Counter
    from common.aif import load_map, view
    from common.split import load_split
    from common.timing import timed

    with timed("ilo", "3.1_candidates", own=OWN, prev=PREV, next=NEXT) as row:
        kept, total, n_pairs, via = Counter(), Counter(), 0, Counter()
        for p in load_split("train"):
            v = view(load_map(p), p.stem)
            cands = candidates(v)
            n_pairs += len(cands)
            cand = {(c["turn"], c["prop"]): c["via"] for c in cands}
            for s, t, y in v.anchors():
                if s in v.turn:
                    total[y] += 1
                    total["ALL"] += 1
                    if (s, t) in cand:
                        kept[y] += 1
                        kept["ALL"] += 1
                        via[cand[(s, t)]] += 1
        report = {"scheme": {"own": OWN, "prev": PREV, "next": NEXT}, "pairs": n_pairs,
                  "positive_share": round(kept["ALL"] / n_pairs, 4),
                  "links_kept": {y: f"{kept[y]}/{total[y]} = {kept[y] / total[y]:.4f}" for y, _ in total.most_common()},
                  "kept_links_found_via": dict(via)}
        row.update(pairs=n_pairs, recall=round(kept["ALL"] / total["ALL"], 4))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
