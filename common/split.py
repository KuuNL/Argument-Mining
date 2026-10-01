"""Episode-grouped train/dev split of the training maps (PLAN: Foundation; D012).

The maps carry no episode label (locution dates are mostly import dates), so episodes are
recovered from their panels:
1. Core episodes: maps that share a speaker and sit close in nodeset-ID order (annotators
   worked through an episode in one run) are joined; groups of >= CORE_MIN maps are cores.
2. A core's panel = speakers in >= PANEL_MIN of its maps. Cores sharing >= TWIN_SHARED panellists
   are one episode (late re-annotation batches) and are merged. Every other map joins the core
   whose panel it overlaps most (panellists shared by several cores count fractionally).
3. Whole cores go to dev until DEV_SHARE of the maps is reached; then any training map that
   shares a panellist unique to a dev episode is moved to dev too (leak guard).

The split in splits/ is tracked in git and fixed for the whole project. Recomputing it only
checks that the code still reproduces it; overwriting needs --write (D027).

    python -m common.split            # recompute and compare with splits/
    python -m common.split --write    # deliberately regenerate splits/ (shows up as a git diff)
"""
import json
import random
import re
from collections import Counter, defaultdict

from . import SPLITS
from .aif import _num, load_map, training_paths, view

HOST = "Fiona Bruce"
GENERIC = re.compile(r"audience|unknown|speaker|^$", re.I)   # "Audience Member 14", "Alan Unknown", ...
ID_GAP, CORE_MIN, PANEL_MIN, TWIN_SHARED, DEV_SHARE, SEED = 20, 10, 3, 3, 0.15, 13


def is_specific(speaker: str) -> bool:
    return speaker != HOST and not GENERIC.search(speaker)


def map_speakers() -> dict[int, set[str]]:
    out = {}
    for p in training_paths():
        v = view(load_map(p))
        out[_num(p.stem)] = {v.locs[i].speaker for i in v.dialogue if is_specific(v.locs[i].speaker)}
    return out


def episodes(speakers: dict[int, set[str]]):
    ids = sorted(speakers)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    occurrences = defaultdict(list)
    for i in ids:
        for s in speakers[i]:
            occurrences[s].append(i)
    for occ in occurrences.values():
        for a, b in zip(occ, occ[1:]):
            if b - a <= ID_GAP:
                parent[find(b)] = find(a)

    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    cores = sorted((g for g in groups.values() if len(g) >= CORE_MIN), key=lambda g: g[0])
    spans = [[(g[0], g[-1])] for g in cores]      # contiguous nodeset-ID runs of each core

    def panel(g):
        c = Counter(s for i in g for s in speakers[i])
        return {s for s, k in c.items() if k >= PANEL_MIN}

    # Late re-annotation batches of an episode form a second core with the same panel: merge
    # cores sharing >= TWIN_SHARED panellists (distinct episodes share at most 2 recurring guests).
    merged = True
    while merged:
        merged = False
        for a in range(len(cores)):
            for b in range(a + 1, len(cores)):
                if len(panel(cores[a]) & panel(cores[b])) >= TWIN_SHARED:
                    cores[a] = sorted(cores[a] + cores.pop(b))
                    spans[a] += spans.pop(b)
                    merged = True
                    break
            if merged:
                break
    panels = [panel(g) for g in cores]
    n_panels = Counter(s for p in panels for s in p)

    episode_of = {i: e for e, g in enumerate(cores) for i in g}
    unassigned = []
    for i in ids:
        if i in episode_of:
            continue
        scores = sorted(((sum(1 / n_panels[s] for s in speakers[i] & p), e) for e, p in enumerate(panels)), reverse=True)
        best, runner_up = scores[0], scores[1] if len(scores) > 1 else (0, None)
        inside = [e for e, sp in enumerate(spans) if any(lo < i < hi for lo, hi in sp)]
        if best[0] >= 1 and best[0] > runner_up[0]:
            episode_of[i] = best[1]
        elif len(inside) == 1:           # no panellist, but inside one episode's ID run
            episode_of[i] = inside[0]
        else:
            unassigned.append(i)
    return episode_of, panels, n_panels, unassigned


def make_split(write: bool = False):
    speakers = map_speakers()
    episode_of, panels, n_panels, unassigned = episodes(speakers)
    members = defaultdict(list)
    for i, e in episode_of.items():
        members[e].append(i)

    order = list(range(len(panels)))
    random.Random(SEED).shuffle(order)
    dev_eps, dev, n = [], set(), len(speakers)
    for e in order:                       # whole episodes, staying within 1 point of DEV_SHARE
        if len(dev) >= (DEV_SHARE - 0.01) * n:
            break
        if len(dev) + len(members[e]) <= (DEV_SHARE + 0.01) * n:
            dev_eps.append(e)
            dev |= set(members[e])

    # Leak guard, repeated until stable: move to dev any training map that shares a panellist
    # unique to a dev episode, or two non-recurring speakers with one dev map (same episode).
    recurring = {s for s, k in n_panels.items() if k >= 2}
    unique_dev_panel = {s for e in dev_eps for s in panels[e] if n_panels[s] == 1}

    def leaks(train):
        return sorted((d, t) for d in dev for t in train if len((speakers[d] & speakers[t]) - recurring) >= 2)

    moved = []
    while True:
        rest = [i for i in speakers if i not in dev]
        step = sorted({i for i in rest if speakers[i] & unique_dev_panel} | {t for _, t in leaks(rest)})
        if not step:
            break
        moved += step
        dev |= set(step)
    train = sorted(set(speakers) - dev)
    leaks = leaks(train)   # the audit: must be empty

    report = {
        "params": {"id_gap": ID_GAP, "core_min": CORE_MIN, "panel_min": PANEL_MIN, "twin_shared": TWIN_SHARED,
                   "dev_share": DEV_SHARE, "seed": SEED},
        "counts": {"maps": len(speakers), "episodes": len(panels), "assigned": len(episode_of),
                   "unassigned": len(unassigned), "train": len(train), "dev": len(dev),
                   "dev_share": round(len(dev) / n, 3), "moved_by_leak_guard": len(moved),
                   "leaky_dev_train_pairs": len(leaks)},
        "leaky_pairs": leaks[:50],
        "dev_episodes": dev_eps,
        "episodes": [{"episode": e, "maps": len(members[e]), "ids": [min(members[e]), max(members[e])],
                      "panel": sorted(panels[e]), "split": "dev" if e in dev_eps else "train"}
                     for e in range(len(panels))],
        "unassigned": unassigned,
        "moved_by_leak_guard": moved,
    }
    if write:
        SPLITS.mkdir(exist_ok=True)
        for name, ids in (("train", train), ("dev", sorted(dev))):
            with open(SPLITS / f"{name}.txt", "w", newline="\n") as f:
                f.write("".join(f"nodeset{i}\n" for i in ids))
        with open(SPLITS / "episodes.json", "w", newline="\n") as f:
            f.write(json.dumps(report, indent=1))
    return report, speakers, set(train), dev, episode_of


def matches_stored(train: set, dev: set) -> bool:
    """True if a recomputed split equals the tracked splits/{train,dev}.txt."""
    stored = {name: {int(s.removeprefix("nodeset")) for s in (SPLITS / f"{name}.txt").read_text().split()}
              for name in ("train", "dev")}
    return stored["train"] == set(train) and stored["dev"] == set(dev)


def load_split(name: str):
    """Paths of the maps in splits/{name}.txt ('train' or 'dev')."""
    from . import DATA
    return [DATA / "training" / f"{s}.json" for s in (SPLITS / f"{name}.txt").read_text().split()]


if __name__ == "__main__":
    import sys
    write = "--write" in sys.argv
    report, _, train, dev, _ = make_split(write=write)
    print(json.dumps(report["counts"]))
    print("written to splits/" if write else f"matches tracked splits/: {matches_stored(train, dev)}")
