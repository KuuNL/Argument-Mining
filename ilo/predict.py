"""Predicted L->I links for any map, using the selected ILO run and its tuned decision layer.

The ARI approaches use these links to place each proposition in the dialogue on dev and test
maps (D040). Inputs are test-format maps: no gold is read.

    python -m ilo.predict --split=dev     # all 228 dev maps (stripped) -> cache/ilo/links_dev.json
    python -m ilo.predict --split=test    # the 11 test inputs           -> cache/ilo/links_test.json
"""
import json

import numpy as np

from common.aif import load_map, strip_to_input, test_paths, view
from common.finetune import Config, load_model, predict, tokenize
from common.split import load_split
from ilo.data import CACHE, LABELS, inputs
from ilo.decode import decode_map
from ilo.train import arg, dirs, selected_run


def calibrate(probs: np.ndarray, t: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-12, 1.0)) / t
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Labeller:
    def __init__(self, run: str | None = None):
        self.run = run or selected_run()
        ckpt, out = dirs(self.run)
        self.th = json.loads((out / "thresholds_shared.json").read_text())
        self.cfg = Config(num_labels=len(LABELS), out_dir=ckpt, log_path=out / "unused.jsonl")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg.model)
        self.sep, self.pad = tok.sep_token, tok.pad_token_id
        self.model = load_model(self.cfg, str(ckpt))

    def links(self, maps: dict[str, dict]) -> tuple[dict[str, list], list[dict], np.ndarray]:
        """maps: name -> test-format AIF. Returns ({name: [(turn, prop, label)]}, rows, raw probs)."""
        rows = []
        for name, aif in maps.items():
            rows += inputs(view(aif, name), name, self.sep)
        ids = tokenize(self.cfg.model, [r["text_a"] for r in rows], [r["text_b"] for r in rows], self.cfg.max_len)
        raw = predict(self.model, ids, self.pad, self.cfg.eval_max_tokens)
        cal = calibrate(raw, self.th["temperature"])
        idx = {}
        for k, r in enumerate(rows):
            idx.setdefault(r["map"], []).append(k)
        out = {name: [list(x) for x in decode_map([rows[k] for k in idx.get(name, [])], cal[idx.get(name, [])],
                                                  self.th["link_min"], self.th["type_min"])] if name in idx else []
               for name in maps}
        return out, rows, raw


def main():
    split = arg("split", "dev")
    if split == "dev":
        maps = {p.stem: strip_to_input(load_map(p)) for p in load_split("dev")}
    else:
        maps = {name: load_map(inp) for name, inp, _ in test_paths()}   # inputs only, gold never read
    lab = Labeller()
    links, rows, raw = lab.links(maps)
    report = {"run": lab.run, "split": split, "maps": len(maps), "links": sum(len(v) for v in links.values())}
    if split == "dev":
        # Consistency: on the ILO dev maps these must equal the links decoded from the run's saved probabilities.
        from ilo.tune import Dev
        dev = Dev(dirs(lab.run)[0] / "dev_probs.npy")
        cal = calibrate(dev.raw, lab.th["temperature"])
        same = all(sorted(map(tuple, links[n])) == sorted(dev.links(cal, n, lab.th["link_min"], lab.th["type_min"]))
                   for n in dev.names)
        report["matches_saved_dev_predictions"] = same
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / f"links_{split}.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(links, f)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
