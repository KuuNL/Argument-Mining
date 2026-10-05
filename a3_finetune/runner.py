"""A3 generation runs with an on-disk answer cache (D065).

    python -m a3_finetune.runner --run=v1 --adapter=ep2.0 --split=dev --greedy [--export]
    python -m a3_finetune.runner --run=v1 --adapter=ep2.0 --split=dev --samples=8 [--seed=0] [--export]
    python -m a3_finetune.runner --run=v1 --adapter=none --split=dev --greedy        # the untrained model

Splits: dev (225 maps, the labeller's links), test (11 inputs), train (the run's own training maps).
Greedy answers are batched across maps, left-padded, within a token budget. Samples continue the map's
greedy answer from just before "relations", so the greedy answer is generated first if it isn't cached.

Each map's answer is one JSON file in cache/a3/<run>/answers/, named by a hash of everything that
determines it: base model, adapter weights, prompt, mode, n, seed, length cap and grammar settings. A cached
answer is never generated again, so a stopped run resumes. Seeds make samples repeatable only with the same
batch and library versions, so the answers themselves are the record: --export writes them to
results/a3/<run>/<adapter>_<split>_<mode>.jsonl.gz, which is committed.
"""
import gzip
import hashlib
import json
import sys
import time

import torch

from common import ROOT
from common.timing import timed
from a3_finetune.data import dev_examples, prompt, test_examples, train_examples
from a3_finetune.generate import Generator, Out
from a3_finetune.grammar import Grammars
from a3_finetune.model import MODELS, load, tokenizer
from a3_finetune.schema import SEPARATORS, problems
from a3_finetune.train import CKPT, OUT

CACHE = ROOT / "cache" / "a3"
GREEDY_MAX, SAMPLE_MAX = 1024, 768
BUDGET = 16384                      # rows x (prompt + length cap) per greedy batch: ~2.3 GB of attention cache


def arg(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


def examples(run: str, split: str) -> list:
    if split == "dev":
        return dev_examples()
    if split == "test":
        return test_examples()
    names = set(json.loads((OUT / run / "config.json").read_text())["train_maps"])
    return [e for e in train_examples() if e.name in names]


def adapter_id(run: str, adapter: str) -> str:
    if adapter == "none":
        return "none"
    return hashlib.sha256((CKPT / run / adapter / "adapter_model.safetensors").read_bytes()).hexdigest()[:16]


def key(**parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


class Runner:
    def __init__(self, run: str, adapter: str, model_key: str):
        self.run, self.adapter, self.model_key = run, adapter, model_key
        self.aid = adapter_id(run, adapter)
        self.dir = CACHE / run / "answers"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tok = tokenizer(model_key)
        self._gen = None
        self._plen = {}

    def plen(self, ex) -> int:
        if ex.name not in self._plen:
            self._plen[ex.name] = len(self.tok(prompt(self.tok, ex.user), add_special_tokens=False)["input_ids"])
        return self._plen[ex.name]

    @property
    def gen(self) -> Generator:
        if self._gen is None:
            model = load(self.model_key, adapter=None if self.adapter == "none" else CKPT / self.run / self.adapter)
            model.eval()
            self._gen = Generator(model, self.tok, Grammars(self.tok, model.config.vocab_size))
        return self._gen

    def close(self):
        """Free the model, so the next adapter can be loaded in the same process."""
        import gc
        self._gen = None
        gc.collect()
        torch.cuda.empty_cache()

    def _key(self, ex, mode: str, **extra) -> str:
        return key(model=MODELS[self.model_key], adapter=self.aid, prompt=prompt(self.tok, ex.user), mode=mode,
                   grammar={"whitespace": False, "separators": SEPARATORS}, **extra)

    def _load(self, k):
        p = self.dir / f"{k}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def _save(self, k, rec):
        (self.dir / f"{k}.json").write_text(json.dumps(rec), encoding="utf-8")

    def _record(self, ex, mode, outs, prefix=(), **extra) -> dict:
        answers = [self.gen.answer(list(prefix) + o.tokens) for o in outs]
        return {"map": ex.name, "mode": mode, "adapter": self.adapter, "adapter_id": self.aid, "prefix": list(prefix),
                "outs": [{"tokens": o.tokens, "logps": [round(x, 5) for x in o.logps], "finished": o.finished,
                          "answer": a, "valid": a is not None and not problems(a, ex.s)} for o, a in zip(outs, answers)],
                **extra}

    def greedy(self, exs: list) -> dict:
        out, todo = {}, []
        for ex in exs:
            k = self._key(ex, "greedy", max_new=GREEDY_MAX)
            rec = self._load(k)
            if rec:
                out[ex.name] = rec
            else:
                todo.append((ex, k))
        todo.sort(key=lambda x: self.plen(x[0]))          # ascending, so each new map is the batch's longest
        batches, batch = [], []
        for ex, k in todo:
            if batch and (len(batch) + 1) * (self.plen(ex) + GREEDY_MAX) > BUDGET:
                batches.append(batch)
                batch = []
            batch.append((ex, k))
        if batch:
            batches.append(batch)
        for batch in batches:
            t0 = time.perf_counter()
            outs = self.gen.greedy_batch([e for e, _ in batch], GREEDY_MAX)
            for (e, kk), o in zip(batch, outs):
                rec = self._record(e, "greedy", [o])
                self._save(kk, rec)
                out[e.name] = rec
            print(f"greedy: {len(out)}/{len(exs)} maps ({len(batch)} in {time.perf_counter() - t0:.1f} s)", flush=True)
        return out

    def samples(self, exs: list, n: int, seed: int) -> dict:
        greedy = self.greedy(exs)
        out = {}
        for j, ex in enumerate(exs):
            g = greedy[ex.name]["outs"][0]
            k = self._key(ex, "sample", n=n, seed=seed, max_new=SAMPLE_MAX, greedy=g["tokens"])
            rec = self._load(k)
            if rec is None:
                t0 = time.perf_counter()
                prefix, outs = self.gen.samples(ex, Out(g["tokens"], g["logps"], g["finished"]), n, seed, SAMPLE_MAX)
                rec = self._record(ex, "sample", outs, prefix, n=n, seed=seed)
                self._save(k, rec)
                print(f"samples: {j + 1}/{len(exs)} {ex.name} ({time.perf_counter() - t0:.1f} s)", flush=True)
            out[ex.name] = rec
        return out


def export(run: str, tag: str, records: dict):
    path = OUT / run / f"{tag}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for name in sorted(records):
            f.write(json.dumps(records[name]) + "\n")
    print("exported", path)


def load_export(run: str, tag: str) -> dict:
    with gzip.open(OUT / run / f"{tag}.jsonl.gz", "rt", encoding="utf-8") as f:
        return {r["map"]: r for r in map(json.loads, f)}


def main():
    run, adapter, split = arg("run"), arg("adapter", "none"), arg("split", "dev")
    cfg = OUT / run / "config.json"
    model_key = arg("model") or (json.loads(cfg.read_text())["model"] if cfg.exists() else "qwen3-4b")
    r = Runner(run, adapter, model_key)
    exs = examples(run, split)
    if arg("maps"):
        exs = exs[:int(arg("maps"))]
    n = int(arg("samples", 0))
    mode = f"samples{n}" if n else "greedy"
    with timed("a3", f"generate:{run}:{adapter}:{split}:{mode}", maps=len(exs)) as row:
        torch.cuda.reset_peak_memory_stats()
        records = r.samples(exs, n, int(arg("seed", 0))) if n else r.greedy(exs)
        outs = [o for rec in records.values() for o in rec["outs"]]
        row.update({"answers": len(outs), "valid": sum(o["valid"] for o in outs),
                    "finished": sum(o["finished"] for o in outs)})
        print(row, flush=True)
    if "--export" in sys.argv:
        export(run, f"{adapter}_{split}_{mode}", records)


if __name__ == "__main__":
    main()
