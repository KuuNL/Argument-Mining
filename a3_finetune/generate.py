"""Constrained generation for one map: a greedy answer, then relation samples that share its turn types (D065).

1. `greedy`: the most likely full answer under the map's grammar. Its turn types are A3's turn types, and its
   relations are the greedy answer (checkpoint selection in A3.3, an ablation in A3.5).
2. `samples`: the greedy answer is cut just before `"relations"`, and n continuations are sampled from there
   at temperature 1 (no top-k / top-p) with a fixed seed. Only the relations part is sampled, which halves the
   generated text.
Every chosen token's log-probability is kept (grammar.Constrain), for the token-probability score.
"""
import json
from dataclasses import dataclass

import torch

from a3_finetune.data import END, prompt
from a3_finetune.grammar import Constrain, Grammars

CUT = '"relations"'


@dataclass
class Out:
    tokens: list[int]          # generated tokens, up to and including the end-of-turn token
    logps: list[float]         # log-probability of each generated token before the end token
    finished: bool             # the end-of-turn token was reached within the length cap


class Generator:
    def __init__(self, model, tok, grammars: Grammars):
        self.model, self.tok, self.g = model, tok, grammars
        self.end = tok.convert_tokens_to_ids(END)

    def prompt_ids(self, ex) -> list[int]:
        return self.tok(prompt(self.tok, ex.user), add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def _run(self, rows: list[list[int]], matchers, sample: bool, max_new: int, seed: int | None) -> list[Out]:
        """rows: one prompt per matcher, left-padded to a common length."""
        width = max(len(r) for r in rows)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.end
        x = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=self.model.device)
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=self.model.device)
        proc = Constrain(matchers, self.g.vocab_size)
        kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
        if seed is not None:
            torch.manual_seed(seed)
        seqs = self.model.generate(x, attention_mask=mask, max_new_tokens=max_new, logits_processor=[proc],
                                   eos_token_id=self.end, pad_token_id=pad, **kw)
        outs = []
        for i, row in enumerate(seqs[:, width:].tolist()):
            n = row.index(self.end) + 1 if self.end in row else len(row)
            outs.append(Out(row[:n], proc.logps[i], self.end in row))
        return outs

    def greedy(self, ex, max_new: int = 1024) -> Out:
        return self.greedy_batch([ex], max_new)[0]

    def greedy_batch(self, exs: list, max_new: int = 1024) -> list[Out]:
        """Greedy answers for several maps in one left-padded batch."""
        return self._run([self.prompt_ids(e) for e in exs], [self.g.matcher(e.s) for e in exs], False, max_new, None)

    def cut(self, tokens: list[int]) -> int:
        """The longest token prefix of a greedy answer that ends before its "relations" key."""
        text = self.tok.decode(tokens, skip_special_tokens=True)
        stop = text.find(CUT)
        k = 0
        while k < len(tokens) and len(self.tok.decode(tokens[:k + 1], skip_special_tokens=True)) <= stop:
            k += 1
        return k

    def samples(self, ex, greedy: Out, n: int, seed: int, max_new: int = 768) -> tuple[list[int], list[Out]]:
        k = self.cut(greedy.tokens)
        prefix = greedy.tokens[:k]
        first = self.g.matcher(ex.s, prefix)
        matchers = [first] + [first.fork() for _ in range(n - 1)]
        return prefix, self._run([self.prompt_ids(ex) + prefix] * n, matchers, True, max_new, seed)

    def answer(self, tokens: list[int]) -> dict | None:
        try:
            return json.loads(self.tok.decode(tokens, skip_special_tokens=True))
        except json.JSONDecodeError:
            return None
