"""Constrained decoding: each map's answer schema (a3_finetune/schema.py) as an xgrammar grammar.

The grammar allows exactly the training format: keys in schema order, separators "," and ":", no whitespace
(any_whitespace=False), so sampling at temperature 1 can't drift into free whitespace (D065). `Constrain` is a
`transformers` logits processor with one matcher per sequence; a matcher can start part-way through an answer
(the greedy turn types shared by the relation samples), so it is seeded with those tokens first.
"""
import json

import torch
import xgrammar as xgr
from transformers import LogitsProcessor

from a3_finetune.schema import SEPARATORS, schema


class Grammars:
    def __init__(self, tok, vocab_size: int):
        self.info = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=vocab_size,
                                                       stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>")])
        self.compiler = xgr.GrammarCompiler(self.info)
        self.vocab_size = vocab_size

    def compile(self, s) -> xgr.CompiledGrammar:
        return self.compiler.compile_json_schema(json.dumps(schema(s)), any_whitespace=False, separators=SEPARATORS)

    def matcher(self, s, prefix: list[int] = ()) -> xgr.GrammarMatcher:
        m = xgr.GrammarMatcher(self.compile(s))
        for t in prefix:
            if not m.accept_token(t):
                raise ValueError(f"prefix token {t} rejected by the grammar of {s.name}")
        return m

    def accepts(self, s, ids: list[int]) -> bool:
        """Whether a full answer (ending with the end-of-turn token) is a complete, valid answer for this map."""
        m = xgr.GrammarMatcher(self.compile(s))
        return all(m.accept_token(t) for t in ids) and m.is_terminated()


class Constrain(LogitsProcessor):
    """Masks every token the grammar forbids, for a batch whose rows each have their own matcher.

    It also records, per row, the log-probability of each chosen token under the constrained distribution
    (the step's scores after masking; no other warper runs at temperature 1, top-k 0, top-p 1 or greedy).
    Only the last step's distribution is held, so `output_scores` (all steps, ~5 MB each) isn't needed.
    """

    def __init__(self, matchers: list[xgr.GrammarMatcher], vocab_size: int):
        self.matchers = matchers
        self.mask = xgr.allocate_token_bitmask(len(matchers), vocab_size)
        self.start = None
        self.prev = None                                   # log-softmax of the previous step, [rows, vocab]
        self.logps = [[] for _ in matchers]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.start is None:
            self.start = input_ids.shape[1]
        else:
            chosen = input_ids[:, -1]
            lp = self.prev.gather(1, chosen[:, None])[:, 0].tolist()
            for i, m in enumerate(self.matchers):
                if not m.is_terminated():
                    self.logps[i].append(lp[i])
                    m.accept_token(int(chosen[i]))
        xgr.reset_token_bitmask(self.mask)                 # all allowed; finished rows stay unconstrained
        for i, m in enumerate(self.matchers):
            if not m.is_terminated():
                m.fill_next_token_bitmask(self.mask, i)
        xgr.apply_token_bitmask_inplace(scores, self.mask.to(scores.device))
        self.prev = torch.log_softmax(scores.float(), dim=-1)
        return scores
