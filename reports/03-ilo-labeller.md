# Phase 3 — The trained ILO labeller: findings

> **Status:** passed (2026-10-01). Branch `phase3-ilo`, commits `c34e68c` (3.1–3.2) and `66df3ad` (3.3–3.5).
> **Sources:** every number below comes from a committed file in [`../results/ilo/`](../results/ilo/). Decisions are D025 and D028–D036 in [`../decisionlog.md`](../decisionlog.md).
> **Scope:** Task B (ILO) for every approach, plus the turn↔proposition links that the ARI approaches need on test maps.

## At a glance

| # | Finding | Number | Consequence |
|---|---|---|---|
| 1 | The trained labeller works | Dev ILO **89.89 general / 80.38 focused** (cross-fitted) | Task B is in good shape: well above the baselines below |
| 2 | Linking beats text similarity | 98.3% of links found, at 99.6% precision | Test maps get reliable positions and speakers for ARI |
| 3 | The gap is in *typing*, not linking | Fixing the minority types is worth up to **+6.9 general / +9.8 focused** ILO; linking under +1 | The next ILO gain comes from rare and Assertive/Rhetorical types |
| 4 | 22 maps have no illocution layer at all | 19 train + 3 dev, the late "giant" batches | Left out of ILO training and scoring; kept for ARI |
| 5 | Similarity misses context-rebuilt turns | "Yes" or "it stinks" whose proposition came from context | Turns borrow candidates from their neighbours (99.6% recall) |
| 6 | Fixed-size batches overflowed the 8 GB card | 7.7 GB peak and 4k tokens/s; after the fix 4.3 GB and 26k tokens/s | Every later training run uses token-budget batches |
| 7 | Flex attention is slower on real batches | 2.7k vs 14.2k tokens/s (standard attention) | Standard attention (SDPA) is the default |
| 8 | Inference is cheap | 0.21 s per map | No cost concern for the final run |

---

## 1. What the labeller does

ILO scores every (locution, proposition) pair. In practice that means **linking each dialogue turn to at most one proposition, and saying how it was said**. The plan first assumed simple rules would do. EDA showed otherwise: "?" is almost absent, and the rare types weigh as much as Asserting in macro-F1. So the user chose a trained model (D025).

```mermaid
flowchart LR
  M[("Map in test format<br/>turns + propositions")] --> C["Candidates<br/>similarity: own 5,<br/>prev 5, next 3"]
  C --> X["Input text<br/>turn + proposition<br/>+ 2 turns either side"]
  X --> B["ModernBERT-base<br/>11 probabilities"]
  B --> T["Temperature<br/>T = 1.05"]
  T --> H["One-to-one matching<br/>link if 1 - P(None) at least 0.10"]
  H --> Y["Type thresholds<br/>questions 0.3-0.5"]
  Y --> O[("YA links<br/>L to I with type")]
  classDef data fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
  classDef step fill:#dcfce7,stroke:#16a34a,color:#14532d
  classDef model fill:#fde68a,stroke:#b45309,color:#78350f
  class M,O data
  class C,X,T,H,Y step
  class B model
```
*Figure 1: the labeller end to end. Blue = data, green = deterministic steps (tuned on dev), yellow = the trained model.*

## 2. Candidates (3.1, D028)

The model only scores candidate pairs; everything else is None. Text similarity alone isn't enough:

| Candidates per turn | Gold links kept | Pairs |
|---|---|---|
| Most similar only (K = 1) | 94.3% | 22k |
| K = 3 | 97.5% | 66k |
| K = 6 | 98.8% | 131k |
| **Own 5 + previous turn's 5 + next turn's 3** | **99.6%** | 206k |

- **Why similarity misses:** 371 of the 433 misses at K = 3 are short turns whose proposition was rebuilt from context. "Exactly." becomes the claim agreed with; "it stinks" becomes what "it" refers to.
- **The fix:** borrow the candidates of neighbouring turns. That rescues 183 links, mostly rare types.
- **Recall per type:** every type is ≥ 98.3%, except Agreeing (93.7%) and Disagreeing (94.7%).
- **The labeller never uses node IDs or node order** (proposed rule D029). In 74% of links the proposition's ID is the turn's ID + 2, an annotation-tool artefact.

## 3. Data (3.2, D030–D031)

One example per candidate pair:
```
A: Jeane Freeman: Exactly.
B: Jeane Freeman can enforce it when people can fly into England … by xxx
   [SEP] before: Michael Forsyth: How are you going to enforce that? / …   [SEP] after: …
```

| | Train | Dev |
|---|---|---|
| Maps (with an illocution layer) | 1,231 | 225 |
| Examples / true links | 158,864 / 11.0% | 29,506 / 10.9% |
| Tokens: median / max | 108 / 300 | 108 / 243 |

- **22 maps have no illocution layer.** 1,614 pairs looked like links (turn text ≈ proposition text) but had no gold link, and 1,553 of them sit in 22 maps with *zero* linked turns: the late re-annotation batches (nodesets 24807–25475), which are also EDA's giant maps. They carry relations but were never annotated for illocutions. They're excluded from ILO training and dev scoring and kept for ARI. Test maps are fully annotated.
- **Rare types are very rare:**

  | Type | Train | Dev |
  |---|---|---|
  | Asserting | 15,721 | 2,984 |
  | Pure Questioning | 1,002 | 161 |
  | Assertive Questioning | 216 | 19 |
  | Rhetorical Questioning | 192 | 29 |
  | Default Illocuting | 123 | 7 |
  | Agreeing | 89 | 11 |
  | Challenging | 38 | 2 |
  | Disagreeing | 18 | 2 |
  | Arguing | 10 | 2 |
  | Restating | 4 | 1 |

## 4. Training (3.3, D032–D034)

**Setup:**
- ModernBERT-base, max 320 tokens, batches capped at 4,096 padded tokens (≤ 64 pairs).
- AdamW at learning rate 5e-5, 6% warm-up then linear decay, bf16.
- Class weights (most frequent / count)^0.5, capped at 30.
- Dev evaluated every half epoch; stop after 2 evaluations without improvement.

**Sanity checks before training:**
- **Decoder ceiling:** decoding the *gold* labels scores 98.63 / 94.73, with 99.16% linking recall. The gap is candidate misses plus turns that express two propositions.
- **Majority baseline:** all-None scores 38.21 / 0.
- **Memorisation:** the model learns 72 examples perfectly in 100 steps.

**Training curve** (dev, simple decode):

| Epoch | ILO general / focused | Link recall / precision |
|---|---|---|
| **0.5** | **89.46 / 78.06** (kept) | 97.5% / 99.7% |
| 1.0 | 86.41 / 72.31 | 96.3% / 99.7% |
| 1.5 | 88.05 / 73.18 | 96.3% / 99.8% |

**Cost:** 19.5 min in total (13 min per epoch), 22.5k tokens/s, 4.35 GB peak.

**Engineering lessons:**

| Problem | Evidence | Fix |
|---|---|---|
| GPU memory overflow | Fixed 32-pair batches reached about 9.6k tokens; 7.72 GB of 7.9 used, and 3.7–4.9k tokens/s as the driver spilled into RAM | Token-budget batches: 4.33 GB and 26.3k tokens/s (6× faster) |
| Flex attention recompiles | Every new batch shape triggers a recompile: 2.7k vs 14.2k tokens/s with standard attention | SDPA is the default; flex only pays off at fixed shapes |
| Tool time limit | The first run was killed at step 600 by the 10-min background limit | Long runs go detached, monitored through the progress log |

## 5. Decision layer (3.4, D035)

Fitted on dev, and scored honestly by **leave-one-episode-out** (tune on 4 dev episodes, score the 5th):
- **Temperature 1.05:** the model is already well calibrated.
- **Link threshold 0.10** on 1 − P(None). Low, because one-to-one matching already blocks most false links. 4 of 5 folds agree.
- **Type thresholds** on P(type | link): Pure and Assertive Questioning 0.5, Rhetorical 0.3, **rare types 0**. Grouping the rare types or not made no difference.
- **Gain over simple decoding:** +0.43 general / +2.32 focused, cross-fitted.

## 6. Results on dev (3.5, D036)

| Dev ILO F1 | General | Focused |
|---|---|---|
| Majority (no illocutions) | 38.21 | 0.00 |
| Text similarity, one-to-one, all Asserting | 75.99 | 59.38 |
| Model, simple decode | 89.46 | 78.06 |
| **Model, tuned (cross-fitted)** | **89.89** | **80.38** |
| Ceiling of this design | 98.63 | 94.73 |

**Linking:** 98.27% recall and 99.62% precision, against 97.43% / 98.68% for similarity with the same matching.

**Per type** (pooled over dev):

| | Asserting | Pure Q. | Rhetorical Q. | Assertive Q. | Six rare types |
|---|---|---|---|---|---|
| Gold / predicted | 2,992 / 2,988 | 162 / 176 | 29 / 14 | 19 / 6 | 26 / **0** |
| F1 | 0.98 | 0.87 | 0.51 | 0.40 | 0 |

**For reference only:** 2024's best ILO was 81.2 / 72.1, but on the 11 *test* maps, so it isn't directly comparable.

## 7. Where the remaining points are

Re-scoring the tuned predictions with one group of types corrected to gold (no retraining; `python -m ilo.evaluate --headroom`):

| Dev ILO F1 if… | General | Focused | Maps affected |
|---|---|---|---|
| As is | 90.16 | 80.67 | — |
| Rare types correct | 93.04 | 85.13 | 22 of 225 |
| Assertive + Rhetorical correct | 94.17 | 85.89 | 40 of 225 |
| Both | **97.07** | **90.49** | — |
| Every linked pair typed correctly | 97.87 | 90.61 | — |

**Linking is nearly solved; typing the minority types holds almost all of the rest.** These are upper bounds. Global is the mean of ARI and ILO, so the full gap is worth about 3.5 Global points (general).

**Likely cause:** the kept checkpoint is from half an epoch, before the rare types were learned. Later checkpoints were judged by the simple decode, which punishes rare-type mistakes hardest, and were discarded.

## 8. Lessons for the other approaches
- **Reuse `common/finetune.py`** (A1.2): token-budget batches, SDPA, and detached launch for runs over 10 minutes.
- **Select checkpoints by the tuned metric, not the simple decode,** and keep each evaluation's dev probabilities. This applies even more to A1, whose Conflict class is rare.
- **The labeller's links give each test proposition its turn and speaker.** A1.1 should build test-format pair inputs from these predicted links, not gold.
- **Rare classes on dev are tiny** (1–12 examples), so thresholds are cross-fitted by episode. The same will hold for Conflict in A1.
- **22 maps lack illocutions but have relations.** ILO ignores them; ARI decides on them in A1.1 (D022).

## 9. Open items
- **Rare-type iteration** (optional, about 30–40 min of GPU time): choose checkpoints by the tuned score, keep all dev probabilities, try a gentler learning rate and weight cap. Target: part of the +6.9 / +9.8 headroom.
- **D029** (no node-ID shortcuts) still needs a user decision. It affects every approach.

## Reproduce
```bash
# from the repo root, inside WSL (see CLAUDE.md §8)
python -m ilo.candidates          # 3.1  recall table
python -m ilo.data                # 3.2  cache/ilo/*.jsonl, results/ilo/dataset.json      (~4 min)
python -m ilo.train               # 3.3  checkpoints/ilo/, results/ilo/train_*.json(l)    (~20 min, run detached)
python -m ilo.tune                # 3.4  results/ilo/thresholds_shared.json               (~2 min)
python -m ilo.evaluate            # 3.5  results/ilo/dev_report.json
python -m ilo.evaluate --headroom #      results/ilo/headroom.json
```
