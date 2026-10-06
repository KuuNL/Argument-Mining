# Phase 3 — The trained ILO labeller: findings

> **Status:** passed (2026-10-01). **Run in use: v1** ([`../results/ilo/selected.json`](../results/ilo/selected.json)). A second run, v2, was trained and evaluated but not adopted.
> **Branch / commits:** `phase3-ilo`: `c34e68c` (3.1–3.2), `66df3ad` (3.3–3.5), `2d70813` (iteration v2).
> **Sources:** every number below comes from a committed file in [`../results/ilo/`](../results/ilo/): `dataset.json` for the data, and `v1/` and `v2/` for each run (training log, summary, thresholds, dev report, headroom). Decisions are D025 and D028–D039 in [`../decisionlog.md`](../decisionlog.md).
> **Scope:** Task B (ILO) for every approach, plus the turn↔proposition links that the ARI approaches need on test maps.
> **Test result (scored with A1, D050):** ILO **92.64 / 84.28** on the 11 test maps, perfect on 7 of them, against 2024's best of 81.17 / 72.09. Every system uses its links, and all but A2's submission (which let Claude override 3 types) use its turn types, so it is behind most of the project's Global margin over 2024 ([final report](05-final-report.md)).

## At a glance

| # | Finding | Number | Consequence |
|---|---|---|---|
| 1 | The trained labeller works | Dev ILO **90.08 general / 80.64 focused** (v1, cross-fitted) | Task B is in good shape: +14 general over text similarity and +52 over predicting nothing |
| 2 | Linking is nearly solved | 98.3% of links found, at 99.6% precision | Test maps get reliable positions and speakers for ARI |
| 3 | Typing the minority types holds the rest | Correct Assertive/Rhetorical and rare types would reach 97.1 / 90.5 | That is where ILO's remaining points are |
| 4 | Undertraining was *not* the cause | v2 (3 full epochs, gentler learning rate) scored 89.74 cross-fitted, and its 7 Agreeing guesses on dev were all wrong | v1 kept. Rare types need a different design, or stay switched off |
| 5 | 22 maps have no illocution layer | 19 train + 3 dev (the late "giant" batches) | Excluded from ILO training and scoring; kept for ARI |
| 6 | Text similarity misses context-rebuilt turns | e.g. "Exactly." or "it stinks" | Candidates borrowed from neighbouring turns: 99.6% recall |
| 7 | Fixed-size batches overflowed the 8 GB card | 7.7 GB and 4k tokens/s, versus 4.3 GB and 26k tokens/s with token budgets | Every training run uses token-budget batches |
| 8 | Flex attention is slower on real batches | 2.7k vs 14.2k tokens/s with standard attention | Standard attention (SDPA) is the default |
| 9 | It's cheap | Training takes 20–42 min; inference 0.21 s per map | No cost concern for the final run |

---

## How Phase 3 ran

```mermaid
flowchart LR
  S1["3.1 Candidates<br/>which pairs to score"] --> S2["3.2 Dataset<br/>inputs + labels"]
  S2 --> S3["3.3 Train v1<br/>first try killed; batching fixed"]
  S3 --> S4["3.4 Decision layer<br/>temperature + thresholds"]
  S4 --> S5["3.5 Evaluate v1<br/>gate passed"]
  S5 --> S6["3.6 Iteration v2<br/>rare types"]
  S6 --> D{"v2 beats v1 by<br/>more than 0.5?"}
  D -->|"no: 89.74 vs 90.08"| K(["Keep v1"])
  classDef step fill:#dcfce7,stroke:#16a34a,color:#14532d
  classDef end_ fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
  class S1,S2,S3,S4,S5,S6 step
  class K end_
```
*Figure 1: the steps of Phase 3 and the final decision.*

| Step | What happened | Decision |
|---|---|---|
| 3.1 | Similarity alone keeps 98.8% of links at best; borrowing neighbours' candidates reaches 99.6% | D028; D029 (no node-ID shortcuts) proposed |
| 3.2 | 22 maps turned out to have no illocution layer; rare types are very rare | D030, D031 |
| 3.3 | First run killed by the tool's time limit and overflowing GPU memory; fixed with token budgets; v1 trained detached | D032–D034 |
| 3.4 | Temperature and thresholds tuned on dev, scored honestly by leaving one episode out | D035 |
| 3.5 | v1 evaluated against baselines: gate passed; minority types weak | D036 |
| 3.6 | v2: longer, gentler training, checkpoint chosen by the tuned score. Not better; v1 kept | D038, D039 |

## 1. What the labeller does

ILO scores every (locution, proposition) pair. In practice that means **linking each dialogue turn to at most one proposition, and saying how it was said**. The plan first assumed simple rules would do. EDA showed otherwise: "?" is almost absent, and the rare types weigh as much as Asserting in macro-F1. So the user chose a trained model (D025).

```mermaid
flowchart LR
  M[("Map in test format<br/>turns + propositions")] --> C["Candidates<br/>similarity: own 5,<br/>prev 5, next 3"]
  C --> X["Input text<br/>turn + proposition<br/>+ 2 turns either side"]
  X --> B["ModernBERT-base<br/>11 probabilities"]
  B --> T["Temperature<br/>v1: 1.05"]
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
*Figure 2: the labeller end to end. Blue = data, green = deterministic steps (tuned on dev), yellow = the trained model.*

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
- **No node IDs:** the labeller never uses node IDs or node order (proposed rule D029). In 74% of links the proposition's ID is the turn's ID + 2, an annotation-tool artefact.

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

## 4. Training run v1 (3.3, D032–D034)

**Setup** (recipe `v1` in `ilo/train.py`):
- ModernBERT-base, max 320 tokens, batches capped at 4,096 padded tokens (≤ 64 pairs).
- AdamW at learning rate 5e-5, 6% warm-up then linear decay, bf16.
- Class weights (most frequent / count)^0.5, capped at 30.
- Dev evaluated every half epoch; stop after 2 evaluations without improvement on the *simple* decode (one-to-one matching, most probable type, no thresholds).

**Sanity checks before training:**
- **Decoder ceiling:** decoding the gold labels scores 98.63 / 94.73, with 99.16% linking recall. The gap is candidate misses plus turns that express two propositions.
- **Majority baseline:** all-None scores 38.21 / 0.
- **Memorisation:** the model learns 72 examples perfectly in 100 steps.

**Curve** (dev, simple decode):

| Epoch | ILO general / focused | Link recall / precision |
|---|---|---|
| **0.5** | **89.46 / 78.06** (kept) | 97.5% / 99.7% |
| 1.0 | 86.41 / 72.31 | 96.3% / 99.7% |
| 1.5 | 88.05 / 73.18 | 96.3% / 99.8% |

**Cost:** 19.5 min (13 min per epoch), 22.5k tokens/s, 4.35 GB peak.

**Engineering lessons** (these apply to every later training run):

| Problem | Evidence | Fix |
|---|---|---|
| GPU memory overflow | Fixed 32-pair batches reached about 9.6k tokens; 7.72 GB of 7.9 used, and 3.7–4.9k tokens/s as the driver spilled into RAM | Token-budget batches: 4.33 GB and 26.3k tokens/s (6× faster) |
| Flex attention recompiles | Every new batch shape triggers a recompile: 2.7k vs 14.2k tokens/s with standard attention | SDPA is the default; flex only pays off at fixed shapes |
| Tool time limit | The first run was killed at step 600 by the 10-min background limit (`v1/train_log_run1_killed.jsonl`) | Long runs go detached, monitored through the progress log |

## 5. Decision layer (3.4, D035)

Fitted on dev, and scored honestly by **leave-one-episode-out** (tune on 4 dev episodes, score the 5th):
- **Temperature** (one number that softens or sharpens all probabilities): 1.05 for v1, already well calibrated.
- **Link threshold** on 1 − P(None): **0.10** for v1. It's low because one-to-one matching already blocks most false links. The search range was widened twice, from 0.20 to 0.05 to 0.01, because the optimum kept landing on the floor.
- **Type thresholds** on P(type | link): Pure and Assertive Questioning 0.5, Rhetorical 0.3, rare types 0. Grouping the rare types or not made no difference.

## 6. Results for v1 on dev (3.5, D036)

| Dev ILO F1 | General | Focused |
|---|---|---|
| Majority (no illocutions) | 38.21 | 0.00 |
| Text similarity, one-to-one, all Asserting | 75.99 | 59.38 |
| Model, simple decode | 89.46 | 78.06 |
| **Model, tuned (cross-fitted)** | **90.08** | **80.64** |
| Ceiling of this design | 98.63 | 94.73 |

**Linking:** 98.27% recall and 99.62% precision, against 97.43% / 98.68% for similarity with the same matching.

**Per type** (pooled over dev):

| | Asserting | Pure Q. | Rhetorical Q. | Assertive Q. | Six rare types |
|---|---|---|---|---|---|
| Gold / predicted | 2,992 / 2,988 | 162 / 176 | 29 / 14 | 19 / 6 | 26 / **0** |
| F1 | 0.98 | 0.87 | 0.51 | 0.40 | 0 |

**For reference only:** 2024's best ILO was 81.2 / 72.1, but on the 11 *test* maps, so it isn't directly comparable. On those same test maps, the labeller later scored 92.64 / 84.28 (A1.7, D050).

## 7. The rare-type iteration: v2 (D038–D039)

**Why:** v1's kept checkpoint came from half an epoch, and v1 never predicted any rare type. v1's probabilities hinted at undertraining: Default Illocuting got probability about 0.001 despite 123 training examples, and Assertive/Rhetorical were often the second choice.

**What changed** (recipe `v2`; data, model, weights and batching unchanged):
- all 3 epochs;
- learning rate 3e-5 instead of 5e-5;
- **every checkpoint judged by its *tuned* score** (about 13 s per evaluation);
- every evaluation's dev probabilities kept.

| v2 epoch | 0.5 | 1.0 | 1.5 | **2.0** | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| Tuned general / focused | 89.06 / 80.36 | 89.41 / 80.12 | 89.93 / 79.99 | **90.39 / 81.97** | 89.66 / 80.43 | 89.39 / 80.09 |
| Temperature | 1.10 | 1.15 | 1.40 | 1.20 | 1.55 | 1.65 |

**Cost:** 42.1 min (14 min per epoch). The rising temperature shows the model growing overconfident with longer training.

**Comparison** (both runs tuned on the same, widened grid):

| Dev ILO | v1 | v2 |
|---|---|---|
| Tuned on all of dev (optimistic) | 90.16 / 80.67 | 90.40 / 82.29 |
| **Cross-fitted (honest)** | **90.08 / 80.64** | 89.74 / 81.29 |
| Linking recall / precision | 98.27 / 99.62% | 98.61 / 99.59% |
| Assertive / Rhetorical Q. F1 | 0.40 / 0.51 | 0.46 / 0.56 |
| Rare-type predictions | none | none: tuning switches them off |

- **Rare types:** with suppression off, v2 predicts **Agreeing 7 times on dev, all wrong**, and still never Default Illocuting. The tuning therefore disables them. Predicting them would add a zero-scoring class to those maps (the rare-label trap).
- **Decision:** the stop rule (cross-fitted general must beat v1 by more than 0.5) failed by −0.34, so **v1 stays selected**. v2 is slightly better on focused and on questions, but all v1/v2 differences are within dev's noise.
- **What this teaches:** the gap is not undertraining. This design (one pair at a time, ±2 turns of context, 89 Agreeing examples) can't learn Agreeing or Default Illocuting well enough to beat the rare-label penalty.

## 8. Where the remaining points are

Re-scoring the tuned predictions with one group of types corrected to gold (no retraining; `python -m ilo.evaluate --run=<run> --headroom`):

| Dev ILO F1 if… | v1 | v2 | Maps affected |
|---|---|---|---|
| As is | 90.16 / 80.67 | 90.40 / 82.29 | — |
| Rare types correct | 93.04 / 85.13 | 93.31 / 86.88 | 22 of 225 |
| Assertive + Rhetorical correct | 94.17 / 85.89 | 93.79 / 86.77 | 40 of 225 |
| Both | **97.07 / 90.49** | **96.73 / 91.47** | — |
| Every linked pair typed correctly | 97.87 / 90.61 | 97.90 / 92.07 | — |

Linking costs under 1 point; minority-type typing holds about 7. These are upper bounds. Global is the mean of ARI and ILO, so the whole gap is worth about 3.5 Global points (general).

## 9. Lessons for the other approaches
- **Reuse `common/finetune.py`** (A1.2): token-budget batches, SDPA, and detached launch for runs over 10 minutes.
- **Choose checkpoints by the tuned metric and keep every evaluation's probabilities.** It's cheap, the selection is honest, and it matters most for rare classes such as A1's Conflict. Recalibrate each checkpoint, because the temperature drifts.
- **Use the labeller's predicted links on test maps:** they give each test proposition its turn and speaker. A1.1 builds test-format inputs from them, not from gold.
- **Rare classes on dev are tiny** (1–12 examples here), so thresholds are cross-fitted by episode and gains under about 0.5 are noise.
- **Not predicting a rare class can be correct.** When a model can't predict it reliably, the tuned layer switching it off beats guessing.
- **22 maps lack illocutions but have relations.** ILO ignores them; ARI decides on them in A1.1 (D022).

## 10. Open items
- **Rare types** (not planned now): add transition-anchored Agreeing / Disagreeing / Challenging examples as training signal; give reply turns the turn they answer as explicit context; or revisit after A1, if ILO becomes the bottleneck.
  - *Outcome:* ILO never became the bottleneck, so the labeller was not retrained. A2 let Claude override its turn types: +0.3 on dev, but −1.1 on test (3 changes, all wrong; D063–D064). A3, A4 and the combination keep the labeller's types.
- ~~**D029** (no node-ID shortcuts) still needs a user decision.~~ Approved on 2026-10-01; every approach follows it.

## Reproduce
```bash
# from the repo root, inside WSL (see CLAUDE.md §8)
python -m ilo.candidates                    # 3.1  recall table
python -m ilo.data                          # 3.2  cache/ilo/*.jsonl, results/ilo/dataset.json      (~4 min)
python -m ilo.train --run=v1                # 3.3  checkpoints/ilo/v1/, results/ilo/v1/            (~20 min, run detached)
python -m ilo.tune --run=v1                 # 3.4  results/ilo/v1/thresholds_shared.json           (~2 min)
python -m ilo.evaluate --run=v1             # 3.5  results/ilo/v1/dev_report.json
python -m ilo.evaluate --run=v1 --headroom  #      results/ilo/v1/headroom.json
python -m ilo.train --run=v2                # 3.6  the same for v2                                  (~42 min, run detached)
```
