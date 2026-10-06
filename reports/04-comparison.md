# 04 — Four approaches compared

> **What this is:** the project's comparison of its four approaches to DialAM-2024 and their combination: how well each does, at what cost and speed, and what each is good at. It also gives a critical answer to whether Jev (A4) is impressive.
> **Evidence:** every number comes from a committed file:
> - test: `results/a1/v1/test_global.json`, `results/a2/test_global.json`, `results/a3/test_global.json`, `results/a4/test_global.json`, `results/combination/test_global.json`;
> - dev: `results/a4/dev/dev_report.json`, `results/a4/robust.json`, `results/a4/discussion.json`, `results/a3/v1/discussion.json`, `results/combination/dev_report.json`;
> - timing: `results/timing.csv`.
>
> Decisions D050, D064, D072, D079–D085. Per-approach reports: [A1](A1-cross-encoder.md), [A2](A2-whole-map-llm.md) · [discussion](A2-discussion.md), [A3](A3-finetuned-llm.md) · [discussion](A3-discussion.md), [A4](A4-jev.md) · [discussion](A4-discussion.md).

## In one minute
- **All four approaches beat the 2024 winner on Global** (67.05 / 48.25), and so does their combination. Most of that margin is the shared, trained ILO labeller (92.6 test ILO, against 2024's 81.2), so the systems are told apart by **relations (ARI)**.
- **On relations, dev (225 maps) ranks A2 > A4 ≈ A1 > A3.** A2 beats A1 by +4.1 and A4 by +2.5, both significant. A4 and A1 are within noise of each other (+1.3 to +1.7 for A4), and A3 is significantly below A1 (−2.9).
- **The best result is a combination:** the mean of A1 + A2 + A4 pair scores, pre-registered, gives **test Global 74.51 / 59.47** and ARI 56.39 / 34.65, above every single system and above the best 2024 relation score. A1 + A2 alone scored higher on test (75.17 / 60.32), so Jev's share of the combination is about zero.
- **Jev is the efficiency winner.**
  - It reaches the trained encoder's accuracy with no training, no GPU and no labelled data: 50.84 untuned, against A1's 50.08.
  - It costs about 1/60 of Claude's price per map and is about 30× faster, at 2.5 points lower ARI.
  - What makes it work is our question design, not the model alone; its weakness is the annotators' conventions.

## 1. Results

**Test (11 maps, each system scored once; F1 general / focused):**

| System | ARI | ILO | Global |
|---|---|---|---|
| Best 2024 entry per column | 55.33 / 35.89 | 81.17 / 72.09 | 67.05 / 48.25 |
| A1 · trained cross-encoder | 51.38 / 30.77 | 92.64 / 84.28 | 72.01 / 57.53 |
| A2 · Claude, whole map | 55.77 / 33.34 | 91.56 / 83.20 | 73.67 / 58.27 |
| A3 · fine-tuned Qwen3-4B | 53.70 / 32.05 | 92.64 / 84.28 | 73.17 / 58.17 |
| A4 · Jev, typed questions | 54.21 / 30.98 | 92.64 / 84.28 | 73.42 / 57.63 |
| **A1 + A2 + A4, averaged (pre-registered)** | **56.39 / 34.65** | 92.64 / 84.28 | **74.51 / 59.47** |
| A1 + A2, averaged (reported beside it) | 57.69 / 36.37 | 92.64 / 84.28 | 75.17 / 60.32 |

**Dev (225 maps, honest: thresholds cross-fitted by episode). This is the reliable comparison:**

| | Dev ARI | Paired bootstrap, ARI general (95% interval) | Dev Global |
|---|---|---|---|
| A1 | 50.50 / 32.61 | — | 70.29 / 56.63 |
| A2 | **54.63 / 34.12** | A2 − A1 **+4.14 [2.21, 6.12]** | 72.53 / 57.63 |
| A3 | 47.65 / 27.23 | A3 − A1 −2.85 [−4.7, −1.0] | 68.90 / 53.95 |
| A4 | 52.17 / 32.45 (second pass 51.75) | A4 − A1 +1.68 [−0.20, 3.65]; A4 − A2 −2.46 [−4.19, −0.73] | 71.16 / 56.56 |
| A1 + A2 + A4 | **55.51 / 37.01** | | **72.83 / 58.84** |

- **Test is too small to rank systems.** One test map moves a mean by about 9%. A3's dev → test jump (+6.1) is within 11-map variation (D072). Rankings come from dev; test numbers are for the 2024 comparison.
- **Run-to-run noise was measured only for Jev:** two identical dev passes differ by 0.4 (D082). A2 averaged 3 answers and A3 8 samples; A1 is deterministic.

## 2. Cost, speed and what each needs

| | A1 encoder | A2 Claude | A3 fine-tuned LLM | A4 Jev |
|---|---|---|---|---|
| Training | 94 min on an 8 GB GPU | none | 110 min on an 8 GB GPU | none |
| Labelled data used | 1,250 maps (~105k pairs) | 4 retrieved maps per prompt; a prompt revised on 20–50 training maps | 1,231 maps | a one-page guide (base rates from the corpus); a 50-map pilot to choose the view |
| Per map | ~1 s | ~25 s (3 answers) | ~22–40 s | **0.8 s** |
| A full dev pass (225 maps) | ~3.5 min (estimated from 0.94 s per map) | 82 min of calls, over ~11.5 h of Pro usage limits | 84 min | **69 s** |
| $ per map | $0 (local GPU) | $0.18 API-equivalent ($0 on Pro) | $0 (local GPU) | **$0.003** |
| Spent in the project | $0 | $0 ($57 API-equivalent) | $0 | $1.70 |
| Needs | a GPU and training data | a frontier-LLM subscription or API | a GPU, training data and a fine-tuning stack | an API key |

## 3. What each approach is good at (dev, matched density: every system predicts as many relations as gold)

| | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| Inference found, right direction (of 1,060) | 301 | 396 | 223 | **471** |
| Inference recall, neighbouring turns (707) | 0.39 | 0.42 | 0.25 | **0.51** |
| Inference recall, 2–3 turns apart (252) | 0.11 | 0.32 | 0.16 | **0.35** |
| Inference direction right | 86% | 84% | 83% | **87%** |
| Rephrase recall, neighbouring turns (~486) | 0.61 | 0.57 | **0.69** | 0.30 |
| Rephrase recall, no shared content words (102) | 0.23 | **0.43** | 0.41 | 0.16 |
| Conflict recall, different speakers (38) | 0.26 | 0.42 | 0.13 | **0.45** |
| Conflict recall, same speaker (91) | 0.09 | 0.07 | 0.11 | **0.12** |
| Unrelated pairs given a relation (fewer is better) | 905 | 812 | **805** | 859 |
| ARI general at matched density | 50.0 | **54.3** | — | 51.1 |

*A3's figures come from `results/a3/v1/discussion.json` (its matched-density ARI wasn't computed there); the other columns come from `results/a4/discussion.json`.*

**Which system is best at what:**
- **A4 (Jev)** finds the most support relations, at any distance, with the best direction. It also finds the most Conflicts, though every system misses most of them.
- **A2 (Claude)** is the most balanced, and the only one that recognises restatements in new words.
- **A3** is best at restatements between neighbouring turns, but loses Inference to Rephrase.
- **A1** is strong on restatements that share words.
- **No system solves Conflict within one speaker's talk** (7–12%).

## 4. Is Jev impressive? A critical answer
- **Yes, on accuracy per unit of cost.**
  - With no training, no GPU and no examples, Jev matches a fine-tuned encoder on relations: dev +1.3 to +1.7 (not significant) and test +2.8.
  - It does so even with no labelled data at all: untuned argmax gives 50.84, against A1's 50.08.
  - Against the only other no-training approach (Claude), it gives up 2.5 ARI on dev (significant) and 1.6 on test, at about 1/60 of the price and 1/30 of the time per map, with no usage limits.
  - All of A4 (probe, pilot, two dev passes, test) took **under 4 minutes of model time and $1.70**.
- **But the speed is impressive against LLMs, not against encoders.** Once trained, A1 is about as fast (~1 s per map) and free to run. Jev's advantage over A1 is that it needs no training data and no GPU, not that it runs faster.
- **The accuracy comes from the question design.**
  - Asked one way (one question per pair), Jev scores 50.5 honest, A1's level.
  - Asked one competing question per proposition, it scores 46.3.
  - Only the average of the two reaches 52.2.
  - It is cheap because its tokens are cheap, not because it reads less: a map costs ~65k input tokens of questions, about twice what A2 reads (3 calls of ~10k).
- **Vendor claims, tested:**
  - *Fast:* confirmed. 0.75–0.9 s per request, even with 350 questions.
  - *Schema-safe:* confirmed. 702 of 702 answers valid.
  - *Calibrated:* partly. Conflict and Rephrase yes; Inference over-confident (ECE 0.070 against A1's 0.018).
  - *Not mentioned by the vendor:* answers vary between identical requests (88.5% of decisions identical), and the order of options changes 9% of choices.
- **It is limited by literal reading.** The annotators' conventions are where it loses: an elaboration counts as Rephrase, a concession as Conflict. It is also weaker on large maps (−5 against A1 at 26+ propositions), and a worked example in the state doesn't help it (+0.1, against A2's +4.5).
- **It adds little in combination:** +0.4 on dev, −1.3 on test, on top of A1 + A2.
- **Verdict:** for argument mining, a decision model is a strong *no-data* baseline. It is the cheapest way to encoder-level relation mining and the best support finder of the four. It isn't a replacement for a frontier LLM where the labels follow conventions, and it isn't a useful third member next to an encoder and an LLM.

## 5. The research question: which post-2024 method class works best, at what cost?

| 2024 failure | Best fix here | Evidence |
|---|---|---|
| 1. Relations are the bottleneck | Whole-map frontier LLM (A2), then A1 + A2 combined | Dev ARI 54.6 single, 55.5 combined (A1 50.5); test 55.8 / 56.4 above 2024's 55.3 |
| 2. Conflict is missed | Not fixed by any method | Recall 12–22% overall; 7–12% within one speaker; test 0–2 of 13 found |
| 3. Direction errors | Every method; best A4, then A1 / A2 | 83–87% right at matched density; reasoning added nothing (A2) |
| 4. No global consistency (over-linking) | Joint decisions: A2 / A3's single list, A4's competing questions | Invented relations 805–859 against A1's 905; A4's source view alone cuts its own by a third |
| 5. LLMs lost to encoders | Reversed only for a frontier LLM | A2 > A1 (+4.1); a fine-tuned 4B LLM still loses (A3 −2.9) |
| (ILO) | A trained labeller with one-to-one decoding and rare-type thresholds | Test ILO 92.6 against 2024's 81.2; LLM turn types hurt (A2 −1.1) |

- **Best accuracy:** a frontier LLM reading the whole map (A2), averaged with a trained encoder (A1 + A2 or A1 + A2 + A4).
- **Best accuracy per cost:** Jev (A4), at encoder level for ~$0.003 per map with no training.
- **Worst trade-off:** fine-tuning a small open LLM (A3). It costs as much training as A1 and scores below it on dev.

## 6. Limits
- **Small test set:** 11 maps, with large per-map swings (each system has maps at 30 and at 75+ ARI). Ranks come from dev.
- **One configuration per approach,** with ablations cut to the core for A3 and A4 (user decisions D071, D076). A1's and A2's ablations are in their reports.
- **The ILO labeller is shared,** so Global differences are mostly ARI differences. "No training" for A2 and A4 refers to relations only.
- **Contamination isn't ruled out** for A2 and A4, whose training data is unknown. Their dev → test changes (+1.1, +2.0) look like A1's (+0.9).
- **The combination is the simplest average.** Confidence routing (A2 only on uncertain pairs) and weighting were not tried.
