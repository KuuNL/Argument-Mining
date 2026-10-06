# A2 discussion: what a whole-map LLM does better and worse than a trained encoder

> **What this is:** a short discussion of approach A2's results, set against A1 and the project's goals. It asks *why* each system wins where it does. How A2 works is in [A2-whole-map-llm.md](A2-whole-map-llm.md).
> **Evidence:**
> - Test scores: [`../results/a2/test_global.json`](../results/a2/test_global.json).
> - Error analysis: [`../results/a2/discussion.json`](../results/a2/discussion.json), from `python -m a2_llm.discuss`.
>   - It covers the **225 dev maps** (1,873 gold relations), with both systems' thresholds fitted on all of dev.
>   - The "why" questions use dev because it is 20× larger than test and reading it opens nothing sealed.
>
> **Published page:** https://claude.ai/artifact/B7wFwx2cGTbfNmGBXmnUCg (the same content).

**In one line:** the encoder learns *the annotators*; the LLM applies *the definition*. Where the two agree, the LLM wins; where the annotation follows habit, the encoder does.

## 1. The task and the two systems
- **The input.** A debate from BBC *Question Time* arrives as a **map**: its **turns** (who said what, in order) and the **propositions** those turns express.
- **Task A, relations (ARI).** For every pair of propositions: does one **support** (Inference), **attack** (Conflict) or **restate** (Rephrase) the other, and in which direction?
- **Task B, turn types (ILO).** How each turn expresses its proposition: Asserting, Pure / Assertive / Rhetorical Questioning, and so on.
- **Scoring.** F1 is computed per map and then averaged.
  - *General* F1 counts every pair; *focused* F1 counts only pairs that have a true relation.
  - Every label present in a map counts equally. So one invented rare label can cut a map's score by a quarter: the **rare-label trap**.

An illustrative map, written the way A2 reads it:
```text
T1  Host: should NHS staff get a pay rise
    [P1] NHS staff should get a pay rise
T2  Minister: yes, they deserve one
    [P2] NHS staff deserve a pay rise
T3  Minister: they worked through the pandemic
    [P3] NHS staff worked through the pandemic

To predict:  P2 Rephrase P1 · P3 Inference P2 · T1 Pure Questioning, T2 and T3 Asserting
```
*The turns and propositions are given. The last line is what a system must predict. An answer counts as a Rephrase of its question.*

| | A1: cross-encoder | A2: whole-map LLM |
|---|---|---|
| Model | ModernBERT-base (~150M parameters), fine-tuned on 1,250 maps | Claude Sonnet 5.5, not trained; 4 similar solved maps in the prompt |
| Reads | two propositions plus the whole dialogue | the whole map |
| Decides | each pair separately, as a probability per label; pairs more than 10 turns apart are skipped | all relations at once, as one list; asked 3 times, and the votes become scores |
| Then | per-type thresholds tuned on dev (the same decision layer for both) | |
| Turn types | the trained ILO labeller (also ModernBERT) | the same labeller, which the LLM may override |
| Cost | 94 min of training on an 8 GB GPU; 1.2 s per map | no training; ~25 s per map; $0 on a Claude Pro plan ($0.18 per map at API prices) |

## 2. Results

| Test F1, general / focused | Relations (ARI) | Turn types (ILO) | Global |
|---|---|---|---|
| Best 2024 entry per column | 55.33 / 35.89 | 81.17 / 72.09 | 67.05 / 48.25 |
| A1 + labeller | 51.38 / 30.77 | **92.64 / 84.28** | 72.01 / 57.53 |
| **A2 + labeller + LLM types** | **55.77** / 33.34 | 91.56 / 83.20 | **73.67 / 58.27** |

- **A2 is the project's best system so far,** and first against the 2024 field. Its general ARI is the best of any system, 2024 included.
- **The gain is precision on support:** Inference precision rises from 0.31 to 0.48 on test (dev: 0.27 → 0.39).
- **It lost 1.1 ILO** by letting the LLM change turn types: 3 changes, all wrong. With the labeller's types it would have scored 74.20 / 58.81. That was checked afterwards and was not submitted.

**Against the five failures of 2024** (the project's starting point):

| 2024 failure | A2 |
|---|---|
| 1. Relations are the bottleneck | Better: the best general ARI (55.8), but still 36 points below turn types |
| 2. Conflict is missed | **Not fixed:** 12% recall on dev and 2 of 13 on test, the same as A1 |
| 3. Direction errors | Better: 85% right on dev (A1: 81%) and 90% on test (A1: 78%) |
| 4. No global consistency | Over-linking is fixed: 7.9 relations per map against 8.3 in gold (A1: 10.9). But it over-corrects: a premise almost never supports two conclusions |
| 5. LLMs lost to encoders | Reversed for relations; still true for turn types |

## 3. Where A2 did well and where it did badly (dev)

| A2 did better | A1 | A2 |
|---|---|---|
| Unrelated pairs given a relation (fewer is better) | 1,278 | **697** |
| … of which invented Inferences | 901 | **405** |
| Gold Conflicts labelled as support | 35 of 130 | **9** |
| Recall: relations 4 or more turns apart (174) | 0.11 | **0.27** |
| Recall: Rephrases that share no content words (102) | 0.25 | **0.43** |
| Recall: Conflicts between different speakers (38) | 0.18 | **0.32** |
| ARI general gain over A1, in every map-size band | | +3.1 to +5.8 |

| A2 did worse | A1 | A2 |
|---|---|---|
| Recall: Inferences between neighbouring turns (707) | **0.51** | 0.40 |
| Recall: Rephrases that share many words (211) | **0.79** | 0.68 |
| Propositions supporting 2+ conclusions (gold: 5.7%) | 16.8%, too many | 0.7%, too few |
| Turn types (test ILO, general) | **92.64** | 91.56 |
| *Neither:* Conflicts within one speaker's talk (91, 70% of all) | 5% | 3% |

## 4. Questions

### Q1. Why does the LLM invent fewer relations?
- **A1 judges each pair alone** (about 100 per map) and never sees its other decisions.
  - Any two related-sounding propositions close together get a fair chance, so errors add up with the number of pairs.
  - Between neighbouring turns, A1 predicts **1,942** relations where gold has 1,279 (+52%). A2 predicts 1,208.
- **A2 writes one list for the whole map,** so candidates compete. Once P3 is the reason for P2, it rarely becomes a reason for P1 as well.
- **A2 also follows a global instruction.** The prompt says half of all propositions start no relation. An encoder can only absorb that as a per-pair prior.
- **Takeaway:** the LLM's structural advantage is deciding jointly. Seeing more context isn't it, because A1 reads the whole dialogue too.

### Q2. Then why does the encoder find more support between neighbouring turns?
- **Annotators often record a speaker's consecutive statements as a chain of support,** even when the link is loose. For example, *"since the protocol came in, there have been difficulties with steel"* supports *"we have worked to sort out issues with steel"*.
- **A1 learned that habit from 1,250 maps:** neighbouring turns, the same speaker and the same topic mean Inference. It gains recall and pays in precision.
- **A2 applies the definition:** is one statement a *reason* for the other? Its prompt also says Inference needs a different point. Being stricter, it finds fewer, and more of those are right.
- **Takeaway:** the encoder learns the *annotators*; the LLM applies the *definition*.
  - Where the two agree, the LLM wins. Where the annotation follows habit, the encoder wins.
  - A2's precision gain and its recall loss come from the same choice.

### Q3. Why does the LLM get direction right more often?
- **Dialogue order gives no clue:** 53% of Inferences point back in time and 47% forward. Which statement is the premise lives in the meaning.
- **An LLM reads that directly** ("X, because Y"). The encoder has to learn direction as two separate labels (A→B, B→A) from a few thousand examples.
- **Reasoning doesn't change it:** 85% right with or without thinking first. The model gets direction from reading, not from deliberating.

### Q4. Why do both still miss Conflict?
- **It isn't a threshold problem.**
  - Across its 3 answers, A2 listed only 32 of the 130 gold Conflicts.
  - A1 gave only 40 of them a probability above 0.2.
  - The signal is absent, not cut off.
- **Most annotated Conflicts aren't disagreements between people.** 70% of them (91 of 130) lie within one speaker's own talk: concessions and contrasts.
  - For example, *"it is legitimate to think about the future"* is attacked by the same speaker's *"Anand Menon would hate it if … we lost sight of the present"*.
  - Only 15% of these have a word like "but" or "however", even in the spoken turn. The contrast is implicit.
  - Each system finds 3–5% of them.
- **Why the LLM misses them.** In ordinary language an attack is someone disagreeing, so a speaker's "yes, but" reads as one coherent position. IAT marks any incompatible pair, whoever holds it.
- **Why the encoder misses them.** Conflict is 10% of training relations, and these pairs look like support: neighbouring turns, the same speaker, the same topic. A1 calls 27% of gold Conflicts Inference (A2: 7%).
- **Across speakers the LLM does better** (32% recall against 18%), because there "attack" means what it says.
- **Takeaway:** this is a problem of convention. More reasoning won't teach it. Examples of concessive Conflict, or fine-tuning (A3), might.

### Q5. Why do they find different Rephrases?
- **When the two propositions share many words, A1 is better** (0.79 against 0.68).
  - A cross-encoder matches words across its two inputs very well, and it was trained to call near-copies Rephrase.
  - The LLM often reads two similar-looking propositions as different points, one narrowing or building on the other, and calls that Inference or nothing.
- **When they share no content words, A2 is better** (0.43 against 0.25). It recognises the same meaning in different words.
- **The Rephrase / Inference boundary is the hardest for both,** at about 220 confusions each.
- **The IAT rule that an answer rephrases its question** is learned equally well by both (64% and 68%). It's in A2's prompt and examples, and in A1's training data.

### Q6. Why did reasoning add only about 1 point?
- **On dev, thinking before answering added +0.8 / +1.0,** and left direction and Conflict unchanged. A 20-map pilot had suggested +11 focused.
- **The errors are mostly about the annotation convention, not logic.** Thinking longer about whether X supports Y doesn't tell the model that annotators would call it Rephrase, or that they count a concession as Conflict.
- **What helped carried the convention:** worked examples (+4.5 general from 0 to 4) and a prompt rewritten from observed errors.

### Q7. Why did the LLM's turn types backfire?
- **The LLM recognises questions well:** 128 Pure Questions right on dev.
- **But IAT splits questions three ways by convention, and the errors fall on those lines.** The LLM called all three of these test turns Pure Questioning:
  - *"Do you want another go, Maggie"* and *"What about the Prime Minister in particular"* are **Assertive** Questioning in gold (the moderator steering);
  - *"I just wondered why has it taken so long"* is **Asserting** (an indirect question).
- **The labeller learned the annotators' prior from 1,250 maps:** Asserting unless sure. It misses rare types but never invents them. Under the rare-label trap, inventing a type a map doesn't have costs more than missing one.
- **A +0.3 dev gain from about 12 changes was too thin** a basis to adopt them.

### Q8. Do map size or memorisation explain the gain?
- **Size: no.** A2 leads in every size band, from the smallest maps (+5.7) to the largest (+5.8, 26+ propositions). A whole map is only about 1k tokens.
- **Memorisation: no sign.** From dev to test, A2 moved +1.1 and A1 +0.9.

## 5. Encoder or LLM: which is better at what

| Ability | Better | Why |
|---|---|---|
| Not inventing relations | LLM | It decides all pairs together and follows a stated base rate |
| Direction of support | LLM | It reads which statement is the reason |
| Telling attack from support | LLM | It understands contrast in meaning |
| Rephrase in different words; distant relations | LLM | It reads meaning, not surface; and A1 skips pairs more than 10 turns apart |
| Loose support between neighbouring turns | Encoder | It learned the annotators' habit |
| Rephrase with heavy word overlap | Encoder | It matches words |
| Turn types | Encoder | It learned the conventions and a cautious prior |
| Scores for thresholds | Encoder | A continuous probability per pair. A2's 3 votes × 3 confidence levels give a coarse 10-step scale, mostly 0 |
| Conflict within one speaker | Neither | The convention contradicts everyday meaning, and the label is rare |
| Speed and running cost | Encoder | 1.2 s against ~25 s per map, though the LLM needs no training |

## 6. What this means for the project
- **Combine the two.** Their errors differ: picking the right system for each pair would score 66.1 / 46.0 on dev, against A2's 54.6 / 34.1. A natural split:
  - A1's probabilities propose candidates (recall);
  - the LLM checks them and sets the direction (precision);
  - the labeller keeps the turn types.
- **A3 (a fine-tuned open LLM)** tests Q2, Q4 and Q7 directly: can an LLM that *learns the annotators* keep its own strengths and gain the encoder's?
- **A4 (Jev)** gives whole-map context and probabilities but has no reasoning step. Q3 and Q6 suggest that's less of a handicap than we assumed, because direction came from reading, not thinking.
- **Conflict remains the open problem for every approach.** The next lever is teaching the convention (concession examples), not more reasoning.

**What happened next** (details in [`04-comparison.md`](04-comparison.md) and the [final report](05-final-report.md)):
- **Combining:** a plain average of A1 + A2 pair scores beat A2 alone (dev ARI 55.12; test Global 75.17 / 60.32). It captures only a small part of the 66.1 oracle, and the propose-and-check split above was not built.
- **Q2's premise reverses at matched density.** When every system predicts as many relations as gold, A1's recall of neighbouring Inferences is 0.39 against A2's 0.42 (A3 report, H1). A1's lead in Q2 came from its looser thresholds, not from learning a habit A2 lacks.
- **A3** answered Q2 and Q4 negatively. Fine-tuning a 4B LLM on the annotations taught the format and the density, but not the neighbour habit or concessive Conflict, and it scored below both A1 and A2 on dev. Q7 (turn types) was not tested.
- **A4** confirmed Q3: Jev got Inference direction right 87% of the time with no reasoning step at all.
- **Conflict** stayed unsolved by every approach: within-speaker recall 7–12% at matched density.

*Limits:*
- Thresholds were fitted on all of dev, so absolute dev numbers are slightly optimistic for both systems.
- Word overlap (the Jaccard overlap of content words) and distance (in turns, from the labeller's links) are simple measures.
- The answers to "why" are explanations that fit the evidence. Only reasoning and examples were tested by switching them off.
