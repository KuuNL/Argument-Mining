# A4 discussion: what a decision model can and can't do in argument mining

> **What this is:** why A4 (Jev, TypeSafe AI's "System One" decision model) performs as it does, against A1 (a trained pairwise encoder), A2 (Claude reading the whole map) and A3 (a fine-tuned Qwen3-4B). It also asks where a model like this is useful. How A4 works is in [A4-jev.md](A4-jev.md).
> **Evidence:**
> - [`../results/a4/discussion.json`](../results/a4/discussion.json): `python -m a4_jev.discuss`, a read-only error analysis of the 225 dev maps;
> - [`dev/dev_report.json`](../results/a4/dev/dev_report.json) and [`test_global.json`](../results/a4/test_global.json).
>
> **The whole project, end to end:** the [final report](05-final-report.md).
>
> As in the A3 discussion, systems are compared at **matched density** (each predicts as many relations as gold, 8.3 per map), so differences reflect the models, not their thresholds.

**In one line:** Jev reads each question literally and fast. That makes it the best of the four at finding support and its direction, and the worst at the annotators' conventions (restatement, concession), at $0.003 a map with no training.

## 1. Results

| | A1: encoder | A2: Claude, whole map | A3: fine-tuned 4B | **A4: Jev** |
|---|---|---|---|---|
| What it is | ModernBERT-base, trained on 1,250 maps | Sonnet 5.5, prompted, 4 examples | Qwen3-4B + LoRA, trained | Decision model, no training, no examples |
| How it decides | a probability per pair | one written list per map, 3 votes | one written list, 8 votes | a probability per typed question, two question forms averaged |
| **Dev ARI, honest** | 50.50 / 32.61 | **54.63 / 34.12** | 47.65 / 27.23 | 52.17 / 32.45 |
| **Test ARI** | 51.38 / 30.77 | **55.77 / 33.34** | 53.70 / 32.05 | 54.21 / 30.98 |
| Test Global | 72.01 / 57.53 | **73.67 / 58.27** | 73.17 / 58.17 | 73.42 / 57.63 |
| Per map | ~1 s, after 94 min of training | ~25 s, $0.18 API-equivalent | ~40 s, after 110 min of training | **0.8 s, $0.003** |

- **Jev ranks 2nd on both dev and test.** The order A1 < A4 < A2 is the same on both, and the dev → test change (+2.0) is close to A1's and A2's (about +1). A3's (+6) was the unstable one.
- **Both pre-registered success tests narrowly fail.** Against A1, +1.68 with an interval of [−0.20, +3.65]; added to A1 + A2, +0.39 general (+0.83 focused). A second identical dev pass scores 51.75 (+1.25 over A1), so the dev result is stable to about 0.4 (D082).

## 2. Where Jev is strong and where it is weak (dev, matched density)

| | A1 | A2 | **A4** |
|---|---|---|---|
| Inference found, right direction (of 1,060) | 301 | 396 | **471** |
| Inference recall, neighbouring turns (707) | 0.39 | 0.42 | **0.51** |
| Inference recall, 2–3 turns apart (252) | 0.11 | 0.32 | **0.35** (pair questions alone 0.40) |
| Inference recall, 4–10 turns apart (79) | 0.00 | 0.14 | **0.24** (pair questions alone 0.32) |
| Gold Inferences called Rephrase | 129 | 121 | **42** |
| **Rephrase found** (of 683) | 358 | **377** | 201 |
| Rephrase recall, many shared words (112) | **0.81** | 0.74 | 0.60 |
| Rephrase recall, no shared content words (102) | 0.23 | **0.43** | 0.16 |
| Gold Rephrases called Inference | 54 | 108 | **169** |
| Conflict found (of 130) | 19 | 23 | **29** |
| Conflict recall, different speakers (38) / same speaker (91) | 0.26 / 0.09 | 0.42 / 0.07 | **0.45 / 0.12** |
| Unrelated pairs given a relation (fewer is better) | 905 | **812** | 859, of which 707 are Inferences |
| ARI general at matched density | 50.0 | **54.3** | 51.1 |

## 3. Why

### Q1. Why does Jev find the most support, and get its direction right without reasoning?
- **Each question asks exactly one thing:** "is P a reason for Q, Q a reason for P, or neither?" Jev answers each against the whole map, with no list to write and no budget of relations to spend.
  - A2 writes one list and is told that annotators link sparingly, so it leaves out loose support.
  - A1 judges each pair from its training, and most of its training pairs are None.
- **Both directions are explicit options,** so direction is a choice between two readings of the same text. On pair questions it is right 91% of the time, and 87% overall. A2 discussion Q3 found direction comes from reading, not deliberating; Jev confirms it with no reasoning step at all.
- **Distance doesn't hurt it much.** Its recall 4–10 turns apart is 0.24 (pair questions 0.32), against A2's 0.14 and A1's 0. A question names its two propositions, so the distance between them doesn't matter to it.
- **The price is over-reading.** Of the relations Jev invents, 707 of 859 are Inferences, and its Inference probabilities are over-confident: at a stated ~0.9, only 35% are right (H1). Two propositions on the same topic often look like reason and claim.

### Q2. Why is Rephrase its weak spot?
- **Rephrase is the annotators' convention, not plain meaning.** In QT30 an elaboration, a narrowing or a sharper version of the same point counts as Rephrase. Read literally, a sharper version of a claim looks like support for it.
  - Jev turns 169 gold Rephrases into Inference, three times A1's count.
  - It misses most Rephrases that share some words (recall 0.25, against A1's and A2's 0.52–0.53).
- **The convention was stated, but not where Jev looks.** The guide in the state says "if the later proposition mainly repeats or elaborates the earlier point … it is Rephrase", the same sentence that helped A2. But Jev "answers the question you wrote" (TypeSafe's own list of weak spots): the decision is shaped by the options' short descriptions ("makes the same point again"), and the guide's caveat doesn't carry into the choice.
- **A worked example didn't help either** (+0.12 on the pilot, against A2's +4.5 from examples). An example map in the state is, to Jev, more content unrelated to the question, and TypeSafe says that lowers accuracy. Jev can't learn a convention from an example the way an LLM can.
- *Confidence: fairly high* for "Rephrase vs Inference is where it fails" (direct counts). *Moderate* for the cause: putting the full definition in every option (the planned `defs="full"` arm) would test it, and was dropped from scope.

### Q3. Why do two question forms work better than one?
- **Pair questions** (independent, one per pair) give the best direction (91%) but over-link: 1,049 invented Inferences.
- **Source questions** (one per proposition, its candidate targets competing) invent a third fewer relations (778 against 1,155; 40% fewer Inferences). But direction falls to 76%: asked "which proposition does P relate to", Jev often picks P's own premise. A rewording didn't change that (A4.2), so the form causes it, not the words.
- **Each finds relations the other misses.** Of the gold Inferences found by either, 276 are found by both, 141 only by pair questions and 118 only by source questions.
- **Averaging gets the best of both:** the competing form damps the pairs the independent form over-scores, and the independent form fixes the direction. Matched-density ARI goes from 47.2 (pair) and 46.1 (source) to 51.1 (both).
- *Lesson:* with a decision model, **how a judgement is decomposed into questions is the main design lever,** more than wording or examples.

### Q4. Did stating the concession rule help with Conflict?
- **A little.** Jev finds the most Conflicts of the four (29 of 130). Its within-speaker recall of 12% is the best of any system (A1 9%, A2 7%), but under the 15% bar (H3).
- **Across speakers it is best too** (0.45), as A2 was: there, attack means what it says.
- **Its Conflict probabilities are well calibrated** (ECE 0.013) and precise (0.52 at the tuned threshold), so the rare-label trap is under control: 8 maps get an invented Conflict, as for A1 and A2.
- **Concession is still mostly missed,** for the same reason as Rephrase: it is a convention against everyday reading. *Confound:* only A4's guide states the concession rule, so this measures the rule and the model together.

### Q5. Why does Jev fall behind on large maps?
- Honest per-map ARI, A4 − A1, by map size:
  - +4.1 at ≤ 10 propositions;
  - +1.5 at 11–16;
  - +0.4 at 17–25;
  - −5.4 at 26+ (6 maps).
  - Against A2: −1.5 → −11.2.
- **This matches TypeSafe's own warning:** accuracy falls as the state fills with content unrelated to the decision. In a large map most of the dialogue is unrelated to any one pair. A whole-map LLM (A2) uses the extra context; Jev is distracted by it.
- *Confidence: moderate;* there are few large maps. A per-pair window of the dialogue in the state would test it.

### Q6. Is Jev "calibrated", as advertised?
- **Partly.** Conflict and Rephrase probabilities are well calibrated (Rephrase better than A1's).
- **Inference is badly over-confident,** so the decision layer's thresholds were still needed: Inference 0.55, Conflict 0.75, Rephrase 0.30. Calibrated or not, a per-map macro-F1 metric needs tuned thresholds.
- **Answers aren't deterministic.** The same request gives the same choice 98% of the time, but probabilities can differ by up to 0.17. Option order matters more: forward and reversed orders agree on 91% of choices, so both orders are needed.

## 4. Where a decision model is useful in argument mining
- **As a no-training substitute for a trained encoder.** Relation scores at or above encoder level (+1.7 on dev, not significant; +2.8 on test) at $0.003 and under a second per map. It needs no GPU and no labelled pairs, only a one-page guide (whose base rates were measured on the corpus). Even with no labelled data at all (untuned argmax) it scores 50.84, against A1's 50.08 (D082). For a new corpus or annotation scheme without training data, this is the cheapest strong baseline we have.
- **For the parts of the task that are plain reading:** finding support and its direction, including between distant turns, and attacks between speakers.
- **Not as an addition to the encoder and the LLM:** adding it to an A1 + A2 average gained only +0.39 on dev (success test 2). The two already cover most of what it finds.
- **Not for convention-heavy labels** (Rephrase vs Inference, concession as Conflict): there an LLM applying definitions with examples (A2) or a model trained on the annotations (A1) does better.
- **Not without engineering in our code:** two question forms, two option orders, thresholds and consistency all live in the pipeline, not in the model.

## 5. Against the 2024 systems
- **Test ARI 54.21 / 30.98:** 2nd on general against the 2024 field (DFKI's DeBERTa pipeline 55.33), and above every other entry (Pungene 46.22, KnowComp 32.75, Turiya 30.75, Pokemon 30.64).
- **Global 73.42 / 57.63:** 1st against 2024's 67.05 / 48.25, with the shared ILO labeller.
- **2024 had nothing like it.** Every winning system was a trained pairwise encoder, and the generative attempts lost. A decision model sits between the two: pairwise like the encoders, but untrained; it reads the whole map like the LLMs, but writes nothing. Its result shows that 2024's main gap, context, can be closed without training.
- **2024's failures, for Jev:**
  - direction is fixed (87%);
  - Conflict is still missed (1 of 13 on test);
  - linked and convergent structure is not modelled, as for every system here.

## 6. Limits
- **Run-to-run noise is small but real:** two identical dev passes give 52.17 and 51.75 (D082). A4's edge over A1 (+1.3 to +1.7) is never significant.
- **Only one wording fix was tried** (A4.2), and the full-definition arm was dropped (D076). So "Jev can't apply the Rephrase convention" is not separated from "our options didn't state it".
- **The 11 test maps** move each system by several points per map: A4 is the best of the four on map 11 (79.3) and the worst on map 9 (32.8).
- *Measures:* matched density uses one threshold per system. "Neighbouring" means one turn apart. Word overlap is the Jaccard overlap of content words.
