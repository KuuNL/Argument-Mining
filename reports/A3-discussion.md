# A3 discussion: why a fine-tuned open LLM lands where it does

> **What this is:** why A3 (a fine-tuned Qwen3-4B) came out as it did against A1 (a trained pairwise encoder), A2 (Claude reading the whole map) and the 2024 shared-task systems. How A3 works is in [A3-finetuned-llm.md](A3-finetuned-llm.md).
> **Evidence:**
> - [`../results/a3/v1/discussion.json`](../results/a3/v1/discussion.json): `python -m a3_finetune.discuss`, a read-only error analysis of the 225 dev maps;
> - [`dev_report.json`](../results/a3/v1/dev_report.json);
> - [`../results/a3/test_global.json`](../results/a3/test_global.json) and [`test_posthoc.json`](../results/a3/test_posthoc.json).
>
> **Published page:** https://claude.ai/artifact/VJCZmPoyRdCkDuhfeMGAuh (the same content).
>
> The "why" questions use dev, which is 20× larger than test. All three systems are compared at **matched density** (8.3 relations per map each), so recall differences reflect the models, not their thresholds.

**In one line:** fine-tuning taught a 4B model the annotators' *format and density*, but not their *judgement*. It is as cautious as A2 but finds fewer real arguments than A1, and it often turns an Inference into a Rephrase.

## 1. Results

| | A1: encoder | A2: Claude, whole map | **A3: fine-tuned Qwen3-4B** |
|---|---|---|---|
| Model | ModernBERT-base, ~150M | Claude Sonnet 5.5, prompted | Qwen3-4B in 4-bit + LoRA (33M trained) |
| Learns from | ~105k labelled proposition pairs | 4 retrieved example maps per prompt | 1,231 whole-map answers |
| Decides | each pair, as a probability | one list per map, 3 answers voted | one list per map, 8 samples voted |
| **Dev ARI, honest** | 50.50 / 32.61 | **54.63 / 34.12** | 47.65 / 27.23 |
| **Test ARI** | 51.38 / 30.77 | **55.77 / 33.34** | 53.70 / 32.05 |
| Test Global | 72.01 / 57.53 | **73.67 / 58.27** | 73.17 / 58.17 |
| Cost | 94 min training; ~1 s per map | $0 on a subscription ($57 API-equivalent); ~25 s per map | 110 min training; ~40 s per map; $0 |

- **Dev and test disagree on A1 vs A3.** On dev (225 maps) A3 is below both. On test (11 maps) it lands between them, after a +6.1 jump against A1's +0.9 and A2's +1.1.
- **The dev ranking is the reliable one.** A3's test mean lies within the range of random 11-map dev subsets (a test-or-higher mean in 7% of them). A3 beats A1 on 5 of 11 test maps and A2 on 5. Its votes and greedy configurations differ by up to 21 points on a single map.
- *Contamination:* not supported by this check, but not excluded, since no memorisation probe was run.

## 2. What A3 learned, and what it didn't (dev, matched density)

| | A1 | A2 | A3 |
|---|---|---|---|
| Unrelated pairs given a relation (fewer is better) | 905 | 812 | **805** |
| Inference recall, neighbouring turns (707) | 0.39 | **0.42** | 0.25 |
| Inference recall, 2–3 turns apart (252) | 0.11 | **0.32** | 0.16 |
| Rephrase recall, neighbouring turns (485) | 0.61 | 0.57 | **0.69** |
| Rephrase recall, no shared content words (102) | 0.23 | **0.43** | 0.41 |
| Conflict recall, different speakers (38) | 0.26 | **0.42** | 0.13 |
| Conflict recall, same speaker (91) | 0.09 | 0.07 | **0.11** |

**What the 1,060 gold Inferences become:**

| | A1 | A2 | A3 |
|---|---|---|---|
| Found, right direction | 301 | **396** | 223 |
| Called **Rephrase** | 129 | 121 | **252** |
| Reversed | 50 | 74 | 45 |
| Missed (no relation) | 577 | 462 | 530 |

- **Learned:**
  - the format: every one of 2,000+ dev and test samples is valid;
  - the annotators' density: the greedy answer lists 9.6 relations per map, against 8.3 in gold;
  - A2-level caution: at the same density it invents no more relations than A2;
  - Rephrase: the best of the three between neighbouring turns, and nearly A2's level for paraphrases that share no words.
- **Not learned:** the Inference judgement. A3 calls twice as many gold Inferences "Rephrase" as A1 or A2 do. It falls away fast with distance, and it misses most Conflicts between different speakers, which A2 finds best.

## 3. Why

### Q1. Why does A3 turn Inferences into Rephrases?
- **The Rephrase / Inference boundary is the hardest call in the corpus.** About 220 confusions each for A1 and A2 (A2 discussion, Q5), and it rests on meaning. Rephrase means *the same point again*; Inference needs *a different point given as a reason*.
- **A small model can learn that two neighbouring, same-speaker propositions are related without learning which kind of relation it is.** A3 found the relation and then took the label that is easier to justify from the surface. Its strong Rephrase recall, and its Inference recall falling by distance, both fit this.
- **A2 resolves the boundary from the definition, and A1 from ~105k pairs with explicit labels.** A3 had neither: no definition in the prompt, and only ~10k listed relations to learn from.
- *Confidence: fairly high.* The confusion count is direct evidence; the cause (model capacity against supervision) is not separated, because the planned ablations weren't run.

### Q2. Why didn't training on the annotators teach A3 their neighbour habit, when it taught A1?
- **The shape of the supervision differs, not the data.** All three systems saw the same annotations; what each learned from them differs.
  - **A1:** every candidate pair is a training example with its own label, None included (~105k decisions), plus class weights and tuned thresholds. "Neighbouring and same speaker" becomes a learned prior on every pair.
  - **A3:** each map is one sequence, and the model learns which relation to write next. Unrelated pairs are never shown as negatives; they are simply not written.
  - About half of the answer tokens are turn types and fixed syntax, which are easy, so most of the loss is spent on tokens that don't decide relations. The loss kept falling for 3 epochs while the relation metric peaked at 2.
- **1,231 sequences is little data for a generative model.** Rephrase, the surface-friendly label, was learned; Inference, the semantic one, wasn't.
- *Confidence: moderate.* This is consistent with the loss / metric divergence and the per-type pattern, but untested: no learning-curve or data ablation was run.

### Q3. Why do the votes help A2 but not A3?
- **A3's samples barely agree with each other.**
  - A sample and the greedy answer share only 21% of their relations (Jaccard).
  - 457 of the greedy answer's 2,159 relations appear in none of the 8 samples.
  - A single sample at temperature 1 scores 37–42, against the greedy answer's 49.5.
- **The relations are written as one list in a fixed order, so the choices are coupled.** A different early choice changes what follows. When the model is unsure, the list's most likely path (greedy) and the frequency of each relation across samples diverge. A2's larger model has a sharper distribution, so its 3 answers were similar and their votes meaningful.
- **The signal is there, but diluted.** 65% of gold relations appear in at least one of A3's samples (A2: 53% in at least one of its 3). But a gold relation's mean vote share (0.35) is close to a wrong relation's (0.20), and wrong listings outnumber right ones 4 to 1. The thresholds can't separate them cleanly.
- *Not position in the answer:* A3's Inference recall *rises* from the first third of a map to the last (0.16 → 0.24), so later relations are not degraded by long answers.
- *Confidence: high* for "the samples are too diffuse". The coupling explanation is likely but untested; a lower sampling temperature, the obvious check, was not run.

### Q4. Why does the gap to A2 grow with distance and map size?
- **Distance:**
  - Inference recall 2–3 turns apart: A3 0.16, A2 0.32.
  - Rephrase 8+ turns apart: A3 0.14, A2 0.39.
- **Map size,** honest per-map ARI A3 − A2: −5.4 at 11–16 propositions, −7.6 at 17–25, −24.0 on the 6 maps with 26+.
- **Linking distant statements needs reading the whole map for meaning:** the reasoning a frontier model does well and a 4B model does poorly. Larger maps also mean more coupled decisions in one answer (Q3).
- *Confidence: moderate;* there are few long-range and large-map cases.

### Q5. Did fine-tuning help at all?
- **Probably a lot, but it wasn't measured.** The untrained model wrote empty lists for 3 of 5 maps (A3.1). The zero-shot ablation (H6) was not run, so the size of the gain is unknown.
- **On test, A3 beats 2024's best encoder on Global,** at $0 and on one 8 GB GPU. That is far beyond the 2024 generative attempts (§4).

### What we can't separate
- **Model size, fine-tuning and 4-bit storage are confounded.** Qwen3-8B couldn't train on the 8 GB card, so size was never varied.
- **Not run (user decision, D071):** zero-shot (H6), LLM turn types (H5), giant-map Conflict examples (H4), and sampling temperature.
- **Single seed and single run;** 11 test maps.

## 4. Against the 2024 shared-task systems
Test ARI from the overview paper (Table 1) and ours, on the same 11 maps and scoring logic:

| System | Method | Test ARI general / focused |
|---|---|---|
| DFKI-MLST (2024, 1st) | DeBERTa-v3 n-ary pair classifier; locution and proposition text combined | 55.33 / 30.40 |
| Pungene | BERTScore alignment, then BERT pair classifiers, built up as a pipeline | 46.22 / 20.51 |
| KnowComp | Prompt-based fine-tuning of DeBERTa, three-stage pipeline | 32.75 / 9.06 |
| Turiya | RoBERTa + biaffine classifiers; also prompted a generative LLM with the whole map | 30.75 / 6.65 |
| Pokemon | Two-step filter + classifier (DeBERTa + RoBERTa-MNLI); tried LLaMA, kept encoders | 30.64 / **35.89** |
| RoBERTa-large baseline | Pair classifier | 26.46 / 22.80 |
| Ours: A1 | ModernBERT pair classifier + whole dialogue + thresholds | 51.38 / 30.77 |
| **Ours: A3** | **Fine-tuned 4B LLM writing the whole map's relations + votes + thresholds** | 53.70 / 32.05 |
| Ours: A2 | Claude Sonnet 5.5 writing the whole map's relations + votes + thresholds | **55.77** / 33.34 |

- **2024's two generative attempts lost badly.**
  - Turiya prompted an LLM with the whole map and scored near the majority baseline (30.75 general, 6.65 focused).
  - Pokemon tried LLaMA and fell back to encoders.
  - The overview paper concluded that the encoder pipelines worked.
- **A3 is the same idea as Turiya's (generate every relation from the whole map), but trained on the annotations,** constrained to a valid format, and put through a tuned decision layer.
  - That closes most of the gap: on test it ranks 2nd in general ARI against the 2024 field, and 1st on Global, as A1 and A2 do.
  - The changes that made the difference are the ones 2024 lacked: supervision, a grammar so no answer is lost, and thresholds against the rare-label trap.
- **But on dev it still trails our own encoder (A1).** 2024's finding, that encoders beat LLMs on ARI, holds for a mid-size fine-tuned LLM in 2026. It breaks only with a frontier model applying definitions (A2).
- **Conflict remains everyone's failure.** The paper notes that systems missed most Conflicts and invented some. A3 found 0 of 13 on test, and A1 and A2 2 of 13 each.
- **Direction and structure:**
  - The paper highlights direction errors in divergent and convergent arguments. A3's test Inference direction is 89% right (A1 78%, A2 90%), and it lists one relation per premise like everyone else: linked arguments are still scored as convergent ones.
  - On dev its direction (83% at matched density) is close to A1's and A2's.

## 5. What this means for the project
- **A3 is not added to the combination.** Adding its pair scores to an A1 + A2 average lowers dev ARI (55.12 → 54.24). The plain A1 + A2 average is already above A2 alone, so the Combination phase should start there. *(It did: the final combination averages A1, A2 and A4, without A3, D083–D084.)*
- **For a whole-map LLM, model competence matters more than learning the convention.** At this size and data volume, fine-tuning taught the format and the base rate but not the judgement. A2 (a larger model, a definition and 4 examples) beat it.
- **If A3 were revisited:**
  - greedy or low-temperature decoding instead of temperature-1 votes;
  - an ordering-free output (score each candidate pair from the fine-tuned model's probabilities);
  - a larger model on a bigger GPU.
- **Methodological lessons:**
  - Compare systems at matched density: at their own thresholds, A1 and A2 looked far more different than they are.
  - Don't read much into an 11-map test that disagrees with a 225-map dev set.

*Limits:* matched density uses one threshold for all types per system. "Neighbouring" means one turn apart. Word overlap is the Jaccard overlap of content words. The "why" answers fit the evidence, but only the parts named as measured were tested.
