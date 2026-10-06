# Reports

One findings report per phase, written when the phase closes. Together they are the source for the final write-up.
Decisions and phase status live in [`../decisionlog.md`](../decisionlog.md); raw numbers live in [`../results/`](../results/) and [`../eda/findings.json`](../eda/findings.json).

| Phase | Report | Headline |
|---|---|---|
| 0 · Environment | [decision log](../decisionlog.md) D005–D011, D032–D033 | ML runs in WSL2 (Windows Smart App Control blocks torch). ModernBERT-base trains at about 26k tokens/s in 4.3 GB with token-budget batches. |
| 1 · Foundation | [decision log](../decisionlog.md) D012–D016, D027 | The scorer port equals the official scripts (688/688). Episode-grouped split 1,250 / 228 with 0 leaks. |
| 2 · EDA | [`../eda/README.md`](../eda/README.md) | 98% of relations lie within 8 turns. Inference direction is a coin flip (53/47). A whole map fits in under 1k tokens. "?" is absent. |
| A1 · Cross-encoder (ARI) | [`A1-cross-encoder.md`](A1-cross-encoder.md) | **Final test: Global 72.01 / 57.53, 1st against the 2024 field** (best 67.05 / 48.25). The win comes from ILO (92.6 / 84.3); ARI 51.4 / 30.8 is about 2024 level (2nd). Ablations: context, hints and pruning each help focused. |
| A2 · Whole-map LLM (ARI) | [`A2-whole-map-llm.md`](A2-whole-map-llm.md) | **Final test: Global 73.67 / 58.27, above A1 (72.01 / 57.53) and 1st against the 2024 field.** ARI 55.77 / 33.34: the best general ARI against 2024 (+4.4 over A1), mostly from Inference. Claude Sonnet 5.5 reads the whole map; $0 via a Claude Pro subscription ($57 API-equivalent). Reasoning gave only ~+1 on dev; the LLM turn types cost 1.1 ILO on test; A1 and A2 get different pairs right (dev oracle 66.1 / 46.0). |
| A2 · Discussion | [`A2-discussion.md`](A2-discussion.md) | Why each system wins where it does (dev error analysis). The encoder learns the annotators' habits; the LLM applies the definitions. So the LLM invents half as many relations and reads direction better, while the encoder finds more loose support between neighbouring turns and is better at turn types. Both miss Conflict, 70% of which lies within one speaker's talk. |
| A3 · Fine-tuned open LLM (ARI) | [`A3-finetuned-llm.md`](A3-finetuned-llm.md) | **A3.1–A3.3 passed:** lossless pipeline; the training loop memorises 32 maps exactly; trained 3 epochs in 110 min, best at epoch 2 (greedy dev ARI 49.5 / 31.2); Qwen3-8B can't train on the 8 GB card, so the model is Qwen3-4B-Instruct-2507 (4.25 GiB, 789 tokens/s). QLoRA on it learns to write each map's relations from 1,231 annotated maps; sampled votes feed A1's decision layer. Tests whether an LLM that learns the annotators beats both A1 and A2. $0, local GPU. |
| 3 · ILO labeller | [`03-ilo-labeller.md`](03-ilo-labeller.md) | Run v1 selected: dev ILO 90.1 / 80.6 (cross-fitted), linking 98.3% at 99.6% precision. The remaining gap is minority-type typing; a longer, gentler v2 didn't close it (not undertraining). |

**Conventions for new reports**
- Name them `NN-short-name.md` for shared phases. Each approach gets one report (`A1-cross-encoder.md`, …), with a section added per phase.
- Use this shape: at a glance → method → results → lessons for other approaches → open items → how to reproduce.
- Every number must come from a committed file under `results/` (or `eda/`), named in the report's header.
- Comparisons with 2024 are only fair on the 11 test maps; label dev-versus-test numbers as references.
