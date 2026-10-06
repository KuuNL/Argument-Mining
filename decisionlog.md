# Decision log

Every decision in the project goes here as it is made, whether for the shared foundation or for an approach: what, why, what else was considered, and its status.
Newest decisions go at the bottom. This file moved here from `a1_cross_encoder/` (D027).

**Status:** `Proposed` (waiting for the user) · `Approved` · `Done` (carried out) · `Finding` (a measured fact, no choice involved) · `Open` (to be decided at a named phase) · `Superseded by Dxxx`.

## Phase map
This table is the **single record of phase numbering and status** (D027). Shared phases are numbered 0–3. Each approach numbers its own phases (A1.1, A1.2, …), and `PLAN.md` names its sections instead of numbering them.
Each phase follows the same loop: explain it, check it against `PLAN.md`, get approval, execute, verify the gate, log it, then move on.

| Phase | Name | Builds | Gate (must pass to move on) | PLAN section | Status |
|---|---|---|---|---|---|
| 0 | Environment | WSL2 venv `~/.venvs/dialam` (torch CUDA, transformers, sklearn); ModernBERT smoke test `a1_cross_encoder/check_env.py` | GPU visible, bf16 works, ModernBERT forward+backward on GPU, peak VRAM recorded | Foundation (deps) | **Passed (D009, D010)** |
| 1 | Foundation | `common/`: loader + graph view, AIF writer, validator, scorer port, episode-grouped dev split, timing harness | Scorer reproduces 29.53 / 35.30 / 32.41 and matches official scripts; validator passes all 1,478 gold maps; the split recomputes to the tracked `splits/` | Foundation | **Passed (D012–D016, D027)** |
| 2 | EDA | `eda/`: Pruning window *k*, direction stats (MA/CA later→earlier), speaker patterns, CA-absent maps, ModernBERT token lengths, leakage check | Every number feeds a named later decision | EDA | **Passed (D018–D024 proposed)** |
| 3 | ILO labeller (trained) | `ilo/`: candidate filter, (turn, proposition, context) dataset, ModernBERT-base classifier, one-to-one decoding + per-type thresholds; reusable fine-tuning loop | Linking accuracy on dev beats text similarity (94.2%); ILO F1 general / focused on dev reported | Shared ILO labeller | **Passed (D028–D039): run v1 selected, dev ILO 90.08 / 80.64 cross-fitted; iteration v2 not adopted** |
| A1.1 | Pair builder | `build_pairs.py`: candidates, 5-way labels, context text → `cache/` | ≥98% of gold relations survive pruning; gold→pairs→AIF round trip scores at the pruning ceiling; samples read by eye | A1 | **Passed (D040–D041): k = 10, dev ceiling 98.64 / 94.44** |
| A1.2 | Model + training loop | `model.py`, `train.py` on `common/finetune.py` (built in Phase 3): weighted/focal loss, bf16, grad ckpt, map-level early stop | Overfits a tiny subset; VRAM and epoch time measured; base vs large chosen | A1 | **Passed (D042): ~27 min/epoch, 5.0 GB** |
| A1.3 | Training run | Checkpoint + `results/timing.csv` | Dev per-map macro-F1 (argmax) recorded | A1 | **Passed (D043): best at 2 epochs, tuned dev ARI 50.9 / 33.9** |
| A1.4 | Decision layer | `tune.py` (temperature + per-class thresholds), `repair.py` | ECE before/after; thresholded ≥ argmax on dev | A1 | **Passed (D044): cross-fitted dev ARI 50.50 / 32.61** |
| A1.5 | Assemble + dev score | `predict.py` → valid AIF with S-nodes and YAs | Validator passes; dev ARI / ILO / Global table next to 2024 | A1 | **Passed (D045): 225/225 valid; dev Global ≈ 70.3 / 56.6** |
| A1.6 | Ablations | −context · −speaker/TA features · −pruning · argmax · (DeBERTa vs ModernBERT) | One lever off per run, re-scored on dev | A1 | **Passed (D047–D048)** |
| A1.7 | Test run | Final A1 row | Run once only, on the user's go | Final run | **Done (D050): test Global 72.01 / 57.53, beating 2024's 67.05 / 48.25** |
| A2.1 | Setup | `common/serialise.py` (map → text, shared with A3/A4); `a2_llm/`: per-map JSON schema, Claude Code caller (`claude -p`, Sonnet 5.5, Pro login), resumable runner with on-disk answer cache, parser → A1's pair format | Gold relations → JSON → parser → scorer gives the no-pruning ceiling (98.92 / 96.51); ~5 live calls valid, Sonnet 5.5 only, no hidden context; tokens, seconds and API-equivalent $ per call measured | A2 | **Passed (D055)**: round trip 98.92 / 96.51; Sonnet-only; ~$0.03 per answer API-equivalent |
| A2.2 | Prompt development | Pilot (~50 train maps, 3 episodes), TF-IDF example retrieval, system prompt; sweeps at 1 sample: examples k ∈ {0,2,4,8}, effort low/medium/high; ≤ 2 revision rounds | Prompt and settings frozen from a pilot table; timeline projected from measured calls per usage window | A2 | **Passed (D056–D059): frozen = prompt v1, k = 4, effort `xhigh`; reasoning +11 focused on the pilot** |
| A2.3 | Dev run + decision layer | Frozen settings, 3 samples × 225 dev maps (D054); per-type thresholds on confidence-weighted votes, cross-fitted | Honest dev ARI vs A1 (50.50 / 32.61); per-type P/R, direction, invented-Conflict maps; overlap of A1 and A2 errors | A2 | **Passed (D060–D062): honest dev ARI 54.63 / 34.12 vs A1 50.50 / 32.61** |
| A2.4 | Assemble + ablations | Valid maps and dev Global; LLM turn types adopted only if they beat the labeller; free ablations + the minimal-reasoning dev pass | 225/225 valid; ablation table | A2 | **Passed (D063): honest dev Global 72.53 / 57.63 vs A1 70.29 / 56.63; reasoning only +1 on dev** |
| A2.5 | Test run + report | Final A2 row; `reports/A2-whole-map-llm.md` | Run once, on the user's go | Final run | **Done (D064): test Global 73.67 / 58.27 (A1 72.01 / 57.53); ARI 55.77 / 33.34** |
| A3.1 | Setup + feasibility | venv `~/.venvs/a3` (peft, bitsandbytes, xgrammar, Unsloth or trl + liger); `a3_finetune/data.py`, `schema.py` / `parse.py` without confidence, `check.py`; memory probe Qwen3-8B → Qwen3-4B-Instruct-2507; generation probe | Round trip 98.92 / 96.51; every gold answer passes its grammar; no leakage; model chosen by the D065 rule (≤ 7.5 GiB, ≥ 200 tok/s, batch ≥ 2) | A3 | **Passed (D066)**: round trip 98.92 / 96.51; 0 grammar rejections; 8B fails the memory rule → **Qwen3-4B-Instruct-2507** (4.25 GiB, 789 tok/s) |
| A3.2 | Training loop | `train.py` (answer-only loss, adapter every half epoch, timing); `generate.py` (greedy types + seeded relation samples, grammar, cache; greedy batched across maps, D066) | Memorises 32 maps (≥ 95% relations exact); loss masking verified; epoch time and peak VRAM measured | A3 | **Passed (D067)**: 32/32 maps reproduced exactly (206/206 relations); masking verified; 3.83 GiB, ~720 tok/s; batched greedy ≈ 5.5 s per map |
| A3.3 | Training run | 1,231 test-like train maps, 3 epochs | Greedy dev ARI at each half epoch; best adapter chosen; diversity rule applied | A3 | **Passed (D068–D069)**: ep2.0 chosen, greedy dev ARI 49.46 / 31.19; 110 min training; votes graded (99.8%), so votes are the main score |
| A3.4 | Dev run + decision layer | Types + 8 relation samples × 225 dev maps (labeller links); scores → A1's pair format → cross-fitted thresholds | Honest dev ARI vs A1 (50.50 / 32.61) and A2 (54.63 / 34.12) with the bootstrap test; H1–H4 at matched density | A3 | **Done (D070)**: honest dev ARI 47.65 / 27.23, below A1 and A2; both success tests fail; H2 holds, H1/H3/H4 fail |
| A3.5 | Assemble + ablations | Valid maps, dev Global; LLM turn types (strict rule); combination preview; ablations | 225/225 valid; combination test; ablation table | A3 | Approved (D065) |
| A3.6 | Test run + report | Final A3 row; `reports/A3-finetuned-llm.md` | Run once, on the user's go | Final run | Approved (D065) |
| A4 | Jev | Phased (A4.1, …) when it starts | — | A4 | Not started |
| — | Combination · Final run | — | — | Combination / Final run | Not started |

---

## Decisions

### D001 — Backbone is ModernBERT
- **2026-09-29 · Setup · by user · Approved**
- A1's encoder is ModernBERT (`answerdotai/ModernBERT-*`), not DeBERTa-v3.
- *Why:* its 8k-token context lets the dialogue window grow far past DeBERTa's 512 tokens. Context is A1's main fix for 2024 failure #1.
- *Consequences:* README step 3 (`--model deberta-v3-large`) needs updating. DeBERTa-vs-ModernBERT stays on the plan's ablation list, and whether to run it is decided at A1.6.

### D002 — Gated, phase-by-phase execution
- **2026-09-29 · Setup · by user · Approved**
- One phase at a time, following the loop above. Nothing runs without approval (consistent with `CLAUDE.md` working rules).

### D003 — A1 builds the shared foundation first
- **2026-09-29 · Setup · by Claude · Approved by user 2026-09-29**
- The repo has no code yet. A1 needs PLAN's Foundation (loader, scorer, dev split), EDA (→ pruning window *k*) and ILO (L→I alignment, which test maps need because they carry no L→I links). These go in a shared `common/` package so A2–A4 can reuse them. A1-only code stays in `a1_cross_encoder/`.
- *Note:* `CLAUDE.md` says the scorer port was checked, but no scorer code exists in the repo. It is rebuilt and re-gated in Phase 1.
- *Alternative:* put everything in `a1_cross_encoder/`. Rejected because A2–A4 would have to duplicate or import across approaches.

### D004 — Dev maps are predicted in test-input format
- **2026-09-29 · Setup · by Claude · Approved by user 2026-09-29**
- Before prediction, dev maps are stripped to what test maps contain (I-nodes, L-nodes, TA chain; no YA L→I links, no S-nodes). Alignment errors then propagate on dev exactly as they will on test.
- *Why:* using gold L→I links on dev would make the dev scores optimistic, and the thresholds would be tuned to a setting that never occurs.

### D005 — Environment: project `.venv`, Python 3.14, torch CUDA 13.0 build
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29 · Superseded by D008** (where it runs; the package choices carry over)
- A `.venv` at the project root, shared by A1 and A3 (the plan says they use the same PyTorch stack). It holds torch 2.14.0+cu130, transformers 5.17, scikit-learn and numpy, pinned in `requirements.txt`.
- *Why cu130:* the RTX 5060 Ti is Blackwell (sm_120), which needs a CUDA ≥ 12.8 build. cu130 wheels exist for cp314, and driver 616.56 supports CUDA 13. cu128 (torch ≤ 2.11) is the fallback.
- *Why a venv:* it keeps the global Python (PyMuPDF etc.) untouched and makes the environment reproducible.
- *Attention kernel:* FlashAttention-2 has no reliable Windows wheels, so ModernBERT runs on PyTorch SDPA. This is slower than FA2 unpadding but needs no custom build.

### D006 — Develop on ModernBERT-base; choose base vs large at A1.2
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29**
- Base (149M params) for fast pipeline iteration. Large (395M) for the real run if the Phase 0 VRAM probe and the A1.2 throughput allow it on 8 GB.

### D007 — Phase 0 blocker: Windows Smart App Control blocks torch
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29: option 1 (WSL2 + Ubuntu)**
- *Finding:* `.venv` is created and torch 2.14.0+cu130, transformers 5.17.0 and scikit-learn 1.9.1 installed cleanly, but `import torch` fails with WinError 4551. The CodeIntegrity log (event 3033/3077) shows Smart App Control (state = 1, on) refusing `torch\lib\c10_cuda.dll`. 12 of torch's 39 DLLs are unsigned, and this affects every Windows torch build. `check_env.py` never reached the GPU.
- *Options:*
  1. **WSL2 + Ubuntu** (recommended). Smart App Control stays on, because Linux binaries run inside the WSL VM, and CUDA reaches the GPU through the Windows driver. Linux also supports Triton/`torch.compile` and FlashAttention-2, which ModernBERT's speed relies on, and bitsandbytes/Unsloth, which A3 needs. *Cost:* a one-off `wsl --install` (admin rights plus a reboot, done by the user), and the environment moves into WSL.
  2. **Turn Smart App Control off.** This is the fastest route, and everything else in D005 stays as it is. *Cost:* it removes a Windows security layer, and on many Windows builds it can't be turned back on without resetting Windows.
  3. **Cloud GPU** (e.g. RunPod or Colab). No local change, but every run costs money and time, and moving data back and forth complicates timing.
- *Not considered:* signing or allow-listing the DLLs ourselves, or any other way around the policy.

### D008 — ML environment runs in WSL2 + Ubuntu 24.04
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29** (the user confirmed WSL works and said to finish Phase 0) (follows from D007)
- **Code and data stay put** in `C:\Argument Mining`, reached from WSL as `/mnt/c/Argument Mining`, so the editor and Windows tools see the same files. Paths in code are relative to the repo, so it runs on either OS.
- **The venv lives on the Linux filesystem** (`~/.venvs/dialam`), as does the HF model cache (`~/.cache/huggingface`). Venvs and caches on `/mnt/c` are slow and have permission problems. The 1,478 small JSON maps are fine to read over `/mnt/c`.
- **Python 3.12** inside WSL, installed with `uv`, instead of 3.14. FlashAttention-2, bitsandbytes and Unsloth publish wheels for 3.12 first. Python code stays compatible with 3.12–3.14.
- **Packages:** torch 2.14.0+cu130, transformers 5.17.0, scikit-learn and numpy, as in D005, pinned in `requirements.txt`.
- **No NVIDIA driver inside Ubuntu.** WSL uses the Windows driver (616.56) through `/usr/lib/wsl/lib`, and installing a Linux driver breaks that.
- **The Phase 0 rerun adds an attention probe:** SDPA (baseline) vs FlashAttention-2 / `torch.compile` for ModernBERT, if they install cleanly. This reopens D005's "SDPA only" note.
- **The Windows `.venv`** (~3 GB, unusable for torch) is kept until the WSL setup works, then deleted only if the user agrees.

### D009 — Phase 0 result: gate passed, with a workaround
- **2026-09-29 · Phase 0 · by Claude · Finding**
- *Measured* (`check_env.py`, WSL, bf16 autocast, fp32 weights + AdamW, SDPA attention):

  | Probe | Peak VRAM | Step | Tokens/s |
  |---|---|---|---|
  | ModernBERT-base (150M), 2048 tok × 1 | 3.32 GiB | 0.139 s | 14,736 |
  | ModernBERT-large (396M), 1024 tok × 2, grad ckpt | 7.44 of 7.9 GiB | 1.422 s | 1,440 |
- *Finding 1:* **torch 2.14 routes some eager GPU ops through Triton**, for example the matmul in ModernBERT's rotary embedding, and Triton needs a C compiler that Ubuntu lacks. Until D010 is done, every run sets `TORCH_DISABLE_NATIVE_JIT=1`, which falls back to the standard CUDA kernels.
- *Finding 2:* **transformers 5.x removed ModernBERT's `reference_compile`.** Compiling now means calling `model.compile()`. The attention options are `sdpa`, `flex_attention` (which needs Triton, so a compiler) and `flash_attention_2`.
- *Finding 3:* **FlashAttention-2 is dropped.** Its prebuilt wheels stop at torch 2.10 (v2.8.3), and building it from source needs the CUDA toolkit and hours of compiling. This supersedes D008's FA2 probe.
- *Finding 4:* **ModernBERT-large full fine-tuning is at the VRAM limit.** At 7.44 of 7.9 GiB its step is about 3× slower than its size predicts, most likely because the driver is spilling into system RAM (to confirm in A1.2). The options for A1.2 are 8-bit AdamW, shorter inputs, or base (see D006).
- *Finding 5:* **WSL gets only 7 GB of the 16 GB RAM** by default. This is fine for now, and can be raised with `.wslconfig` if data loading needs more.

### D010 — Install a C compiler in WSL (`build-essential`)
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29**
- *Why:* it removes the D009 workaround and enables `torch.compile` and `flex_attention`, both possible speedups. Base is already fast enough, so this is not blocking.
- *How:* run `wsl -u root apt-get install -y build-essential` from Windows (this needs no Ubuntu password), then re-run `check_env.py` without the workaround.
- *Done:* installed `build-essential` (gcc 13.3) plus `python3.12-dev`, because Triton also needs `Python.h`. `TORCH_DISABLE_NATIVE_JIT` is **no longer needed**. Re-run results for ModernBERT-base at 2048 tokens:

  | Attention | Tokens/s | Peak VRAM |
  |---|---|---|
  | SDPA | 15,089 | 3.32 GiB |
  | SDPA + `model.compile()` | 14,372 | 3.12 GiB |
  | **flex_attention** | **18,746 (+24%)** | 3.21 GiB |
- *Consequence:* flex_attention is the working default, and it gets re-checked with real, variable-length padded batches in A1.2. `torch.compile` gave no gain here, so it's dropped for now. Large is unchanged (7.44 GiB, 1,502 tokens/s).

### D011 — Delete the unusable Windows `.venv`
- **2026-09-29 · Phase 0 · by Claude · Approved by user 2026-09-29 · Done** (3.5 GB removed)
- It is about 3 GB and torch can't load there (D007). `requirements.txt` now pins the WSL environment (59 packages).

### D012 — Episodes are recovered from panels; dev = 5 whole episodes
- **2026-09-30 · Phase 1 · by Claude · Done** (within the approved Phase 1 scope)
- *Problem:* the maps carry no episode label. Locution `start` dates are mostly import dates (824 maps share 2020-10-15, and 416 have none).
- *Method* (`common/split.py`):
  1. Join maps that share a specific speaker within 20 nodeset IDs. The host and "audience/unknown/speaker" labels don't count.
  2. Groups of ≥10 maps become episodes, and each episode's panel is the speakers in ≥3 of its maps.
  3. Merge groups that share ≥3 panellists. These are late re-annotation batches of the same episode, e.g. 18751… and 25554….
  4. Every other map joins the episode whose panel it overlaps most, or else the episode whose contiguous ID run contains it.
  5. Pick whole episodes (seed 13) until dev is about 15%.
  6. Leak guard: any training map that shares a panellist unique to a dev episode, or two non-recurring speakers with one dev map, moves to dev. This repeats until nothing changes.
- *Result:* 29 episode groups (the corpus has 30). 1,392 maps assigned; 86 unassigned, which go to train. **Train 1,250 / dev 228 (15.4%)** from episodes 0, 3, 10, 18 and 19. **0 leaky pairs.** Files: `splits/{train,dev}.txt`, `splits/episodes.json`.
- *Caught along the way:* annotator names were being read as speakers (see D014), and twin groups hid a leak. The audit found a real leaked map (21441, two audience members from dev episode 18) before the ID-run rule fixed it.

### D013 — Maps with no gold relations are left out of the focused mean
- **2026-09-30 · Phase 1 · by Claude · Done**
- The official script crashes (sklearn `ValueError`) when a map's focused list is empty. `common/score.py` leaves such maps out of that mean and reports how many it left out. This affects 3 training maps and none of the test maps.

### D014 — Analyst copies are bookkeeping, detected per map
- **2026-09-30 · Phase 1 · by Claude · Done**
- An L-node is an **analyst copy** if its prefix (`"Anna: Speaker : …"`) belongs to an analyst. A prefix counts as an analyst's if it yields an exact copy of another L-node in the same map. This catches the partial copies too: 2,484 of 2,488 compared with a global name list, and it also finds the test set's new analyst "Amelie". **Dialogue** = the non-copies that have a speaker or sit in the TA chain.
- *Evidence* (all training maps):

  | L-node kind | n | In TA chain | Anchors an I-node |
  |---|---|---|---|
  | Dialogue | 25,741 | ~98% | ~80% |
  | Copies (exact + partial) | 23,203 | 0% | **0%** |

  Only 7 of 20,746 gold L→I anchors start outside the dialogue.
- *For Phase 3:* **2,947 of 23,465 I-nodes (12.6%) have no L→I anchor at all**, so the linking must allow a proposition to have no locution.

### D015 — Scorer gate: plan numbers exact; the paper's table is not a per-map mean
- **2026-09-30 · Phase 1 · by Claude · Done**
- The port equals the **unmodified official scripts in 688/688 comparisons**: the 11 majority maps, 11 gold-vs-gold maps, and 150 perturbed training maps exercising every quirk.
- Majority baseline F1 (general) = **29.53 / 35.30 / 32.41**, exactly the plan's gate. The paper prints 29.52 / 35.29 / 32.40. Its ARI recall of 30.28 can't be a mean of per-map recalls, because each is 1/k and ours is exactly (4·¼ + 7·⅓)/11 = 30.30. So the gap comes from how the paper computed its table, not from the port.
- `CLAUDE.md` said "reproduces exactly", which was inaccurate, and is now corrected.

### D016 — Gold irregularities are reported, not repaired
- **2026-09-30 · Phase 1 · by Claude · Done**
- 60 of 1,478 training maps have irregular nodes: 56 orphan YAs, about 180 S-nodes missing a premise or a target, and 12 S-nodes pointing at an L-node or another S-node. The scorer reads them exactly as the official one does. Our own outputs must pass `validate()` with zero problems, and stripped maps do.
- *Also checked:* `strip_to_input(gold)` reproduces all 11 test inputs exactly (D004). The whole Phase 1 check runs in about 13 s (`results/timing.csv`).

### D017 — Phase 2 scope, location and data
- **2026-09-30 · Phase 2 · approved by user 2026-09-30**
- *Location:* the analysis is shared by every approach, so it lives in a top-level `eda/`. It holds the script (`eda/run_eda.py`), the raw numbers (`eda/findings.json`) and the findings document (`eda/README.md`), which has a narrative and diagrams, as the user asked.
- *Data:* the **training split only** (1,250 maps), so dev stays clean for tuning. Test **inputs** are read only for the size comparison and the text-overlap check. Evaluation gold is not read.
- *Deferred:* the plan's Jev-assisted profiling moves to approach 4. It is Jev-specific and costs API money.
- *Map boundaries* (user question): maps are the annotators' nodesets and come with the data. They are the scoring unit, and relations never cross maps. The median map has 14 dialogue turns (10th–90th percentile 9–22) and 2 speakers, and 27% contain more than one stretch of dialogue. Test maps stand alone, so model context comes only from inside the map.

### Phase 2 result
- **2026-09-30 · Phase 2 · Done.** The findings are in **`eda/README.md`**, a narrative with 5 figures. The numbers are in `eda/findings.json`, produced by `python -m eda.run_eda` in about 2.5 minutes, and the run is logged in `results/timing.csv`. The analysis used the training split only; pair statistics use the 1,231 test-like maps (≤ 40 propositions). Each number below feeds one of the proposed decisions D018–D024.

### D018 — Pruning window k = 8 turns, with distance also given to the model *(k amended to 10 by D041)*
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Approved by user 2026-10-01**
- k = 8 keeps **98.2%** of related pairs (98% was the plan's target) and drops 27% of pairs. Per type, the 98% window is Inference 6, Conflict 8, Rephrase 10. Propositions with no position (2.7%) are always kept as candidates.
- *Why distance is also an input:* neighbouring pairs are related 43% of the time, against 7.8% overall. Pruning alone leaves most of the imbalance (10.5% positives at k = 8).
- *Alternatives:* k = 5 (95.7% of relations, 52% of pairs) as a speed ablation, and no pruning.

### D019 — Context = the whole map's dialogue, max length 1,024 (supersedes the "dialogue window")
- **2026-09-30 · Phase 2 → A1.1/A1.2 · by Claude · Approved by user 2026-10-01**
- The pair plus the *entire* dialogue is ≤ 837 tokens (p99 621) in the test-like maps, and whole test maps are ≤ 953 tokens. The model sees everything, which is the direct fix for 2024's no-context failure.
- *Cost:* about 1.8× the tokens of ±2 turns, roughly 30 vs 17 min per epoch for base (to be measured in A1.2). The ±2-turn window becomes an ablation.

### D020 — Label scheme: 5 classes; Rephrase/Conflict direction from dialogue order
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Approved by user 2026-10-01** (tie-break amended: see the last bullet)
- Rephrase and Conflict run later → earlier in 97.8% and 97.7% of cases, so their direction comes from the turn order at a cost of about 2%. When both propositions are in the same turn (1.2% of Rephrases), ~~node order~~ **the canonical pair order (by proposition text) breaks the tie**, because node order is ruled out by D029. **Inference is 53% later→earlier and 47% earlier→later, so the model predicts its direction** (A→B Inference and B→A Inference are separate classes).
- Only 12 pairs carry two relations, so one label per pair is fine.

### D021 — Input features: distance and same-speaker marked explicitly
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Approved by user 2026-10-01**
- Same-speaker rates are Inference 94.5%, Rephrase 81.7% and Conflict 56.1%, against a 66.2% base rate among candidates. Distance: see D018. Both go into the input as short marker tokens; the "−speaker/TA features" ablation removes them.

### D022 — The 19 giant training maps (> 40 propositions) are an open question for A1.1
- **2026-09-30 · Phase 2 → A1.1 · by Claude · Decided (Approved by user 2026-10-01): left out of A1's first run; chunked variant in A1.6**
- They hold 60% of all pairs but only 12.8% of relations, and no test map is that large (max 35). Options: train only on their pruned candidates (k = 8 already cuts them sharply), cap them, or drop them. A1.1 decides once pair counts after pruning are known.
- *Measured 2026-10-01 (train split, k = 8, EDA positions):*

  | | Test-like maps (1,231) | Giant maps (19) |
  |---|---|---|
  | Candidate pairs | 92,465 | **81,227** |
  | Relations kept | 9,721 (97.8%) | 987 (**67.7%**) |
  | Propositions that can't be placed | 9 | **528 of 2,592** (no L→I layer, D030) |

  Other facts: their dialogues of about 200 turns don't fit the 1,024-token whole-map input (D019), and they hold about 19% of all Conflict examples (204 of 1,069).
- *Recommendation:* leave them out of A1's first training run. Add them back as a chunked variant (about 25-turn windows, positions from the ILO labeller's predicted links) in A1.6, to test whether their Conflict examples help.

### D023 — ILO rules (Phase 3): question words and "xxx", not "?"; similarity-first alignment
- **2026-09-30 · Phase 2 → Phase 3 · by Claude · Superseded by D025.** There are no ILO rules any more. The model learns these cues. The similarity finding survives as D025's candidate filter and baseline.
- Only 7–8% of question turns contain "?", because the transcripts have almost none. A turn starting with a question word marks 74.5% of Pure Questioning, 67.9% of Rhetorical and 41.8% of Assertive Questioning, against 4.0% of Asserting. The proposition contains "xxx" in 45.7% of Pure Questioning, against 0.4% of Asserting.
- Alignment: text similarity alone picks the gold proposition for 94.2% of turns. Links are one-to-one (98.8% / 99.8%), and 20% of turns express no proposition.
- *Corrects* the "?" cue assumed in `PLAN.md`'s ILO section and `CLAUDE.md` §6.

### D024 — Unscored transition illocutions in the output maps
- **2026-09-30 · Phase 2 → A1.5 · by Claude · Proposed**
- Inference → Arguing (97.9%), Rephrase → Restating (85.9%; Default Illocuting 13.7%), Conflict → Disagreeing (94.3%). These are used when assembling valid AIF. They aren't scored, so the majority choice is enough.

### D025 — Phase 3 is a trained ILO labeller only (no rules)
- **2026-09-30 · Phase 3 · by user (approach) / Claude (design) · Approved by user 2026-10-01 (go-ahead for 3.1–3.2, then review)**
- *User decision:* ILO type labelling is learned, not rule-based. This supersedes the rules in PLAN's ILO section and D023. `PLAN.md`, `CLAUDE.md` §6–7 and `a1_cross_encoder/README.md` are updated to match.
- *Why:* ILO is half the Global score. "?" is almost absent (EDA §3). Macro-F1 weighs the rare types (Assertive / Rhetorical Questioning, Agreeing, Challenging) as heavily as Asserting. The rare-label trap also applies here (one invented type drops a map to 72.8), so the model must produce probabilities that can be thresholded. All top 2024 ILO systems were trained.
- *Design* (code in a top-level `ilo/`, shared by all approaches; the fine-tuning loop goes in `common/finetune.py` so A1 reuses it in A1.2):

  | Step | What | Check before moving on |
  |---|---|---|
  | 3.1 Candidates | Pair each dialogue turn with its top-K most similar propositions; choose the smallest K keeping ≥ 99.5% of gold links (measured on train) | Recall at K = 1…5 |
  | 3.2 Dataset | Input `[CLS] Speaker: turn [SEP] proposition [SEP] ±2 neighbouring turns`; 11 labels = None + the 10 L→I types seen in training | Label counts, token lengths, samples read by eye |
  | 3.3 Train | ModernBERT-base, bf16, flex attention, class-weighted cross-entropy, early stopping on dev ILO macro-F1 (per map, full decode) | First overfits 64 examples; VRAM and epoch time logged |
  | 3.4 Decode | Temperature scaling; one-to-one matching on 1 − P(None) above a link threshold; per-type thresholds (fallback Asserting), all tuned on dev | Thresholded ≥ argmax on dev |
  | 3.5 Evaluate | Dev maps in test-input format (D004); thresholds cross-fitted across dev episodes, so the reported number isn't flattered by tuning | Linking accuracy vs the similarity-only baseline on dev; ILO F1 general / focused and per type; timing |
- *Estimated cost:* about 65k candidate pairs per epoch × ~150 tokens ≈ 10M tokens, so roughly 10 min/epoch on ModernBERT-base, and 30–50 min for 3–5 epochs. No API spend.
- *This is the project's first training run and first dev scoring*, both of which need the user's go-ahead under the working rules.
- *Later ablations (A1.6):* the whole map as context instead of ±2 turns, argmax instead of thresholds, no one-to-one decoding.

### D026 — The repository is under git
- **2026-09-30 · Setup · by user · Approved · Done** (baseline commit `7462dc3`)
- *What:* `git init` on `main`, with a baseline commit of the Phase 0–2 state (26 files).
  - `.gitignore` leaves out `data/` (the re-downloadable DialAM release; `data/evaluation/` holds the test gold and must never enter history), Python caches and venvs, model checkpoint files (`*.pt`, `*.safetensors`, …) and `.claude/settings.local.json`.
  - `.gitattributes` stores text with LF line endings, so Windows git and WSL git see identical files, and treats PDFs as binary.
- *Why:* every change becomes reviewable and reversible, and each result can be tied to the exact code that produced it.
- *Deferred by the user:* whether to track `splits/`, and any change to the folder layout. Both wait for the project sweep and its proposals.
- *Mechanics:* commits made from Claude Science pass the author per command (`git -c user.name=… -c user.email=… commit`), because its sandbox keeps `.git/config` read-only. Commits from a normal terminal use the user's global git identity.

### D027 — Project sweep before Phase 3
- **2026-10-01 · Setup · proposals P1–P7 by Claude Science, X1–X3 by Claude · Approved by user 2026-10-01 · Done**
- **P1, track `splits/`:** already done by the user in `bfb6718`.
  - **X1:** `python -m common.check_foundation` used to *rewrite* `splits/` on every run. Now the split is only recomputed and compared: `python -m common.split` checks, and `--write` regenerates (which shows up as a git diff).
  - A new gate, `split_matches_tracked_files`, brings the foundation check to **13/13 passing**, with the split unchanged.
- **P2, decision log moved to the root** with `git mv` (`git log --follow decisionlog.md` keeps its history) and retitled "Decision log". 6 path references updated: `CLAUDE.md` ×2, `PLAN.md`, the A1 README, `eda/README.md`, `requirements.txt`.
- **P3, one numbering scheme:** shared phases are 0–3 and A1's are A1.1–A1.7 (old 4→A1.1, 5→A1.2, 6→A1.3, 7→A1.4, 8→A1.5, 9→A1.6, 10→A1.7). A2–A4 get their own numbers when they start. `PLAN.md` sections are now named, not numbered, and all phase references in the log, READMEs and docstrings were renamed.
- **P4, the phase map is the single status record.** The A1 README's "Status: plan only" now points here. The Project Context "Current state" in Claude Science lives outside the repo and has to be changed there.
- **P5 / X2, stale text corrected:**
  - "pruning removes most None pairs" (`CLAUDE.md` §7, `PLAN.md` A1): it removes about a quarter at k = 8;
  - "analyst copies are linked by Analysing" (`CLAUDE.md` §5): only about 2% are;
  - "far fewer" in the A1 README's pruning figure;
  - the log's environment row (`.venv` → WSL venv) and A1.2 row (now uses `common/finetune.py`);
  - `PLAN.md`'s "ILO is close to solved".
  - The "dialogue window" wording is left as it is until D019 is decided.
- **P6, complete status legend:** Done, Finding and Open added.
- **P7, `cache/` ignored** before Phase 3, because Phase 3 already writes tokenised caches.
- **X3, line endings:** 8 working files had CRLF, because Windows Python writes CRLF by default; they are normalised to LF. Edits now write LF explicitly.

### D028 — Phase 3.1: ILO candidates = own top-5 + previous turn's top-5 + next turn's top-3
- **2026-10-01 · Phase 3 · by Claude · Done** (within the approved 3.1 scope)
- *Measured on the train split* (`python -m ilo.candidates`): similarity alone (difflib ratio) keeps 94.3% of the 17,509 gold links at K = 1, 97.5% at K = 3 and 98.8% at K = 6. No K reached the 99.5% target.
- *Why it misses:* 371 of the 433 misses at K = 3 are short turns whose proposition was rebuilt from context ("Yes" becomes the claim agreed with; "it stinks" becomes what "it" refers to). These are mostly Agreeing, Disagreeing and question types, which weigh most in macro-F1.
- *Fix:* a turn also borrows the candidates of its neighbouring turns. **Own 5 + previous 5 + next 3 keeps 99.59%**, with 205,955 pairs (8.5% positive). Every type is ≥ 98.3% except Agreeing (93.7%, 89/95) and Disagreeing (94.7%, 18/19).
- *Alternatives measured:*

  | Scheme (own / prev / next) | Links kept | Pairs |
  |---|---|---|
  | 3 / 2 / 0 | 98.8% | 97k |
  | 3 / 3 / 0 | 99.0% | 113k |
  | 4 / 3 / 2 | 99.3% | 155k |

- *Cost:* about 2× the pairs of own-5 alone (≈ 31M tokens per epoch, roughly 30 min). Training on a subsample of the None pairs is the fallback if that proves too slow.

### D029 — No node-ID or node-order shortcuts, in any approach
- **2026-10-01 · All approaches · by Claude · Approved by user 2026-10-01**, including the documented exception for ordering separate dialogue stretches
- *Finding:* in 74% of gold links, the proposition's node ID is exactly the turn's ID + 2, because the annotation tool creates L, YA and I in sequence. Test inputs keep these IDs, so a model could link by ID instead of by meaning.
- *Proposal:* never use node IDs, or the order of nodes in the JSON, as a signal. Use only text, speakers and the TA dialogue order. Results then measure language understanding, transfer to unannotated debates, and stay comparable across A1–A4.
- *Cost:* linking can't exploit the artefact. Expect a few points of linking accuracy left on the table compared with a system that does.
- *One existing use to decide on:* `MapView.dialogue` orders the separate stretches of a map's TA chain (27% of maps have more than one) by node ID, because nothing else orders them. This never links a turn to a proposition, but it sets distances *across* stretches (EDA, A1). Options: keep it as a documented exception, or treat stretches as unordered.

### D030 — Maps without an illocution layer are left out of ILO training and ILO dev scoring
- **2026-10-01 · Phase 3 · by Claude · Done** (within the approved 3.2 scope; easy to reverse)
- *Finding:* 1,614 "hard negatives" (a turn nearly identical to a proposition, yet no gold link) are not noise spread across the data. **19 train maps and 3 dev maps have no L→I links at all** (0 linked turns among 140 to 400 turns each), and they hold 1,553 of them. These are the late re-annotation batches (nodesets 24807 to 25475), which are **the same giant maps as in D022**. They carry relations but were never annotated for illocutions.
- *Rule:* a map with no L→I link at all is skipped by `ilo/data.py`. The maps stay in for ARI (A1), and the test maps are fully annotated.
- *Effect:* train goes from 205,973 to 158,864 examples, dev from 36,112 to 29,506; no labelled example is lost. ILO dev scoring covers 225 of 228 dev maps.

### D031 — Phase 3.2: the ILO dataset
- **2026-10-01 · Phase 3 · by Claude · Done**
- *Input:* `text_a` = "Speaker: turn"; `text_b` = "proposition [SEP] before: (up to 2 turns) [SEP] after: (up to 2 turns)". There are 11 labels: None plus the 10 L→I types seen in training. Dev inputs are built from stripped maps (D004).
- *Size:* train 1,231 maps, 158,864 examples (11.0% links); dev 225 maps, 29,506 examples (10.9%). Tokens: median 108, p99 186, max 300, so **max length 320** truncates nothing.
- *Rare types are very rare.* Train has Default Illocuting 123, Agreeing 89, Challenging 38, Disagreeing 18, Arguing 10 and Restating 4. Dev has 7, 11, 2, 2, 2 and 1. Per-type thresholds cannot be tuned reliably on dev for these. See the open question for 3.4.
- *Files:* `cache/ilo/{train,dev}.jsonl` (not in git, rebuilt by `python -m ilo.data` in about 4 min); stats in `results/ilo/dataset.json`.

### D032 — Phase 3.3 setup: training loop, sanity checks, attention kernel
- **2026-10-01 · Phase 3 · by Claude · Done**
- *Training loop* (`common/finetune.py`, reusable in A1.2): length-grouped batches, bf16 autocast with fp32 weights, weighted cross-entropy, fused AdamW (lr 5e-5, weight decay 0.01, 6% warm-up, linear decay), gradient clip 1.0, batch 32, max 320 tokens, up to 3 epochs. Dev is evaluated twice per epoch; training stops after 2 evaluations without improvement. The best checkpoint and its dev probabilities go to `checkpoints/ilo/`, which is git-ignored.
- *Class weights:* (most frequent / count)^0.5, capped at 30. None 1.0, Asserting 3.0, Pure Questioning 11.9, Assertive / Rhetorical about 26–27, and the six rare types 30.
- *Early-stopping metric:* dev ILO general F1 per map with the simple decode in `ilo/decode.py`: one-to-one Hungarian matching on 1 − P(None) ≥ 0.5, then the most probable type. Thresholds come in 3.4.
- *Sanity checks:*
  - Decoder with gold labels as input: **ILO 98.63 general / 94.73 focused**, linking recall 99.16%. This is the ceiling of this design on dev (candidate misses, plus turns that express two propositions).
  - All None: 38.21 / 0, the majority baseline on dev.
  - The model memorises 72 examples (100% train accuracy) in 100 steps: 10.6 s, 15.3k tokens/s, 3.43 GiB peak.
- *Attention kernel:* on real variable-length batches **SDPA runs at 14,184 tokens/s, but flex_attention only at 2,749**, because it recompiles for every new shape. This supersedes D010's "flex as working default", which held only for fixed shapes. Training uses SDPA.

### D033 — Batches are capped by tokens, not by sequence count
- **2026-10-01 · Phase 3 · by Claude · Done**
- *What happened:* the first full training run was **stopped at step 600 (0.12 epoch)** by the background-tool time limit (10 min). Its log, `results/ilo/train_log_run1_killed.jsonl`, also showed a real problem: throughput fell from 4.9k to 3.7k tokens/s, and peak VRAM reached 7.72 of 7.9 GiB. Fixed 32-sequence batches of the longest examples (about 9.6k tokens), on top of about 2.4 GB of weights and optimizer state, overflowed the card, and the driver spilled into system RAM.
- *Fix* (`common/finetune.py`):
  - Batches are packed under a padded-token budget: **4,096 tokens** for training (at most 64 sequences) and 16,384 for prediction.
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` limits fragmentation from variable shapes.
- *Measured* (`python -m ilo.train --bench`, 400 steps): **26.3k tokens/s steady and 4.33 GiB peak**; dev prediction runs at 587 examples/s with a 1.25 GiB peak. One epoch is about 11–15 min, so a full run takes about 45–50 min including dev evaluations.
- *For A1:* the same budget applies. Longer A1 inputs mean fewer pairs per batch, not more memory.

### D034 — Phase 3.3 result: training stopped early at 1.5 epochs; best checkpoint at 0.5 epoch
- **2026-10-01 · Phase 3 · by Claude · Finding**
- *Launch:* detached as a hidden Windows process (user's choice) so it wasn't bound by the 10-minute tool limit. `scratchpad/run_ilo_train.sh` was monitored through `results/ilo/train_log.jsonl`.
- *Dev ILO general / focused F1 (simple decode)* by evaluation: 0.5 epoch **89.46 / 78.06** (best), 1.0 epoch 86.41 / 72.31, 1.5 epoch 88.05 / 73.18. Patience 2 stopped the run.
- *Cost:* 19.5 min in total (13.0 min per epoch, including evaluations), 22.5k tokens/s, 4.35 GiB peak. Best checkpoint in `checkpoints/ilo/` (598 MB, git-ignored).
- *Caveat:* the best checkpoint is very early (0.5 epoch, while the learning rate was still near its peak). It never predicts the six rare types on dev (see D036). Later checkpoints may have learned them but scored lower under the simple decode, and they weren't kept.

### D035 — Phase 3.4 result: decision layer tuned on dev; the rare-type grouping question is moot
- **2026-10-01 · Phase 3 · by Claude · Done**
- `ilo/tune.py` fits a temperature, then thresholds by coordinate ascent on mean per-map ILO general F1. The fast metric equals the official scorer on all 225 dev maps (asserted).
- *Result:* **T = 1.05** (already well calibrated) and **link_min = 0.10** (4 of 5 folds pick 0.10, one picks 0.05). The first run's grid floor of 0.20 was too high and was widened to 0.05. P(type | link) minimum: Pure Questioning 0.5, Assertive Questioning 0.5, Rhetorical Questioning 0.3, and **0 for all six rare types**.
- *Scores:* simple decode 89.46 / 78.06; tuned on all of dev 90.16 / 80.67 (optimistic); **leave-one-episode-out 89.89 / 80.38 (honest)**. *Updated in D039:* after the link grid was widened to 0.01, v1's cross-fitted score is **90.08 / 80.64**; the all-dev result is unchanged.
- *Grouping question:* "shared" and "per-type" thresholds give **identical results**, because the data set every rare-type threshold to 0. No user decision is needed. Files: `results/ilo/thresholds_{shared,per-type}.json`.

### D036 — Phase 3.5 result: gate passed; weak on rare and assertive/rhetorical types
- **2026-10-01 · Phase 3 · by Claude · Finding** (`python -m ilo.evaluate`, `results/ilo/dev_report.json`; 225 dev maps in test-input format)

  | Dev ILO F1 (general / focused) | |
  |---|---|
  | Majority (no illocutions) | 38.21 / 0.00 |
  | Text similarity, one-to-one, all Asserting (threshold tuned) | 75.99 / 59.38 |
  | Model, simple decode | 89.46 / 78.06 |
  | **Model, tuned decision layer (cross-fitted)** | **89.89 / 80.38** (90.08 / 80.64 with the widened grid, D039) |
  | *Ceiling of this design (gold labels decoded)* | 98.63 / 94.73 |

- **Gate passed:** the model links 98.27% of gold links at 99.62% precision, against 97.43% at 98.68% for text similarity with the same one-to-one matching, and about 94% for similarity top-1.
- *Per type:* Asserting 0.98 F1, Pure Questioning 0.87, Rhetorical Questioning 0.51 (14 predicted vs 29 gold), Assertive Questioning 0.40 (6 vs 19). **Default Illocuting, Agreeing, Challenging, Disagreeing, Arguing and Restating are never predicted** (26 gold links in dev). This is the main room left between 80.4 focused and the 94.7 ceiling.
- *Inference:* 0.21 s per map (48 s for all of dev). A reloaded checkpoint reproduces the saved probabilities exactly.
- *Reference only (different maps):* 2024's best ILO on the test set was 81.2 / 72.1.
- *Possible improvements:* select checkpoints by the *tuned* metric and keep every evaluation's probabilities; lower the learning rate or class-weight cap; give rare types more epochs. These are proposed for a Phase 3 iteration, or as A1.6 ablations, at the user's call.

### D037 — Findings reports live in `reports/`, one per phase
- **2026-10-01 · Setup · requested by user · Done**
- `reports/README.md` is the index: one line per phase, with the headline and a link. `reports/03-ilo-labeller.md` is the Phase 3 report. Phase 2's report stays at `eda/README.md`, where it is already linked; Phases 0–1 are covered by their decision-log entries.
- *Convention:* `NN-short-name.md`, shaped as at a glance → method → results → lessons → open items → reproduce. Every number comes from a committed file under `results/` or `eda/`. These reports are the source for the final write-up (now noted in `PLAN.md` and `CLAUDE.md` §5).
- *Added for the report:* `python -m ilo.evaluate --headroom` (`results/ilo/headroom.json`). Correcting rare and Assertive/Rhetorical types to gold would lift dev ILO from 90.16 / 80.67 to **97.07 / 90.49**; perfect typing of correct links reaches 97.87 / 90.61. Typing, not linking, holds the remaining gap.

### D038 — ILO rare-type iteration (run v2), and one folder per training run
- **2026-10-01 · Phase 3 · approved by user 2026-10-01 · Done; result in D039**
- *Why:* v1 never predicts Default Illocuting, Agreeing or the other rare types, and it is weak on Assertive and Rhetorical Questioning. On v1's dev probabilities the evidence points to **undertraining**, not unlearnable types:
  - **Default Illocuting** (123 training examples) gets probability about 0.001.
  - **Assertive and Rhetorical Questioning** put the right type first or second in 31 of 48 dev links.
  - **Agreeing** turns get a median link probability of 2%.

  The kept checkpoint was from 0.5 epoch, because checkpoints were judged by the simple decode, which punishes rare-type mistakes hardest.
- *What changes in v2 (one recipe, fixed in `ilo/train.py` RUNS):*
  - All 3 epochs (patience 99).
  - **The checkpoint is chosen by the tuned score** (`ilo.tune.tuned_score`: temperature plus thresholds on all of dev, about 13 s per evaluation).
  - **Every evaluation's dev probabilities are kept** (`checkpoints/ilo/v2/evals/`).
  - **Learning rate 3e-5** instead of 5e-5.
  - Data, class weights (cap 30), batching and model are unchanged.
- *Expectation:* Challenging, Disagreeing, Arguing and Restating (38, 18, 10 and 4 training examples) will likely stay unpredicted, which is correct given the rare-label trap.
- *Stop rule:* adopt v2 only if its **cross-fitted** dev ILO general F1 beats v1's 89.89 by more than about 0.5. Otherwise keep v1 and move to A1.1.
- *Caveat:* choosing among 6 checkpoints on dev adds slight optimism. Cross-fitting covers the thresholds, not the checkpoint choice. v1's early stopping had the same property.
- *Structure:* each run has its own `checkpoints/ilo/<run>/` and `results/ilo/<run>/`. v1's files were moved with `git mv`, so their history is kept. `results/ilo/selected.json` names the run the rest of the project uses (now v1). `ilo.tune` and `ilo.evaluate` take `--run=`.

### D039 — Rare-type iteration result: v2 not adopted, v1 stays selected
- **2026-10-01 · Phase 3 · by Claude (applying the stop rule in D038) · Done**
- *v2 training* (all 3 epochs, lr 3e-5; tuned on all of dev at each evaluation):

  | Epoch | 0.5 | 1.0 | 1.5 | **2.0** | 2.5 | 3.0 |
  |---|---|---|---|---|---|---|
  | Tuned general / focused | 89.06 / 80.36 | 89.41 / 80.12 | 89.93 / 79.99 | **90.39 / 81.97** | 89.66 / 80.43 | 89.39 / 80.09 |
  | Temperature | 1.10 | 1.15 | 1.40 | 1.20 | 1.55 | 1.65 |

  The temperature rises as the model grows overconfident. Run time 42.1 min (14.0 min per epoch), 20.8k tokens/s, 4.35 GiB peak.
- *Final tuning:* the link-threshold grid was widened to 0.01 (`ilo/tune.py`), because v2's optimum sat on the old 0.05 floor. **Both runs were re-tuned on it:**

  | Dev ILO | v1 | v2 |
  |---|---|---|
  | Tuned on all of dev (optimistic) | 90.16 / 80.67 | 90.40 / 82.29 |
  | **Cross-fitted (honest)** | **90.08 / 80.64** | 89.74 / 81.29 |
  | Linking recall / precision | 98.27 / 99.62% | 98.61 / 99.59% |
  | Assertive / Rhetorical Questioning F1 | 0.40 / 0.51 | 0.46 / 0.56 |
- *Rare types:* v2's tuning sets every rare-type threshold to "never" (1.01). With suppression switched off, **v2 predicts Agreeing 7 times on dev and is wrong every time**, and it still never predicts Default Illocuting. v1 predicts none either way.
- **Stop rule** (cross-fitted general must beat v1 by more than 0.5): **not met**, since v2 is 0.34 lower. v1 stays in `results/ilo/selected.json`. Every v1/v2 difference is within dev's noise.
- *Lesson:* the rare-type gap is **not undertraining**. With this design (one pair at a time, ±2 turns, 89 Agreeing examples), the model can't learn Agreeing or Default Illocuting well enough to beat the rare-label penalty. Options for later, none done now:
  - add the transition-anchored Agreeing / Disagreeing / Challenging examples as extra training signal;
  - give short reply turns the turn they answer as explicit context;
  - accept these types as unpredicted, which the tuned decision layer already does safely.
- *For A1:* choosing checkpoints by the tuned metric and keeping every evaluation's probabilities is cheap (about 13 s per evaluation) and makes the selection honest. Use it in A1.2, and recalibrate each checkpoint because the temperature drifts.

### D040 — Where A1 gets each proposition's turn and speaker
- **2026-10-01 · A1.1 · by Claude · Approved by user 2026-10-01**
- **Training maps:** gold L→I links. **Dev and test maps:** the ILO labeller's *predicted* links (the selected run, v1), because that is all a test map offers (D004).
- *Fallback chain on every split:* link → most similar turn (similarity ≥ 0.8) → unplaced. Unplaced propositions are always candidates (D018).
- *Not used:* the gold-only transition anchors (0.7%), so training doesn't see a signal that test can't provide.

### D041 — A1.1 result: pairs built; the window gate just missed at k = 8, and k = 10 recommended
- **2026-10-01 · A1.1 · by Claude · Approved by user 2026-10-01: k = 10** (pairs rebuilt at k = 10; the k = 8 figures below are kept for the record)
- *Built:* `ilo/predict.py` (labeller links for any map; reproduces the saved v1 dev predictions exactly), `common/text.py`, `common/positions.py`, `a1_cross_encoder/build_pairs.py` and `assemble.py`. Train 1,231 maps / 92,465 pairs, dev 225 / 17,373, test 11 / 1,048 (inputs only). Inputs are at most 857 tokens with the whole dialogue (D019 holds, no windowing). All round-trip maps validate. Details in `results/a1/pairs.json` and `reports/A1-cross-encoder.md`.
- *Gate:* 97.72% (train) and 97.76% (dev) of related pairs fall inside k = 8, against the ≥ 98% target. The EDA's 98.2% included gold-only transition anchors, which D040 excludes.
- *Ceiling* (gold labels → map → official ARI, dev): k = 8 gives 97.92 / 92.19; k = 9 gives 98.39 / 93.56; **k = 10 gives 98.64 / 94.44 for +13% pairs**; no window gives 98.92 / 96.51. The remaining ~3.5 focused points come from the dialogue-order direction rule for Conflict/Rephrase and from pairs with two relations.
- *Recommendation:* k = 10, which recovers about 2.3 focused points of ceiling for about 13% more training time.

### D042 — A1.2 result: training loop ready; three issues found and fixed on the way
- **2026-10-01 · A1.2 · by Claude · Done**
- *Built:* `a1_cross_encoder/decide.py` (decision layer core plus an exact, fast ARI metric) and `a1_cross_encoder/train.py` (recipes; `--overfit`, `--bench`, full run). `common/finetune.py` gains gradient accumulation (`accum_steps`) and compact tokenisation.
- *Recipe v1:* ModernBERT-base, max 1,024 tokens, 4,096-token batches (about 10 pairs) × 3 accumulated, so about 30 pairs per update. lr 3e-5, 3 epochs, no early stop. **Checkpoints judged by the tuned dev ARI score** (temperature plus Inference / Conflict / Rephrase thresholds on all of dev), and every evaluation's dev probabilities kept, as Phase 3 taught (D039). Class weights (most frequent / count)^0.5: None 1.0, Rephrase 4.9, Inference B→A 6.0, Inference A→B 6.4, Conflict 10.5.
- *Fixed 1, metric:* the fast ARI metric first disagreed with the official scorer on 13 of 675 map scorings, because **gold contains a few self-loops** (a proposition related to itself), which the official scorer never sees. After excluding them: **0 mismatches**, and gold-as-probabilities reproduces the A1.1 ceiling exactly (98.64 / 94.44).
- *Fixed 2, RAM:* the first benchmark was **killed by Linux for running out of memory** (7.3 GB RSS; WSL has 7 GB). 104k inputs of about 374 tokens became about 40M Python ints, stored twice (ids and mask). Tokenising in chunks into int32 arrays without masks, and freeing the texts, brings the peak to **3.6 GB**.
- *Check, memorisation:* 72 pairs (including 9 Conflict) reach 83% after 100 steps and **100% (loss 0.0) after 300 steps**. Longer inputs need more steps than ILO did; nothing is misaligned.
- *Benchmark* (300 optimizer steps on the real data): **24.4k tokens/s, 4.98 GiB** peak, 39.1M tokens per epoch, so **about 27 min per epoch**. Dev prediction runs at 129 pairs/s (about 2.5 min per evaluation, plus tuning). Full run (A1.3): about 1 h 40 min.
- *Base vs large (D006):* base. Large was about 3× slower and at the 8 GB limit in Phase 0.

### D043 — A1.3 result: best checkpoint at 2 epochs, dev ARI 50.9 / 33.9 (tuned)
- **2026-10-01 · A1.3 · by Claude · Finding** (run v1, launched detached; `results/a1/v1/train_log.jsonl`)

  | Epoch | 0.5 | 1.0 | 1.5 | **2.0** | 2.5 | 3.0 |
  |---|---|---|---|---|---|---|
  | Tuned ARI general / focused | 42.07 / 19.75 | 45.53 / 25.50 | 49.31 / 30.43 | **50.90 / 33.91** | 50.37 / 30.65 | 50.59 / 33.44 |
  | Simple decode | 37.69 / 11.95 | 41.87 / 15.85 | 48.78 / 31.29 | 50.08 / 33.82 | 49.45 / 30.84 | 49.79 / 31.41 |
  | Temperature | 0.70 | 0.95 | 0.95 | 0.90 | 1.30 | 1.35 |

- *Cost:* 94.4 min in total (31.5 min per epoch including dev evaluations), 20.7k tokens/s, 5.0 GiB GPU, 3.6 GB RAM.
- *Learning:* relations are learnt much more slowly than illocutions. ILO peaked at 0.5 epoch; ARI climbs until 2 epochs, then flattens as the model grows overconfident (temperature above 1). Choosing checkpoints by tuned score picked epoch 2.

### D044 — A1.4 result: thresholds tame the rare-label trap; honest dev ARI 50.50 / 32.61
- **2026-10-01 · A1.4 · by Claude · Done** (`python -m a1_cross_encoder.tune --run=v1`, `results/a1/v1/thresholds.json`)
- *Fitted on dev:* temperature 0.90; thresholds Inference **0.30**, Conflict **0.65**, Rephrase **0.45**. They are stable across folds (Inference 0.30–0.40, Conflict 0.55–0.65, Rephrase 0.45).
- *Dev ARI:* simple decode 50.08 / 33.82; tuned on all of dev 50.90 / 33.91; **leave-one-episode-out 50.50 / 32.61**. Thresholds help general (+0.4) but cost focused (−1.2), because they are tuned for the general metric.
- *Rare-label trap:* maps predicting Conflict with none in gold fall from **34 to 8**.
- *Per type (pooled, tuned):*
  - Inference: P 0.27, R 0.39, F1 0.32 (1,542 predicted vs 1,060 gold, so over-predicted);
  - Conflict: P 0.31, **R 0.09**, F1 0.14 (39 vs 130);
  - Rephrase: P 0.42, R 0.53, F1 0.47.
- *Direction:* among gold Inference pairs found, **81% get the right direction**.
- *Calibration:* top-label ECE is 0.93% before the temperature and 1.13% after. The model is already well calibrated, so the temperature step is neutral for A1 and the thresholds do the work. That gate item is recorded.
- *Consistency repair:* nothing needed for scoring, since every unordered pair gets one label. Linked-RA merging is unscored and skipped (convergent RAs are valid IAT).

### D045 — A1.5 result: complete, valid maps; dev Global about 70.3 / 56.6
- **2026-10-01 · A1.5 · by Claude · Done** (`python -m a1_cross_encoder.predict --run=v1 --split=dev`, `results/a1/v1/dev_global.json`)
- Each dev map is the test-format input, plus A1's relations (tuned thresholds), plus the ILO labeller's L→I links, plus unscored transition illocutions (Arguing / Restating / Disagreeing, D024). **225 / 225 maps pass the strict validator.** The official scorer on the assembled maps reproduces each component's tuned-on-all-dev numbers exactly.

  | Dev (thresholds tuned on all of dev) | General P / R / F1 | Focused P / R / F1 |
  |---|---|---|
  | ARI | 50.21 / 57.03 / **50.90** | 45.08 / 30.56 / **33.91** |
  | ILO | 90.39 / 90.26 / **90.16** | 81.01 / 80.79 / **80.67** |
  | **Global** | 70.30 / 73.64 / **70.53** | 63.05 / 55.67 / **57.29** |

- *Honest (cross-fitted components):* Global about **70.3 general / 56.6 focused**.
- *For reference only (different maps):* 2024's best was 67.05 / 48.25 on the 11 test maps, which come from later debates and were chosen to be rich in arguments. Only the final run decides.

### D046 — A1 improvement, step 1: decoding-level levers are nearly exhausted
- **2026-10-01 · A1 improvement · by Claude · Finding** (requested by user: "improve A1 first")
- *Structure check* (dev, v1 tuned): the model attaches one proposition to several others far more often than gold does. The share of propositions that are the source of 2+ relations of one type:

  | Relation | Gold | Predicted |
  |---|---|---|
  | Inference | 8% | 26% |
  | Rephrase | 1% | 10% |
  | Conflict | 0% | 11% |

  On average 10.9 relations are predicted per map, against a gold 8.3.
- *Fix tried: cap the outgoing relations per proposition and type* (`decide.py`, `tune.py --cap=N`), with thresholds re-tuned and scored by leave-one-episode-out:

  | Honest dev ARI | No cap | Cap 1 | Cap 2 |
  |---|---|---|---|
  | General / focused | 50.50 / 32.61 | **50.95** / 32.20 | 50.58 / **33.57** |

  The cap removes correct Inferences about as often as wrong ones (precision 0.27 → 0.28, recall 0.39 → 0.34). The gains are within dev noise, so **no cap stays the default**, and the option is kept.
- *Checkpoint ensembling* (averaging the saved dev probabilities of several evaluations): epochs 2.0 + 2.5 + 3.0 give 50.15 / 31.43, and 1.5–3.0 give 49.75 / 32.83, both **worse** than epoch 2 alone. Checkpoints of one run make correlated errors, and the later ones overfit.
- *Loss curve:* the training loss (class-weighted) falls steadily to the end (about 1.09 → 0.33), while dev peaks at epoch 2. That is **mild overfitting**, not underfitting.
- *Conclusion:* the remaining gains need a better *model*: more training data, stronger weight on Conflict, or more capacity. They don't come from decoding.

### D047 — A1.6 ablations: one lever off per retrain; no further A1 improvement
- **2026-10-02 · A1.6 · by user (no improvement; do the ablations) / Claude (design) · Done; results in D048**
- *User decision:* don't retrain A1 to improve it (step 2 options from D046 declined). Run the ablations instead.
- *Ablations* (each identical to v1 except one lever; scored by tuned checkpoint selection and leave-one-episode-out dev ARI):
  - **abl-ctx2:** dialogue context limited to ±2 turns around the pair, instead of the whole map (D019).
  - **abl-nomark:** no written distance / same-speaker hints (D021). Turn markers and speaker names in the dialogue stay.
  - **abl-k99:** no pruning, so every pair is compared (D018).
  - **argmax** instead of thresholds: no retrain needed (D044: 50.08 / 33.82 vs 50.50 / 32.61 honest).
- *Left out:* DeBERTa vs ModernBERT. DeBERTa reads only 512 tokens, so it would confound the model with the context length, and it costs ~2–3 h more. Available if the user wants it.
- *Run:* the three retrains are chained in one detached job (~4.5 h).
- *Two problems caught before any ablation training (both fixed):*
  1. **Variable clash:** in `build_pairs.map_pairs` the new pruning parameter `k` was reused as the loop variable that marks dialogue turns. After the first pair of each map the window became the last turn index, so the ctx2 / nomark builds had **no effective pruning** (124,773 train pairs and distances up to 36, against v1's 104,662 and ≤ 10). Each ablation would have changed two levers at once. The parameter now has its own name, and the strict check is that ctx2 and nomark have exactly v1's pairs. v1's own data was unaffected, because it used the constant.
  2. **RAM hang:** the build's token-length check tokenised every pair in one call. On k99 that exhausted WSL's 7 GB, and WSL stopped responding (`WSAETIMEDOUT`). WSL was restarted (`wsl --terminate`; only the stuck build died), and the check now runs in chunks. This is the same lesson as D042: never hold tens of millions of token ids as Python ints.

### D048 — A1.6 result: every component helps focused F1; whole-map context is the only clear gain on general
- **2026-10-03 · A1.6 · by Claude · Finding** (one run each; `results/a1/abl-*/`; honest = leave-one-episode-out dev ARI)

  | Variant | Removed | Honest general (Δ) | Honest focused (Δ) | Notes |
  |---|---|---|---|---|
  | v1 | — | 50.50 | 32.61 | best at epoch 2; 94 min |
  | abl-ctx2 | whole-map context (±2 turns only) | 48.99 (−1.51) | 30.36 (−2.25) | Conflict recall 9.2% → 4.6%; direction 81% → 79%; 51 min |
  | abl-nomark | distance / same-speaker hints | 50.15 (−0.35) | 29.78 (−2.83) | 34% fewer Inferences predicted (precision 0.33, recall 0.31); 90 min |
  | abl-k99 | pruning (every pair) | 50.49 (−0.01) | 28.47 (−4.14) | very cautious: 833 Inference and 6 Conflict predicted; 120 min |
  | argmax | thresholds (same model as v1) | 50.08 (−0.42) | 33.82 (+1.21) | 34 maps with an invented Conflict, against 8 |

- *Reading:* on general F1 only whole-map context matters beyond noise (about 0.5). On focused F1 every lever helps: hints and pruning mostly help the model *find* relations, and without them it predicts far fewer. Pruning is a net win even though it lowers the ceiling (96.5 → 94.4 focused): the model learns better from the less imbalanced pairs, and training is 23% cheaper. Thresholds trade 1.2 focused for 0.4 general and protect against the rare-label trap.
- *Caveat:* one seed per variant, so treat these as indicative. Direction accuracy rises as models predict fewer Inferences (79–86%), because a cautious model only commits to the easier pairs.

### D049 — Final test run for A1 + the ILO labeller, now
- **2026-10-03 · A1.7 · by user · Approved · Done; results in D050** (replaces the earlier "defer to the final run")
- *What runs:* the 11 test inputs go through the ILO labeller (v1, dev-tuned, links already predicted from inputs only), then A1 v1 (epoch-2 checkpoint, temperature and thresholds tuned on all of dev, `results/a1/v1/thresholds.json`), then assembly into complete maps.
- *Rules:* the gold in `data/evaluation/` is read **once, only to score**. Nothing is tuned or changed after the scores are seen. The predicted maps are saved as deliverables in `results/a1/v1/test_predictions/`.
- *Consequence for later approaches:* A1's test number is now fixed. If A2–A4 are built later, they are tuned on dev only and scored on test once each.

### D050 — Final test result: Global 72.01 / 57.53, first against the 2024 field; the gain comes from ILO
- **2026-10-03 · A1.7 · by Claude · Finding** (`results/a1/v1/test_global.json`; predicted maps in `results/a1/v1/test_predictions/`)
- *Protocol:* dev was re-run through the modified script first and reproduced exactly (70.53 / 57.29). On test, the 11 maps were assembled from inputs only, validated (11 / 11) and saved before the gold was read. Nothing was tuned afterwards.

  | Test, F1 general / focused | A1 + ILO labeller | Best 2024 (system) | Rank among 2024 entries |
  |---|---|---|---|
  | ARI | 51.38 / 30.77 | 55.33 (DFKI-MLST) / 35.89 (POKEMON) | 2nd / 2nd |
  | ILO | **92.64 / 84.28** | 81.17 (PUNGENE) / 72.09 (RoBERTa baseline) | 1st / 1st |
  | **Global** | **72.01 / 57.53** | 67.05 / 48.25 (DFKI-MLST) | **1st / 1st** |

  Global P / R: 72.61 / 73.25 general, 67.05 / 54.81 focused.
- *Reading:* the overall win comes from **ILO (+11.5 / +12.2 over 2024's best)**. **ARI is at about 2024 level:** −4.0 general against DFKI, level with DFKI on focused (30.77 vs 30.40), −5.1 against POKEMON. Dev predicted test well (ARI 50.50 → 51.38, ILO 90.08 → 92.64).
- *Test diagnostics* (pooled):

  | Relation | Gold | Predicted | Precision | Recall | F1 |
  |---|---|---|---|---|---|
  | Inference | 65 | 59 | 0.31 | 0.28 | 0.29 |
  | Conflict | 13 | 3 | 0.67 | 0.15 | 0.25 |
  | Rephrase | 46 | 63 | 0.46 | 0.63 | 0.53 |

  Inference direction is right for 78% of the 23 found. No invented Conflict in any map; one map has an invented Rephrase.
- *Variance:* per-map ARI general ranges **23.5 to 77.0**, so each of the 11 maps moves the average by about 9%. Per-map ILO is perfect (100 / 100) on 7 maps; maps 1, 2 and 9 sit around 73 general.
- *Cost:* A1 test inference 10.4 s for all 11 maps (0.94 s per map); the ILO labeller takes about 0.21 s per map.

### D051 — A2 plan: whole-map Claude, sampled confidence, dev kept clean
- **2026-10-04 · A2 · by Claude · Approved by user 2026-10-04, amended by D052–D053** (model, route, budget and ablations changed; the rest stands). Full plan: [`reports/A2-whole-map-llm.md`](reports/A2-whole-map-llm.md).
- *Goal:* fix what A1 can't, because A1 judges each pair alone: choosing between conclusions, direction, and Conflict (recall 9–15%). Do it at a measured cost, and save per-pair outputs that the Combination can reuse without new API calls.
- *Design:*
  - **Model:** Claude Opus 5.5 (`claude-opus-5-5`, $4 / $20 per MTok; thinking can't be turned off, so effort is set explicitly). The pilot also tries Sonnet 5.5 (half the price) and Fable 5.1 (2.5×) to see whether model capability matters.
  - **Input:** the whole dialogue in order. Each turn shows its speaker, its text and the proposition(s) the ILO labeller links to it. IDs are renumbered P1… / T1… in dialogue order, so node IDs never leak (D029). Train maps use gold links (D040).
  - **Output:** structured outputs, with a JSON schema whose ID enums come from the map. The output holds relations {type, from, to, confidence} plus an illocution type for each linked turn.
  - **Confidence:** 5 samples per map, scored as a confidence-weighted vote share. Opus 5.5 doesn't accept a temperature setting, so the stated confidence covers the case where samples barely differ.
  - **Decision layer:** A1's (per-type thresholds, leave-one-episode-out). Conflict and Rephrase direction comes from dialogue order (D020).
  - **ILO:** links stay with the labeller. The LLM's types replace the labeller's only if that beats 90.08 cross-fitted.
  - **Prompt development uses train-split maps** (about 50, from 3 episodes, kept out of the example pool). Dev is then used only for the thresholds, and stays an honest, A1-comparable estimate.
  - **Reproducibility:** every response is cached on disk by a hash of the request, and all scores are recomputed from the cache.
- *Changes to PLAN.md's A2 section (applied on approval):*
  - the pilot runs on train maps, not dev;
  - "reasoning off" becomes low effort, because Opus 5.5 can't disable thinking;
  - stated confidence is added to the votes;
  - n is swept for free, by subsampling the 5 stored samples;
  - a contamination probe is added;
  - outputs are saved in A1's pair format for the Combination.
- *Estimated cost:* about $150–200 on Opus 5.5: about $1 smoke test, $50 pilot, $75 dev (5 samples, Batch API), $50 ablations and $5 test. Thinking tokens are the main unknown, and A2.1 measures them.
- *Prerequisite:* an Anthropic API key (none is set in Windows or WSL). *Superseded by D052: no key is needed.*

### D052 — A2 runs Claude Sonnet 5.5 on the user's Claude Pro subscription, at $0
- **2026-10-04 · A2 · by user · Approved**
- **Model:** Claude Sonnet 5.5 (`claude-sonnet-5-5`) only, with no other model in any phase ("good enough for simple decisions such as these"). This replaces D051's Opus 5.5, its pilot comparison with Sonnet 5.5 and Fable 5.1, and any refusal fallback to another model.
- **Route:** no API spend. Calls go through Claude Code's scripted mode (`claude -p`) on the user's existing Claude Pro login.
  - Already installed and logged in: Claude Code 2.1.283 on Windows, reachable from WSL through interop; `claude auth status` shows `authMethod: claude.ai`, `subscriptionType: pro`.
  - *Flags:*
    - `--safe-mode`, because `--bare` ignores subscription logins;
    - `--tools ""` and `--disallowedTools "mcp__*"`;
    - `--system-prompt-file`, `--json-schema`, `--effort`, `--output-format json`;
    - `--no-session-persistence`.
  - *Alternatives considered:* the paid API (≈ $75–100 for this plan); other providers' free tiers (Gemini API, GitHub Models, OpenRouter free models, Groq / Cerebras), which mean a different model with changing limits; a local open model on the 8 GB GPU, which is far weaker and is A3's no-fine-tuning baseline.
- *Consequences:*
  - **No Batch API** and no API key.
  - **Speed is bounded by Pro's usage limits** (5-hour and weekly windows, shared with the user's own Claude use).
  - Spend is $0. The API-equivalent cost is still logged per call (`total_cost_usd`), as PLAN Principle 4 asks.
  - Every answer's `modelUsage` must show Sonnet 5.5 only; anything else is discarded.
- **At a usage limit the runner pauses** and the user restarts it ("resume when I say"). The on-disk answer cache makes restarts lossless.
- **The 5 samples use the same prompt** (approved; *3 samples since D054*). Confidence = the average stated confidence over the samples.
- **Raw answers are committed**, gzipped, per run (`results/a2/<run>/responses.jsonl.gz`), because runs can't be regenerated exactly.
- *Key storage* (asked before the route changed; the user chose WSL `~/.profile`): it now applies only to the fallback. If interop fails, Claude Code is installed in WSL, and the token from `claude setup-token` goes in `~/.profile` as `CLAUDE_CODE_OAUTH_TOKEN`.

### D053 — A2 ablations: the free set plus one paid ablation (minimal reasoning)
- **2026-10-04 · A2 · by user (selection) / Claude (list) · Approved**
- *Free* (re-scored from saved answers, always run):
  - 1 / 3 / 5 samples;
  - votes vs stated confidence;
  - thresholds vs majority vote;
  - direction rule vs the model's own direction;
  - relations more than 10 turns apart dropped;
  - LLM vs labeller turn types.
  - The pilot sweeps also give k = 0/2/4/8 and effort low/medium/high on pilot maps.
- *With new calls:* **minimal reasoning only.** A dev pass at effort `low`, with 1 sample, compared with sample 1 of the main run (225 calls, or 90 on 2 dev episodes if quota is tight).
- *Dropped by the user:* no-definitions, random examples, and the contamination (memorisation) probe. Contamination is reported as a caveat, with the free dev → test comparison against A1 in A2.5.
- *Expected total:* about 1,800 model calls, at $0 spend and an estimated $50–150 API-equivalent. *Updated by D054: about 1,350 calls.*

### D054 — 3 samples per map instead of 5
- **2026-10-04 · A2 · by user · Approved**
- Every A2 run asks the model 3 times per map (dev 675 calls, test 33); the free sample ablation becomes 1 / 2 / 3, and majority vote becomes ≥ 2 of 3.
- Total ≈ 1,350 calls (≈ $40–110 API-equivalent, $0 spent). Scores are coarser, which matters most for the rare-label thresholds; stated confidence still grades them.

### D055 — A2.1 result: the harness is lossless and Sonnet-only; Sonnet doesn't think by default
- **2026-10-04 · A2.1 · by Claude · Finding** (`results/a2/check.json`, `results/a2/smoke*/`; report §9)
- *Built:*
  - `common/serialise.py`;
  - in `a2_llm/`: `schema.py`, `prompt.py` (TF-IDF example pool), `claude_cli.py`, `runner.py` (content-addressed answer cache, re-judging, pause at the usage limit), `parse.py` and `check.py`;
  - the draft prompt `prompts/system_v0.md`.
- *Gate:*
  - **round trip 98.92 / 96.51** on 225 dev maps, exactly the no-pruning ceiling, with 0 invalid maps;
  - 3 samples were read by eye;
  - prompts are 5.5k / 21.5k / 37.6k characters (median) at k = 0 / 4 / 8;
  - 8 valid answers;
  - a re-run sends 0 calls.
- *Two fixes found by the smoke test:*
  1. Claude Code sends a side request to **Haiku 4.5 to name the session**. `CLAUDE_CODE_DISABLE_TERMINAL_TITLE=1` removes it: it must be forwarded with `WSLENV=…/w` (`/u` forwards the wrong way, which was the first attempt's bug). Effort is pinned the same way, through `CLAUDE_CODE_EFFORT_LEVEL`, because the user's global settings set `effortLevel: medium`. The verification call shows Sonnet 5.5 only. The 3 earlier answers are kept: Sonnet wrote them, and Haiku only wrote a ~15-token title.
  2. The runner **retried a failure that was certain to repeat** (the strict model check), wasting 6 of the 12 calls. Now only transient failures are retried, and acceptance is re-judged from the saved results.
- *Measured:*
  - 12 calls, 4–9 s each;
  - ~6–8k input tokens at k = 2, of which the cached prefix is ~2k (system prompt + schema tool, no Claude Code prompt);
  - **$0.374 API-equivalent** in total (Claude Code reports $0.795, with `costBasis: unknown` for Sonnet 5.5); $0 spent.
- *Findings for A2.2:*
  - **thinking tokens were 0 at effort medium and high.** Sonnet 5.5's adaptive thinking chose not to reason before writing the structured answer, and it can't be forced on or off;
  - the samples vary (19 vs 21 relations on one map);
  - it over-links (13 and 19–21 relations listed against 11 and 12 in gold);
  - the whole plan is only a few hours of calls; Pro quota is the bottleneck.
- *Proposed for A2.2* (adds 3 settings to the pilot sweep): before the effort sweep, test what makes Sonnet reason: a think-first instruction in the system prompt, the `ultrathink` keyword, and effort `xhigh`. If none of them does, the minimal-reasoning ablation (D053) has nothing to compare against and is replaced by a comparison of these levers. *Tested at once (user's request): D056.*

### D056 — Only effort `xhigh` makes Sonnet 5.5 reason; the effort sweep becomes medium vs xhigh
- **2026-10-04 · A2.2 · test requested by user, result by Claude · Finding + Done** (`results/a2/reason-*/`; 9 calls, $0.33 API-equivalent)
- *Test:* the 3 smoke maps, k = 2, 1 sample each, against the medium baseline (0 thinking tokens on 3/3).

  | Lever | Answers that thought | Thinking tokens | Seconds |
  |---|---|---|---|
  | Think-first instruction in the system prompt (`system_v0think.md`) | 0 / 3 | 0 | 6–8 |
  | `ultrathink` keyword in the user message | 1 / 3 | 0–667 | 4–11 |
  | **Effort `xhigh`** | **3 / 3** | 805–1,435 | 11–19 |

- *Consequences:*
  - **Sweep B is now effort `xhigh` against medium.** Medium comes from sweep A, so only 50 new calls are needed. Low and high would reason no more than medium, so they're dropped (100 calls saved).
  - **The minimal-reasoning ablation (D053)** is now a dev pass at whichever of medium / xhigh is *not* chosen: thinking off vs on.
  - The think-first instruction and `ultrathink` are not used.

### D057 — The pilot shrinks to 20 maps from sweep B on
- **2026-10-04 · A2.2 · by user · Approved** ("this is taking too long … can try 20 maybe")
- **Sweep A** (k = 0/2/4/8, effort medium) was almost done when the user asked, so it finished and is scored on all 50 maps (`maps_50`).
- **Every later pilot run uses 20 maps** (`maps` in `results/a2/pilot_maps.json`): sweep B (`xhigh`) and the prompt revisions.
  - The 20 are a size-stratified subset of the 50 (4–26 propositions, median 14), so sweep A's answers on them are reused for the medium vs xhigh comparison.
  - Pilot calls from here on: ≤ 20 per setting (≤ 60 for sweep B and two revisions, instead of ≤ 150).
- *Cost:* comparisons on 20 maps only separate large differences (roughly 5+ points). Close calls go to the cheaper setting, as before.

### D058 — Pilot sweeps: k = 4 worked examples, effort `xhigh`; reasoning adds ~11 focused points
- **2026-10-04 · A2.2 · by Claude (pre-set rules) · Done** (`results/a2/pilot_table.json`; 1 sample per map, scored with every listed relation kept, and with medium + high only)
- **Sweep A, examples** (50 maps, effort medium, prompt v0; 200 calls):

  | k | ARI all listed | ARI medium+ | Inference direction | Input tokens |
  |---|---|---|---|---|
  | 0 | 45.0 / 26.4 | 48.5 / 24.3 | 80% | 3.6k |
  | 2 | 47.6 / 30.5 | 50.8 / 29.2 | 83% | 6.8k |
  | 4 | 49.5 / 30.6 | 51.2 / 29.1 | 81% | 10.1k |
  | 8 | 50.7 / 31.5 | 52.2 / 30.3 | 85% | 16.5k |

  More examples help, with shrinking gains. k = 8 leads k = 4 by 1.2 / 0.9, within noise, so **k = 4**: 40% fewer input tokens.
- **Sweep B, reasoning** (20 maps, k = 4, prompt v0; 20 calls). Medium scores 50.5 / 32.5 (all listed) and 52.7 / 30.4 (medium+), with 79% direction and 0 thinking tokens. **`xhigh` scores 55.0 / 44.0 and 58.9 / 39.5, with 88% direction** and ~970 thinking tokens, at 12.7 s and ~$0.05 per call. **Effort `xhigh`** is chosen: +4.5 general and +11.5 focused is well beyond noise.
- *Errors* (k = 4, medium, 50 maps, `results/a2/pilot_errors_pilot-k4.md`):
  - Inference: 167 invented against 99 right; invented relations sit further apart (median 2 turns vs 1);
  - 47 gold Rephrases read as Inference;
  - Conflict: 18 invented, 14 missed, 7 right.
  - Gold is sparse: 48% of propositions start no relation, 48% start one, and 5% start two or more.
- *Revision 1* (`system_v1.md`): Rephrase covers restating, narrowing or a sharper version of the same point; Inference needs a *different* point as reason or evidence; the sparsity facts are added. Conflict is unchanged (the thresholds handle it).

### D059 — A2.2 result: frozen configuration = prompt v1, k = 4 retrieved examples, effort `xhigh`
- **2026-10-05 · A2.2 · by Claude (pre-set rules) · Done**
- *Revision 1* (20 maps, k = 4, xhigh; 20 calls):

  | Prompt | ARI all listed | ARI medium+ | Invented Inferences | Rephrases read as Inference |
  |---|---|---|---|---|
  | v0 | 55.0 / 44.0 | 58.9 / 39.5 | 64 | 20 |
  | **v1** | **57.2** / 42.8 | **61.9 / 42.0** | 52 | 13 |

  v1 is better on 3 of 4 scores and fixes what it targeted. **No second revision:** the remaining gaps are within the noise of 20 maps, and over-linking is left to the 3-sample thresholds.
- **Frozen for A2.3–A2.5:** Sonnet 5.5, `system_v1.md`, 4 TF-IDF-retrieved worked examples, effort `xhigh`, 3 samples.
- *Pilot cost:* 249 calls (sweep A 200, sweep B 20 + 20 reused, revision 20, reasoning test 9) and no usage limit hit. A2.3 is 675 calls at ~13 s ≈ 2.5 h of calls.
- *Caveat:* pilot scores are on *training* maps with gold links and 1 sample, so they are not comparable with A1's dev numbers. A2.3 gives the comparable score.

### D060 — A2 runs auto-resume after usage limits (replaces "resume when I say")
- **2026-10-05 · A2.3 · by user ("plan to continue when usage resets") · Approved · Done**
- *What happened:* the dev run (`dev-v1`, 2 calls at a time) hit the Pro **5-hour session limit after 76 of 675 answers** ("resets 3:10am (Europe/London)"). At effort `xhigh`, a session holds far fewer calls than the pilot's medium-effort calls suggested. The remaining 599 calls need about 8 sessions, so roughly 1.5–2 days of calendar time, unless the weekly limit binds first.
- *Change:* `runner.py --until-done`. At a usage limit the runner reads the reset time in the message, sleeps until 5 minutes after it, and continues. This covers the 5-hour form and the weekly form with a date, and an unreadable message retries in 30 minutes. Answers are saved one by one, so nothing is lost or re-sent. This replaces D052's "pause; resume when I say" for A2's runs.
- *Note for the user:* while it runs, it uses each fresh Pro session as soon as it resets, which competes with their own Claude use. Stopping it is safe at any time.

### D061 — A full Pro session holds ~450 `xhigh` calls; the runner sleeps by the wall clock
- **2026-10-05 · A2.3 · by Claude · Finding + Done**
- *Capacity:* the session after the 3:10 am reset produced **457 answers** (2 at a time, ~50 minutes) before the limit. The first session's 76 was low because the pilot had already used most of it. 533 of 675 dev answers were done by ~04:00.
- *Bug:* the runner slept with one `time.sleep(seconds)`. While the PC slept, WSL was paused and that timer stopped counting, so a wake-up due at 08:15 hadn't happened by 11:27. It now sleeps in 60-second steps against the real clock. The idle runner was stopped and relaunched at 11:28; no answers were lost.

### D062 — A2.3 result: honest dev ARI 54.63 / 34.12, above A1's 50.50 / 32.61
- **2026-10-05 · A2.3 · by Claude · Finding** (`results/a2/dev-v1/dev_report.json`, `thresholds.json`; `python -m a2_llm.tune --run=dev-v1`)
- *Run:* 225 dev maps × 3 answers at the frozen settings (D059), with the ILO labeller's turn links. 675/675 valid, Sonnet-only, 13.7 s per call, $35.61 API-equivalent, $0 spent.
- *Dev ARI:*

  | | General | Focused |
  |---|---|---|
  | A1, honest (D044) | 50.50 | 32.61 |
  | A2, every listed relation (no thresholds) | 52.07 | 39.41 |
  | A2, tuned on all of dev (optimistic) | 54.81 | 34.16 |
  | **A2, cross-fitted (honest)** | **54.63** | **34.12** |

- *Thresholds (kept for test):* Inference 0.35, Conflict 0.50, Rephrase 0.25. They are the same in 4 of 5 folds; one fold picks Conflict 0.35.
- *Per type (pooled, tuned), P / R / F1:*
  - Inference 0.39 / 0.35 / 0.37 (A1: 0.27 / 0.39 / 0.32);
  - Conflict 0.43 / 0.12 / 0.19 (A1: 0.31 / 0.09 / 0.14);
  - Rephrase 0.48 / 0.55 / 0.51 (A1: 0.42 / 0.53 / 0.47).
- *Other diagnostics:*
  - Inference direction is right for 85% of the 432 found (A1: 81%);
  - 7.9 relations per map against 8.3 in gold (A1: 10.9);
  - 0.7% of propositions start 2+ relations against 5.7% in gold, so it under-draws divergent arguments;
  - 8 maps have an invented Conflict (A1: 8).
- *Run-to-run:* one answer alone scores 52.1–53.0 general (every listed relation); 3 answers with thresholds add about 2.
- *Overlap with A1* (both tuned on all dev): of the pairs where gold or either system has a relation, both are right on 509, only A2 on 1,173, only A1 on 622, and neither on 1,186. **The oracle per-pair choice scores 66.1 / 46.0**, the upper bound for the combination phase.
- *Reading:*
  - A2 wins on general (+4.1) and focused (+1.5), with better precision on every type.
  - Thresholds tuned for general give up focused F1 (39.4 → 34.1), as for A1.
  - Conflict recall stays low (12%).

### D063 — A2.4 result: honest dev Global 72.53 / 57.63; LLM turn types adopted; reasoning adds only ~1 on dev
- **2026-10-05 · A2.4 · by Claude (pre-set rules) · Finding + Done** (`results/a2/dev-v1/a24_report.json`; `python -m a2_llm.evaluate`; reasoning run `dev-medium`: 225 calls, $9.39 API-equivalent)
- *Turn types:*
  - honest dev ILO goes from **90.16 / 80.67** (labeller alone) to **90.43 / 81.13** (with LLM overrides);
  - the overrides that survive tuning are Asserting → Pure Questioning (9 right, 3 wrong) and 1 Default Illocuting;
  - the other types are never overridden (threshold 1.01).
  - **Adopted** (rule: beat the labeller alone).
- *Complete maps:* 225/225 valid. Honest dev **ARI 54.63 / 34.12, ILO 90.43 / 81.13, Global 72.53 / 57.63**, against A1's 50.50 / 32.61, 90.08 / 80.64 and 70.29 / 56.63.
- *Free ablations* (honest dev ARI; main 54.63 / 34.12):
  - 1 answer: 53.53 / 33.03;
  - 2 answers: 53.34 / 34.69;
  - plain votes: 53.60 / 35.57;
  - majority vote ≥ 2 of 3 (untuned): 53.34 / 36.14;
  - the model's own Conflict / Rephrase direction: unchanged (54.83 vs 54.81, tuned on all dev);
  - relations over 10 turns dropped: −0.05 (11 relations, 4 right).
- *Reasoning on / off* (1 answer each, honest):
  - **`xhigh` 53.53 / 33.03 vs medium 52.72 / 32.00 (+0.8 / +1.0)**;
  - unthresholded, 52.09 / 35.24 vs 52.05 / 33.54;
  - unchanged: Inference direction (85% / 85%) and Conflict F1 (0.21 / 0.22);
  - what reasoning does change: Rephrase recall 0.57 vs 0.45, and invented-Conflict maps 12 vs 17.
  - Cost: 13.7 s vs 5.1 s per call.
- *Reading:*
  - **The pilot's +4.5 / +11.5 for reasoning did not replicate on dev.** Likely causes: the pilot compared reasoning under prompt v0 (v1 fixes much of what reasoning compensated for), 20-map noise, and annotated links.
  - A pilot effect's size must be confirmed on the larger set.
  - The frozen settings (`xhigh`) stay for the test run as decided before dev was seen. Medium is the efficient choice for future runs and the combination.

### D064 — A2.5 result: test Global 73.67 / 58.27; ARI 55.77 / 33.34; the LLM turn types cost 1.1 ILO
- **2026-10-05 · A2.5 · run on the user's go · Finding** (`results/a2/test_global.json`, `test_predictions/`, `test_posthoc.json`)
- *Protocol:*
  - 11 test inputs × 3 answers (33 calls, valid, Sonnet-only, $1.93 API-equivalent);
  - dev-fitted relation thresholds and turn-type rule;
  - maps assembled, validated (11/11) and saved before the gold was read, then scored once, with nothing changed after.

  | Test F1, general / focused | ARI | ILO | Global |
  |---|---|---|---|
  | A1 + labeller (D050) | 51.38 / 30.77 | 92.64 / 84.28 | 72.01 / 57.53 |
  | **A2 + labeller + LLM types** | **55.77 / 33.34** | 91.56 / 83.20 | **73.67 / 58.27** |

- *Against the 2024 field:* Global 1st / 1st; ARI general 1st (DFKI 55.33); ARI focused 2nd (POKEMON 35.89).
- *Per type (pooled), P / R / F1:*
  - Inference 0.48 / 0.40 / 0.44 (A1: 0.31 / 0.28 / 0.29);
  - Conflict 0.50 / 0.15 / 0.24 (2 of 13; A1 also 2 of 13);
  - Rephrase 0.46 / 0.54 / 0.50 (A1: 0.46 / 0.63 / 0.53).
  - Direction is right for 90% of the Inferences found (A1: 78%), and no map gets an invented type.
- *Per map:* A2's ARI beats A1's on 7 of 11 maps (+12 to +15 on maps 3, 7 and 8; −7 on map 4, −5 on map 2).
- *Turn types:* 3 LLM changes on test, all Asserting → Pure Questioning and all wrong. Maps 1 and 2 got a type absent from gold (the rare-label trap), costing 1.1 ILO. *Post-hoc only (not submitted):* with the labeller's types, Global is 74.20 / 58.81.
- *Contamination:* the dev → test ARI change is +1.1 for A2 against +0.9 for A1, so there's no sign of memorised test answers.
- *A2 totals:* 1,194 calls, $0 spent, $57.08 API-equivalent.

### D065 — A3 plan: a QLoRA-tuned Qwen3 writes each map's relations; sampled votes feed A1's decision layer
- **2026-10-05 · A3 · by Claude · Approved by user 2026-10-05** ("go ahead and start A3.1"; revised the same day after an independent review, before approval). Open items take their stated defaults: the model rule, pre-quantized weights only if RAM requires, and the retrain ablations confirmed at A3.5. The full plan is in [`reports/A3-finetuned-llm.md`](reports/A3-finetuned-llm.md). PLAN.md's Approach 3 section is rewritten (done); phases A3.1–A3.6 are in the phase map.
- *Question:* A2's discussion found that the encoder learns *the annotators* and the LLM applies *the definition*. Can an LLM that learns the annotators through its weights keep the LLM's precision and direction, and also gain the encoder's neighbour-Inference recall and turn types? Concessive Conflict too, if possible.
- *Design:*
  - **Input:** A2's serialised map after a short fixed instruction, with no worked examples and no reasoning. Training maps use gold links (D040); dev and test use the labeller's links.
  - **Target:** A2's answer without `confidence`, written as compact JSON with exact separators: illocutions, then relations in P order.
  - **Training data:** the 1,231 test-like training maps. The giant maps (no turn links; 528 of 2,592 propositions unplaceable, D022 / D030) are a retrain ablation, chunked to ≤ 35 propositions.
  - **Method:** QLoRA (4-bit NF4 layers; LoRA r = 16, α = 32, all linear layers; answer-only loss; 3 epochs; an adapter every half epoch).
    - *Why:* on 8 GB, full fine-tuning fits only under ~0.5B (ModernBERT-large already used 7.44 GiB, D009), and 16-bit LoRA only up to ~2–3B.
    - *Memory path built in:* we wrap the model in PEFT ourselves, so TRL doesn't copy the 16-bit embeddings to 32-bit, and we use a fused loss (Unsloth, or TRL + Liger).
  - **Model rule:** Qwen3-8B if, at the longest training example, its peak is ≤ 7.5 GiB, training runs at ≥ 200 tokens/s, and generation fits a batch of ≥ 2. Otherwise Qwen3-4B-Instruct-2507. No maps are capped to make 8B fit.
    - *Why borderline:* Qwen3-8B's untied 16-bit embeddings take ~2.5 GB, so its weights are ~6.2 GB; the 4B's (tied) are ~2.7 GB. Configs checked on Hugging Face.
    - *Other models checked on the web, 2026-10-05:*
      - Qwen3.5 (Mar 2026): Unsloth advises against 4-bit training, and its 16-bit 4B needs 10 GB.
      - Gemma 4: E4B needs 10 GB even in 4-bit; E2B sits at 8 GB.
      - Llama 3.1 8B is older, gated and custom-licensed.
      - ≥ 12B and mixture-of-experts models don't fit.
  - **Prediction:** turn types written once (greedy, with their probabilities), then the relations sampled 8 times from that fixed start (seeded). A grammar built from each map's schema, with no free whitespace, keeps answers valid, and answer length is capped.
  - **Pair scores:** vote shares in A1's 5-label format. Token probabilities are kept as well.
    - *Diversity rule (A3.3, fixed now):* if fewer than 10% of listed relations get a vote share strictly between 0 and 1, token probability becomes the main score.
    - Pair scores are then decided by A1's thresholds, cross-fitted by episode, and assembled by `build_map`.
  - **Checkpoint:** best greedy dev ARI over the 6 half-epoch adapters (as A1, D043).
  - **Turn types:** links stay with the labeller. LLM types replace the labeller's only if they gain ≥ 0.5 honest ILO *and* win in ≥ 4 of 5 dev folds; this is stricter than A2's rule, after D064.
  - **Environment:** a separate venv `~/.venvs/a3`, so A1, A2 and the ILO labeller keep working.
- *Success (honest dev, fixed now):* A3 succeeds if either holds:
  1. **Beats A2:** A3's ARI general is above A2's 54.63, with a paired per-map bootstrap 95% interval above 0.
  2. **Adds to the combination:** a cross-fitted average of A1 + A2 + A3 pair scores beats A1 + A2 alone by ≥ 1.0 general.
- *Hypotheses, with decision rules:* H1 and H2 are measured at matched density (8.3 relations per map).
  - H1: neighbour-Inference recall ≥ A2 + 0.05.
  - H2: invented relations ≤ 90% of A1's.
  - H3: Inference direction ≥ 85%.
  - H4: within-speaker Conflict recall ≥ 15%.
  - H5: the strict turn-type rule passes.
  - H6: fine-tuned beats untrained zero-shot by ≥ 5 general.
- *Ablations:*
  - *Free:* samples 1 / 2 / 4 / 8 · votes vs token probabilities · greedy vs votes · thresholds vs majority vote · score per half epoch · direction rule · LLM vs labeller types.
  - *Generation:* untrained zero-shot · no constrained decoding.
  - *Retrain:* with the giant-map chunks (recommended) · 4B vs 8B (only if 8B fits).
- *Review:* an independent reviewer agent rated the first draft "approve with changes", and every finding was adopted (report §12). The main ones:
  - the 8B memory case;
  - a success criterion that couldn't fail;
  - hypotheses with no decision rules;
  - a confounded Qwen3.5-2B ablation, now dropped;
  - A2's schema and parser requiring `confidence`;
  - giant maps in the main run.
- *Changes to PLAN.md's A3 section (on approval):*
  - the separate venv;
  - vote shares, with token probabilities kept, replace answer-token probabilities alone;
  - "split long maps" becomes "giant maps as a chunked retrain ablation";
  - the model rule;
  - the success test and the hypothesis rules;
  - the ablation list above.
- *Cost:* $0, on the local RTX 5060 Ti with no API calls. Roughly 15–35 GPU hours.
- *Open for the user:*
  1. Approve the model rule (8B if it fits, else Qwen3-4B-Instruct-2507, locally at $0), rather than a free Kaggle / Colab GPU or a paid rental.
  2. If WSL's 7 GB RAM can't load 8B: pre-quantized weights (default) or raising WSL memory in `.wslconfig`.
  3. Which retrain ablations to run (giant chunks recommended; 4B vs 8B only if 8B fits).

### D066 — A3.1 result: gate passed; Qwen3-8B fails the memory rule, so A3 uses Qwen3-4B-Instruct-2507
- **2026-10-05 · A3.1 · by Claude (pre-set rule) · Finding + Done** (`results/a3/check.json`, `probe_qwen3-8b.json`, `probe_qwen3-4b.json`, `samples/`; `requirements-a3.txt`; `python -m a3_finetune.check` and `python -m a3_finetune.probe`)
- *Environment* (`~/.venvs/a3`, pinned in `requirements-a3.txt`):
  - **Packages:** torch 2.14.0+cu130, transformers 5.18.0, peft 0.21.2, bitsandbytes 0.50.2, liger-kernel 0.8.4, xgrammar 0.2.8.
  - **4-bit works on this GPU:** bitsandbytes' NF4 runs on the RTX 5060 Ti (sm_120) under CUDA 13.
  - **Unsloth not installed:** its current release pins torch 2.12.1 and transformers 5.5, and the 4B fits without it.
- *Training loop:* our own small loop rather than TRL's SFTTrainer (TRL is installed but unused). It makes D065's memory path explicit and checkable:
  - the model is wrapped in PEFT directly, so nothing is copied to 32-bit;
  - the loss reads hidden states at answer positions only, into Liger's fused cross-entropy.
- *Gate checks (no model):*
  - **Lossless round trip:** 98.92 / 96.51 on the 225 dev maps; 0 invalid maps, 0 schema failures.
  - **Grammar:** every gold answer is accepted and ended by its map's grammar: 1,231 train, 225 dev, 67 giant chunks, 0 rejected.
  - **No leakage:** no dev map name or identical map text in train; no test map text in train.
  - **Tokens** (Qwen tokenizer, prompt + answer): train median 1,182, max 2,634; answers median 301, max 809; test prompts at most 1,712.
  - **Giant chunks:** 67 chunks from 19 maps. 545 of 2,592 propositions can't be placed. Only 719 of 1,453 relations are kept (49%): 493 are lost to unplaced propositions and 241 to window edges. Chunks run to 4,217 tokens.
- *Memory probe.* The Windows desktop already holds ~1.1 GiB of the 7.93 GiB card.

  | | Qwen3-8B | Qwen3-4B-Instruct-2507 |
  |---|---|---|
  | Loaded (PyTorch / whole card) | 5.82 / 7.32 GiB | 2.59 / 3.79 GiB |
  | 1,200-token example, forward + backward, no optimizer | 6.98 GiB reserved; card full (7.93 used) | — |
  | Longest examples (~2.6k tokens), with optimizer | **Doesn't run.** Crashes at 1,797 tokens in the sweep, and at the longest example in the training probe ("device not ready": WSL can't map more GPU memory). An earlier diagnostic run spilled into system memory instead: 173 s for one 2,590-token step | **4.25 GiB peak, 789 tokens/s** (20 steps) |

  - **Rule (D065): 8B fails** (it exceeds 7.5 GiB well before the longest example). **A3 uses Qwen3-4B-Instruct-2507.** No examples were capped.
  - *Consequences:* the 4B vs 8B retrain ablation is dropped, since size can't be compared on this card. H6 keeps only "fine-tuned vs untrained".
- *Generation probe* (untrained 4B, 5 dev maps):
  - 55/55 answers valid under the grammar.
  - Greedy decoding runs at ~22 tokens/s; 8 samples in one batch at ~63 tokens/s, peaking at 5.67 GiB.
  - The untrained model writes empty lists for 3 of the 5 maps.
- *Projections:*
  - **Training:** an epoch is ≈ 1.5M tokens ≈ 33 min, so 3 epochs take ≈ 1.7 h.
  - **Dev run** (greedy + 8 samples × 225 maps): ≈ 2 h.
  - **Checkpoint scoring:** 6 checkpoints, greedy on all 225 dev maps one map at a time, would take ~5 h. So A3.2 adds greedy decoding batched across maps.

### D067 — A3.2 result: the training loop memorises 32 maps exactly; the full run is projected at ~2 h
- **2026-10-05 · A3.2 · by Claude · Finding + Done** (`results/a3/overfit/`: `train_log.jsonl`, `config.json`, `overfit_report.json`, `ep10.0_train_greedy.jsonl.gz`)
- *Built:*
  - **`train.py`:**
    - one map per forward pass, gradient accumulation with every answer token weighted equally;
    - paged 8-bit AdamW, linear warm-up and decay, gradient clipping;
    - an adapter every half epoch, with the teacher-forced dev answer loss.
    - Before every run it checks the loss mask: the labelled tokens must decode to exactly the gold answer plus `<|im_end|>`, with the prompt fully masked.
  - **`runner.py`:** cached generation.
    - Answers are keyed by base model, a hash of the adapter weights, prompt, mode, seed, length cap and grammar settings.
    - Greedy decoding is batched across maps (left-padded, ≤ 16k tokens per batch); samples continue the greedy answer.
    - `--export` writes committed `.jsonl.gz` files.
  - **`evaluate.py`:** the overfit gate, plus greedy dev ARI per adapter and the checkpoint rule (for A3.3).
- *Overfit test:* 32 seeded training maps, 10 epochs, 320 updates.
  - Loss 0.41 → 0.0003.
  - 8.6 min in total: 634 tokens/s overall, ~720 while training; peak 3.83 GiB.
- *Gate:*
  - Greedy answers are **identical to gold on 32 / 32 maps**: 206 / 206 relations (precision and recall 1.0), every turn type right, all valid and finished.
  - Masking verified: 8,323 answer tokens trained on, 24,555 prompt tokens masked.
- *Speed:* batched greedy takes ≈ 5.5 s per map at 8 maps per batch, against ≈ 14 s one at a time. Greedy on the 225 dev maps is therefore ≈ 21 min per adapter.
- *Projection for A3.3:*
  - An epoch is 1.48M tokens, 393k of them answer tokens: ≈ 34 min.
  - 3 epochs plus 6 dev-loss passes (270k tokens each) take ≈ 2 h.
  - Greedy scoring of the 6 adapters takes ≈ 2 h more, run after training.

### D068 — Memory: WSL returns its file cache to Windows; the 4-bit model is loaded from a saved copy; samples split under a token budget
- **2026-10-06 · A3.3 · by user (WSL setting, approved) / Claude · Done** (`results/a3/quantized_check.json`)
- *What happened:* Claude Code stopped the checkpoint-scoring job twice because Windows ran critically low on memory (1.4 of 15.5 GB free).
  - WSL held 5.0 GB, of which Linux itself used 0.5 GB. The rest was file cache, filled each time the 8 GB original weights were read to be quantized, and WSL doesn't return that to Windows by itself.
  - Firefox held another ~3.5 GB.
  - Separately, the diversity check ran out of GPU memory: 8 samples of a long map in one batch.
- *Done:*
  1. **`C:\Users\Kunal\.wslconfig`** now sets `[experimental] autoMemoryReclaim=dropcache` (user-approved), followed by `wsl --shutdown`.
  2. **The 4-bit base model is saved once** to `~/.cache/a3/Qwen3-4B-Instruct-2507-nf4` (2.47 GB, on the Linux filesystem) and loaded from there.
     - Check: all 1,659 tensors are bit-identical to the quantize-on-load model, so cached answers stay valid.
  3. **Samples are split under the token budget,** as greedy batches already were: a map's 8 samples run in as few equal calls as the budget allows (e.g. 4 + 4), with call c using seed + c. Rows per call are in the cache key and the record.
- *After the fix:* the remaining work (ep3.0 scoring and the diversity check) ran to completion without interruption.

### D069 — A3.3 result: ep2.0 chosen (greedy dev ARI 49.46 / 31.19); the sampled votes are graded, so votes stay the main score
- **2026-10-06 · A3.3 · by Claude (pre-set rules) · Finding + Done** (`results/a3/v1/`: `train_log.jsonl`, `greedy_dev.json`, `diversity.json`, the exported greedy answers per adapter and the 20-map samples)
- *Training:* 1,231 test-like training maps, 3 epochs, 231 updates.
  - 110 min in total (36.6 min per epoch, including dev-loss checks); 674 tokens/s overall; peak 4.92 GiB.
  - The loss mask was verified on all 1,231 maps (392,713 answer tokens).

  | Adapter | ep0.5 | ep1.0 | ep1.5 | **ep2.0** | ep2.5 | ep3.0 |
  |---|---|---|---|---|---|---|
  | Dev answer loss | 0.0888 | 0.0804 | 0.0765 | 0.0753 | 0.0757 | 0.0747 |
  | Greedy dev ARI, general / focused | 35.91 / 12.34 | 43.94 / 21.90 | 47.49 / 27.93 | **49.46 / 31.19** | 46.21 / 25.41 | 47.01 / 26.81 |
  | Relations per map (gold 8.3) | 11.5 | 8.36 | 9.76 | 9.60 | 9.15 | 9.37 |

  - **Rule:** the best greedy ARI general → **ep2.0**. As in A1 (D043), the score peaks at 2 epochs while the loss still edges down.
  - All 1,350 greedy answers are valid; 1 of them (at ep0.5) hit the length cap.
  - Greedy scoring takes ≈ 15 min per adapter.
  - *Reference only, not comparable:* this is one greedy answer, with no thresholds. A1's honest score is 50.50 / 32.61, and one untuned A2 answer scored 52.09 / 35.24 (D063).
- *Diversity rule:* ep2.0, the first 20 dev maps, 8 samples each, 160 / 160 valid.
  - **99.8% of the 1,099 distinct listed relations have a vote share strictly between 0 and 1.** By the rule, **vote shares are A3.4's main score.**
  - The votes are spread wide. 720 relations appear in 1 of 8 samples and only 2 in all 8; a sample lists 11.7 relations on average, against 9.6 for greedy.
  - Sampling at temperature 1 is therefore far from all-or-nothing, but individual samples disagree a lot. The thresholds will sit low, and token probabilities stay as a free ablation.

### D070 — A3.4 result: honest dev ARI 47.65 / 27.23, below A1 and A2; both success tests fail
- **2026-10-06 · A3.4 · by Claude (pre-set rules) · Finding** (`results/a3/v1/dev_report.json`, `thresholds.json`, `ep2.0_dev_samples8.jsonl.gz`; `python -m a3_finetune.runner --run=v1 --adapter=ep2.0 --split=dev --samples=8 --export`, `python -m a3_finetune.tune --run=v1`)
- *Run:* 225 dev maps × 8 relation samples with ep2.0; 1,800 / 1,800 answers valid and finished; ≈ 2.5 h.
- *Thresholds:* Inference 0.15 (≥ 2 of 8 samples), Conflict 0.5, Rephrase 0.3. Identical in 3 of 5 folds.

  | Dev ARI, general / focused | |
  |---|---|
  | A1, honest | 50.50 / 32.61 |
  | A2, honest | 54.63 / 34.12 |
  | **A3, votes + thresholds, honest (the pre-registered main score)** | **47.65 / 27.23** |
  | A3, one greedy answer with every relation kept (nothing tuned) | 49.46 / 31.19 |
  | A3, token-probability score, honest | 47.86 / 25.89 |
  | A3, one sample at temperature 1, every relation kept (8 samples) | 37.1–42.0 / 18.4–22.2 |

- *Success test 1 fails:*
  - A3 − A2 = −6.99 general, 95% interval [−8.9, −5.2]; A2 is better on 153 maps, A3 on 69.
  - A3 − A1 = −2.85, interval [−4.7, −1.0].
- *Success test 2 fails:* cross-fitted averages of pair scores.
  - A1 + A2 = 55.12 / 36.18, and A1 + A2 + A3 = 54.24 / 35.45: adding A3 costs 0.88.
  - *Aside for the Combination phase:* A1 + A2 alone already beats A2 alone (+0.5 / +2.1).
- *Hypotheses at matched density* (8.32 relations per map for all three; A3's vote ties broken by token probability for this measure only):
  - **H1 fails.** Neighbour-Inference recall is 0.25, against A1 0.39 and A2 0.42.
  - **H2 holds.** 805 unrelated pairs are given a relation, against A1 905 (rule ≤ 815) and A2 812.
  - **H3 fails.** Inference direction is right 83% of the time (A1 86%, A2 84%; all close).
  - **H4 fails.** Within-speaker Conflict recall is 11% (10 of 91), against A1 9% and A2 7%; the bar was 15%.
- *Per type* (pooled P / R / F1, tuned thresholds):
  - Inference 0.26 / 0.27 / 0.26 (A1 0.32 F1, A2 0.37);
  - Conflict 0.29 / 0.06 / 0.10 (A1 0.14, A2 0.19);
  - Rephrase 0.37 / 0.52 / 0.43 (A1 0.47, A2 0.51).
  - Direction is right for 76% of the Inferences found; 11 maps get an invented Conflict (A1 8, A2 8).
- *Reading:*
  - The fine-tuned 4B learns the format and the annotators' density, and it is as precise as A2 at matched density. But it **finds fewer true relations, Inference above all**. It does not acquire the encoder's neighbour habit, and every relation type is below both A1 and A2.
  - **Sampling hurts it.** A single temperature-1 sample (37–42) is far below the greedy answer (49.5). Averaging 8 samples recovers most of the gap, but not all of it.
  - A3's best configuration on dev is therefore the greedy answer, which is still below A1 and A2.
- *Open for the user (A3.5 / A3.6):*
  1. Which configuration to test. The pre-registered main is votes + thresholds; greedy is +1.8 / +4.0 on dev. Proposed: freeze both before the test run, then report both.
  2. Which A3.5 ablations to run now that A3 trails. Zero-shot (H6), no grammar and turn types (H5) are cheap. The giant-map retrain (H4) costs ≈ 5 GPU hours.
