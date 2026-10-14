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
| A3.5 | Assemble + ablations | Valid maps, dev Global; LLM turn types (strict rule); combination preview; ablations | 225/225 valid; combination test; ablation table | A3 | **Done, reduced (D071–D072)**: 225 / 225 valid, dev Global 68.90 / 53.95; no ablations (user) |
| A3.6 | Test run + report | Final A3 row; `reports/A3-finetuned-llm.md` | Run once, on the user's go | Final run | **Done (D072)**: test Global 73.17 / 58.17, ARI 53.70 / 32.05 (between A1 and A2; dev ranks A3 below both) |
| A4.1 | Setup + feasibility | `a4_jev/`: HTTP client (`/v1/evaluate`), state + pair / source questions, cached runner, parser → A1's pair format, `check.py`, `probe.py` | Round trip 98.64 / 94.44 with no call; live answers on 5 training maps valid; billing, speed, repeatability, option-order bias measured; budget projection within the cap | A4 | **Passed (D076)**: round trip 98.64 / 94.44; 15 / 15 requests valid; state billed once per request; both option orders kept; remaining passes ≈ $0.80 |
| A4.2 | Pilot (one pass) | 50 training-split maps (A2's `maps_50`, gold links), both views and orders; ≤ 1 wording fix, only for a clear failure | View chosen by the pre-set rule (D076); settings frozen | A4 | **Done (D078)**: k = 0, view `both`, wording v1 frozen; pilot both-view 51.83 / 31.13 at matched density; $0.40 |
| A4.3 | Dev run + decision layer | 225 dev maps, frozen settings, labeller links; A1's temperature + thresholds, cross-fitted; calibration | Honest dev ARI next to A1–A3; success tests and H1–H5 decided | A4 | **Done (D079)**: honest dev ARI 52.17 / 32.45 (A1 50.50, A2 54.63); both success tests narrowly fail; H4, H5 pass; $0.62 |
| A4.4 | Assembly + combination test | Valid maps, dev Global (labeller's turn types); A1 + A2 + A4 vs A1 + A2, from saved answers; no ablations (D076) | 225 / 225 valid; success test 2 | A4 | **Done (D079)**: 225 / 225 valid; honest dev Global 71.16 / 56.56; A1 + A2 + A4 = +0.39 over A1 + A2 |
| A4.5 | Test run + report | Final A4 row; `reports/A4-jev.md` | Run once, on the user's go | Final run | **Done (D080)**: test Global 73.42 / 57.63 (ARI 54.21 / 30.98), between A3 and A2; A4 total $1.08 |
| C | Combination | `combination/run.py`: mean of A1 + A2 + A4 pair scores → A1's decision layer, cross-fitted; test once (all answers cached) | Pre-registered main (A1 + A2 + A4) and comparison (A1 + A2) fixed before the test gold is read | Combination | **Done (D084)**: test Global 74.51 / 59.47 (main), the project's best pre-registered system; A1 + A2 75.17 / 60.32 |
| 4 | Four-approach comparison | `reports/04-comparison.md` | Every number from a committed result | Final run and write-up | **Done (D085)** |
| 5 | Final report | `reports/05-final-report.md`; per-approach reports updated with the final results | Two independent reviews, each followed by a revision; every number from a committed result or the 2024 overview paper | Final run and write-up | **Done (D086)** |
| 6 | Short report | `reports/06-short-report.md`, `reports/figures/short-*.png`, `reports/short_report_numbers.py`, `reports/short_report_figures.py` | Every number and figure recomputed from committed files | Final run and write-up | **Done (D087)** |
| 7 | Final README + relation-plateau analysis | `README.md`; §10 of the final report and §6 of the short report; `reports/figures/short-why-relations.png` | Every claim traced to a committed file; every link resolves | Final run and write-up | **Done (D088)** |
| 8 | Write-ups reframed as an exploration | `README.md` and every report; `reports/short_report_numbers.py`, `reports/short_report_figures.py` and the figures they draw | No result or method changed; no combination in any write-up or figure; every number traced to a committed file; every link resolves | Final run and write-up | **Done (D089)** |
| 9 | Claude as ILO labeller (dev test) | `ilo_llm/`, `results/ilo_llm/`, `reports/figures/ilo-llms-vs-labeller.png` (replaced the Claude-only figure, D101); brief in `plans/claude-ilo-dev-test/README.md` | Gold round trip ≥ 98.6 / 94.7; labeller reference 90.08 / 80.64 reproduced; 675/675 valid dev answers; verdict by the D090 stop rule | Final run and write-up | **Done (D090–D098)**: round trip 98.70 / 96.28; reference reproduced; 675 / 675 valid; Claude honest dev ILO 88.03 / 77.48 vs the labeller's 90.08 / 80.64, **not better** |
| 10 | Jev as ILO labeller (dev test) + labeller discussion | `ilo_jev/`, `results/ilo_jev/`, three-way figure, `reports/03-ilo-discussion.md` | Gold round trip = Claude's (98.70 / 96.28); cost projection ≤ $2; 225 / 225 valid dev maps; verdict by the D099 stop rule | Final run and write-up | **Done (D099–D101)**: Jev honest dev ILO 85.35 / 75.76 vs the labeller's 90.08 / 80.64, **not better**; best linking of the three (98.98% / 99.56%); $0.34 |

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

### D071 — A3 finishes without ablations; test configuration frozen before the test run
- **2026-10-06 · A3.5–A3.6 · by user (scope) / Claude (configuration, as proposed in D070) · Approved**
- **Scope (user).** None of the A3.5 ablations run. The not-run list:
  - zero-shot (H6), no grammar, LLM turn types (H5);
  - sample counts, majority vote, the direction rule;
  - the giant-map retrain (H4), and a lower sampling temperature.
  - H5 and H6 stay untested. A3 ends with the test run, the report and a discussion of why the results came out this way.
- **Test configuration, frozen now (before the test run):**
  1. **Votes (main, pre-registered):** 8 relation samples per map → vote shares → the thresholds fitted on all of dev (Inference 0.15, Conflict 0.5, Rephrase 0.3).
  2. **Greedy (reported beside it):** the greedy answer, every relation it lists kept. It is better on dev (+1.8 / +4.0), but was not pre-registered.
  3. **Turn types:** the ILO labeller's links and types for both. The LLM's types are not used: the strict rule was never evaluated, and A2's LLM types cost 1.1 ILO on test (D064).
- **Protocol (as A2.5):** both configurations' 11 maps are assembled, validated and saved before `data/evaluation/` is read, then scored once; nothing is changed afterwards. Dev Global comes from the same assembly with cross-fitted thresholds.

### D072 — A3.6 result: test Global 73.17 / 58.17 (ARI 53.70 / 32.05), between A1 and A2; dev ranks A3 below both
- **2026-10-06 · A3.6 · run on the user's go · Finding** (`results/a3/test_global.json`, `test_predictions/`, `test_predictions_greedy/`, `v1/ep2.0_test_*.jsonl.gz`, `v1/dev_global.json`, `test_posthoc.json`)
- *Protocol (D071):* 11 test inputs, with greedy + 8 samples each (88 / 88 samples valid). Both configurations' 22 maps were assembled, validated and saved before the gold was read, then scored once.

  | Test, general / focused | ARI | ILO | Global |
  |---|---|---|---|
  | A1 + labeller (D050) | 51.38 / 30.77 | 92.64 / 84.28 | 72.01 / 57.53 |
  | A2 + labeller + LLM types (D064) | **55.77 / 33.34** | 91.56 / 83.20 | **73.67 / 58.27** |
  | **A3 votes + labeller (pre-registered)** | **53.70 / 32.05** | 92.64 / 84.28 | **73.17 / 58.17** |
  | A3 greedy + labeller | 52.25 / 31.74 | 92.64 / 84.28 | 72.44 / 58.01 |

- *Against the 2024 field:* Global 1st / 1st (DFKI 67.05 / 48.25). ARI general 2nd (DFKI 55.33); ARI focused 2nd (POKEMON 35.89).
- *Per type (votes):*
  - Inference P / R / F1 0.37 / 0.39 / 0.38, with direction right for 89%;
  - Rephrase 0.50 / 0.57 / 0.53;
  - **Conflict 0 of 13 found** (none predicted);
  - no map gets an invented relation type.
- *Dev, same assembly with cross-fitted thresholds (225 / 225 valid):*
  - votes: ARI 47.65 / 27.23, ILO 90.16 / 80.67, **Global 68.90 / 53.95**;
  - greedy: Global 69.81 / 55.93;
  - for comparison, A1 70.29 / 56.63 and A2 72.53 / 57.63.
- *The dev → test change is +6.05 ARI*, against A1's +0.9 and A2's +1.1. Checked post-hoc, read-only:
  - **11 maps reproduce it often enough.** A3's test mean lies within the 95% range of random 11-map dev subsets (39.9–55.6; a test-or-higher mean in 6.6% of subsets). A3 − A1 ≥ +2.3 occurs in 10% of subsets.
  - **Per map it's a split:** A3 beats A1 on 5 of 11 maps and A2 on 5.
  - **A3's own outputs are unstable:** votes and greedy differ by up to 21 points on one map.
  - *Reading:* the test ranking (A1 < A3 < A2) differs from dev's (A3 < A1 < A2). The dev set is 20× larger, so its ranking stands. A jump this size is uncommon but within 11-map variation; contamination isn't supported, but no memorisation probe was run.

### D073 — A4 plan: Jev answers typed questions about the whole map; pair and source questions feed A1's decision layer
- **2026-10-06 · A4 · by Claude · Approved by user 2026-10-06** ("update everything else", with the gateway key supplied). Open items 2–4 take their recommended defaults (below). The full plan is in [`reports/A4-jev.md`](reports/A4-jev.md). Jev facts come from TypeSafe's and Vercel's docs, read 2026-10-06.
- *Question:* can a decision model with no training and no reasoning step reach encoder level, given whole-map context and a probability for every pair? Does it add a useful, different opinion to the A1 + A2 combination?
- *Design:*
  - **State:** a guide (A2's `system_v1` definitions and measured habits, condensed, plus a concession rule for Conflict) and the serialised map (`common/serialise.py`, D029).
  - **Questions, all in one request per map:**
    - one 5-label `choice` per candidate pair (A1's labels, pruned at 10 turns);
    - one `choice` per proposition over its candidate targets × relation types, plus "none". The targets compete: this tests A2 discussion Q1 without generation;
    - one `choice` per labeller link over the 7 turn types.
  - Every question is asked in two option orders and the probabilities averaged, against Jev's documented first-option bias.
  - **Scores:** answers → A1's 5-label pair format → A1's temperature + per-type thresholds, cross-fitted by episode → `build_map`.
  - **Turn types:** the labeller's, unless Jev's pass A3's strict rule (D065).
  - **Development on training maps only:** A2's 50-map pilot set; dev is used once, for thresholds and the honest score.
- *Route:* Python posts to the Vercel AI Gateway's `/v1/evaluate` with `httpx` (already in the venv). The AI Gateway free tier gives $5 a month, so the expected spend is $0 (the no-spend rule). Credits are never bought, which would also end the free tier.
  - This replaces PLAN's Node runner: the HTTP endpoint didn't exist when PLAN was written.
  - A4.1 verifies that Jev is on the free-tier model list.
- *Success (honest dev), fixed now:* either
  1. A4's ARI general > A1's 50.50, with a paired bootstrap 95% interval above 0; or
  2. A1 + A2 + A4 averaged beats A1 + A2 (55.12) by ≥ 1.0 general.
- *Hypotheses (matched density):*
  - H1: class-wise ECE ≤ A1's;
  - H2: source view ≥ pair view + 1.0 general, with ≤ 90% of its invented relations;
  - H3: within-speaker Conflict recall ≥ 15%;
  - H4: Inference direction ≥ 85%;
  - H5: ≤ 1 s per map.
- *Estimated cost:* ≈ $2 at list price for all of A4 if the state is billed once per request; up to ≈ $20 if it is billed once per question (A4.1 measures this). If the projection exceeds the free credit, a fixed trimming order applies, then the user decides.
- *Changes on approval (done):*
  - PLAN.md's Approach 4 section is rewritten to match: HTTP route, source questions, 4b deferred, the pilot on training maps, the success tests. Its EDA bullet on Jev profiling is marked dropped;
  - CLAUDE.md §7's A4 setup (`experimental_evaluate` is now a deprecated alias of `experimental_decide`; no Node runner; free tier, never buy credits);
  - the phase-map rows A4.1–A4.5 are approved.
- *Settled at approval:*
  1. **Key:** the user supplied an AI Gateway key. It is saved as `AI_GATEWAY_API_KEY` in WSL `~/.profile` (mode 600), not in the repo. No request has been sent yet, so the free tier is checked in A4.1.
  2. **Iterative commitment (4b):** deferred. It runs only if H2 fails.
  3. **D017's Jev corpus profiling:** dropped, because A4.3's per-type breakdown covers it.
  4. **Success test 1's bar:** beat A1.

### D074 — A4.1 blocker: Jev is not available on the AI Gateway free tier
- **2026-10-06 · A4.1 · by Claude · Finding** (two free read-only calls, plus one 1-question request that was refused)
- *Key:* valid. `GET /v1/credits` shows a balance of $4.99994645 of the free $5 ($0.00005355 used before this session). `GET /v1/models` lists `typesafe-ai/jev`:
  - type `evaluation`, providers DigitalOcean (primary) and TypeSafe;
  - $0.000000042 per input token, output $0;
  - "64,000 tokens total per request; 32,000 tokens for the state plus the longest question";
  - `no_training: all`.
- *Refused:* the first `POST /v1/evaluate` (one choice question, ~60 tokens of state) returned **403 `RestrictedModelsError`**: "Free tier users do not have access to this model. Upgrade to paid credits…". Nothing was charged; the balance is unchanged.
- *Other routes checked (web, 2026-10-06):*
  - **TypeSafe direct:** no free tier; signups opened 2026-09-20 with $5 credit, then paused on 2026-09-22.
  - **DigitalOcean Serverless Inference:** $0.042 / 1M input; prepaid only, payable from "promotional credits or a positive prepaid balance". Whether a new account's trial credit applies is unverified.
  - **Third-party playgrounds** (no pipeline access) and n8n Cloud's free Jev offer (until 2026-10-10, workflow-only) don't fit a 1,200-request research run.
- *Consequence:* the plan's rule applies ("Jev not on the free tier: stop and ask; nothing is bought"). A4.1 is paused before any code depends on the route.

### D075 — A4 runs on a small paid AI Gateway credit, with safeguards
- **2026-10-06 · A4 · by user · Approved** (chosen from: buy a small gateway credit / try DigitalOcean credit / skip A4 / a free local substitute)
- **Route:** the user tops up the minimum AI Gateway credit. Auto top-up stays off, and a gateway budget cap (e.g. $10) is set if the dashboard allows. This supersedes D073's "free tier, $0" route; everything else in D073 stands.
- **Spending rule:**
  - A4.1's probes (well under $0.05) measure how Jev bills the state.
  - Claude then reports the projected cost of A4.2–A4.5 before anything larger runs. Work stops and asks if the projection exceeds the cap.
  - Every request's gateway cost is logged.
- *Consequence the user accepted:* buying credit permanently ends the account's free $5-a-month gateway credit.
- *Expected:* ≈ $2 at list price if the state is billed once per request; up to ≈ $20 if once per question.

### D076 — A4 trimmed to its core (user); A4.1 result: gate passed, ≈ $0.80 for everything left
- **2026-10-06 · A4 · scope by user ("as efficient as possible … only the core requirements of the A4 approach, no ablations that are unnecessary and bloating") / result by Claude · Approved + Finding** (`results/a4/check.json`, `probe.json`, `probe/`, `probe-repeat/`; `python -m a4_jev.check`, `python -m a4_jev.runner --run=probe --export`, `python -m a4_jev.probe`)
- **Scope (supersedes D073 where they differ):**
  - **Dropped:**
    - every new-call ablation (no pruning, no guide, names-only options);
    - the A4.2 sweeps (definitions placement, worked examples): definitions stay in the guide, with k = 0;
    - turn-type questions: the ILO labeller keeps Task B, as in A3's final configuration;
    - the separate billing and request-limit probes: billing is read from the probe's own requests;
    - the free ablations beyond those H1–H5 and the view choice need.
  - **A4.2 becomes one pilot pass** on the 50 training maps (both views, both orders).
    - *Rule, fixed now:* take the view with the best matched-density ARI general. If `pair` or `source` alone is within 1.0 of `both`, take that single view, which halves later requests.
    - At most 1 wording fix, and only for a clear failure seen in the errors.
  - **A4.3–A4.5 as planned:** one dev pass, the success tests and H1–H5 from saved answers, assembly and the combination test, then one test run.
  - **Rule fixed before the probe:** later runs ask only one option order if forward and reversed choices agree on ≥ 95% of questions with a mean total-variation distance ≤ 0.05.
- **A4.1 gate, offline:**
  - lossless round trip: pair view **98.64 / 94.44**, equal to A1's ceiling. The source view gives 98.60 / 94.26 (it loses propositions starting 3+ relations), and `both` gives 98.64 / 94.44;
  - 225 / 225 maps valid; no node IDs in any request (D029);
  - a median dev map is 162 pair + 28 source questions (both orders) in 2 requests, with at most 88 options per source question.
- **A4.1 live probe** (5 training maps: A2's 3 smoke maps + the 2 largest outside the pilot episodes; 15 requests + 2 repeats; **$0.035 in total**):
  - **Validity:** every answer has a known choice and probabilities summing to 1; the model is `typesafe-ai/jev`.
  - **Billing: the state is billed once per request.** 691k input tokens were billed, against 5.4M if it were billed per question.
  - **Tokens:** 2.96 characters per token on average, but 2.69–3.27 per request. The first attempt (estimating 4.0) sent one request over Jev's 64k limit, which was refused with `max_tokens_exceeded` and not charged. The estimator is now 3.0, with a 50k request budget (≤ ~58k real tokens at the densest ratio seen). The probe runs keep the 56k budget they used.
  - **Speed:** 0.90 s per request at the median (max 1.35 s), even with 356 questions in one request. Run one after another, a map's requests take 1.4–4.5 s (median 2.7 s); they can run in parallel.
  - **Option order:** forward and reversed choices agree on 90.8% of 1,041 questions (pair 91.8%, source 83.3%), with a mean distance of 0.083. By the rule, **both orders stay.** On average the first-listed option *loses* 0.027 probability.
  - **Repeatability:** the same request sent twice gives the same choice on 98.1% of questions, with probabilities differing by up to 0.17. Jev isn't deterministic, so the cache and the committed answers carry reproducibility.
  - **Sanity score** (the 5 maps, gold links, ARI at matched density; nothing is chosen from 5 maps): pair 47.9 / 27.8, source 53.8 / 32.6, **both 56.2 / 37.1**. The answers carry signal.
- **Projection at the measured ratio:** pilot $0.13 + dev $0.63 + test $0.04 = **≈ $0.80** with both orders. The balance is $14.96, and D075's cap holds.

### D077 — A4 budget $10, run to the end without stopping; one performance lever back in the pilot
- **2026-10-06 · A4 · by user · Approved** ("you can use 10 dollars … do not sacrifice core parts … the best it can be within the budget, but nothing bloating … how Jev could be useful and how it performs … Go ahead and run, do not stop").
- **Budget:** up to $10 of gateway credit for all of A4. The balance before A4.2 was $14.96. Every run's cost is logged, and the projection is re-checked before the dev and test passes.
- **Authorisation:** A4.2 → A4.5 run in sequence without pausing for approval, **including the one-time test run** (A4.5), under the usual protocol: the test configuration is frozen from dev before `data/evaluation/` is read, and the maps are assembled, validated and saved before scoring.
- **Pilot (A4.2), revised from D076:** two arms on the 50 training maps instead of one pass:
  1. the base design (k = 0);
  2. **one TF-IDF-retrieved worked example in the state (k = 1)**, from the pool without the pilot episodes, as A2.
  - *Why:* worked examples were A2's largest lever (+4.5 general from k = 0 to 4, D058). Jev's documented weak spot (unrelated state lowers accuracy) makes the sign unknown. This is a performance lever, not an ablation.
  - *Rule (fixed now):* keep k = 1 only if it gains ≥ 1.0 ARI general at matched density on the pilot (best view of each arm), else k = 0. Then the view rule of D076 picks pair / source / both on the chosen arm.
- **Still out (D076):** every other ablation, the definitions sweep and turn-type questions.
- **Where it runs:** A4's analysis reuses A3's comparison code (`a3_finetune/tune.py`: A1 / A2 alignment, cross-fitting, bootstrap, matched density), which imports A3's libraries. So A4 runs in `~/.venvs/a3` from here on; it also has `httpx`.

### D078 — A4.2 result: no worked example, both views, original wording; frozen for dev and test
- **2026-10-06 · A4.2 · by Claude (pre-set rules) · Done** (`results/a4/pilot_table.json`, `pilot-k0/`, `pilot-k1/`, `pilot-k0-w2/`; `python -m a4_jev.pilot`)
- *Runs:* A2's 50 pilot maps (training split, gold links). 3 arms × ~88 requests, 50 / 50 maps valid in each, **$0.40** in total.
- *ARI at matched density (one threshold, predicting as many relations as gold: 396), general / focused:*

  | Arm | Pair | Source | Both |
  |---|---|---|---|
  | k = 0 (base) | 47.22 / 25.28 | 48.68 / 27.57 | **51.83 / 31.13** |
  | k = 1 worked example | 46.83 / 24.52 | 50.06 / 28.01 | 51.95 / 31.32 |
  | k = 0, source wording v2 | 47.56 / 25.08 | 48.36 / 26.50 | 51.21 / 30.68 |

- *Rules (D076, D077, and the wording rule set before its run):*
  - **k = 1 gains +0.12** (< 1.0), so **k = 0**: the worked example doesn't help Jev, unlike A2 (+4.5).
  - **View `both`:** pair and source alone are 4.6 and 3.2 below it, so neither single view qualifies. The two views carry different information.
  - **Wording v2 not adopted (−0.62):** A4.2's one wording fix.
    - *Why it was tried:* source questions reverse Inference direction far more often than pair questions (76% right against 91%) and turn 69 Rephrases into Inferences. The suspected cause was the word "bear on", since a conclusion bears on its premise too. v2 asked what P "supports, restates or attacks".
    - *Result:* direction stayed at 76.5%. The reversal comes from the question form, not the wording. When both directions are explicit options (pair questions), Jev reads direction well; asked only to pick a target, it often picks its own premise.
- *Errors, base arm, both views:* 134 invented Inferences and 101 found; 87 missed. 40 Rephrases read as Inference; Conflict 7 found, 18 missed, 10 invented.
- *Reference only (different measure):* one A2 answer on the same 50 maps scored 45.0–50.7 general with every listed relation kept (D058).
- **Frozen for A4.3–A4.5:** guide v1, short options, k = 0, source wording v1, pair + source questions, both option orders, scores = mean of the two views, A1's decision layer.

### D079 — A4.3–A4.4 result: honest dev ARI 52.17 / 32.45, between A2 and A1, with no training; both success tests narrowly fail
- **2026-10-06 · A4.3–A4.4 · by Claude (pre-set rules) · Finding** (`results/a4/dev/`: `dev_report.json`, `thresholds.json`, `dev_global.json`, `responses.jsonl.gz`; `python -m a4_jev.runner --run=dev --export`, `python -m a4_jev.tune`)
- *Run:* the frozen configuration (D078) on 225 dev maps with the labeller's links. 402 requests, all valid; **$0.62**, 14.7M input tokens; 0.75 s per request at the median. All 24,169 rows align with A1's and A2's.
- *Thresholds (kept for test):* Inference 0.55, Conflict 0.75, Rephrase 0.30. Inference is the same in all 5 folds; Conflict 0.75–0.80.

  | Dev ARI, general / focused | |
  |---|---|
  | A1, honest | 50.50 / 32.61 |
  | A2, honest | **54.63 / 34.12** |
  | A3, honest | 47.65 / 27.23 |
  | **A4, both views, cross-fitted (honest)** | **52.17 / 32.45** |
  | A4, pair view alone (honest) | 50.50 / 32.57 |
  | A4, source view alone (honest) | 46.32 / 26.73 |

- *Success test 1 fails, narrowly:* A4 − A1 = **+1.68 general, 95% interval [−0.20, +3.65]**; A4 is better on 123 maps, A1 on 100. Against A2: −2.46 [−4.19, −0.73].
- *Success test 2 fails:* cross-fitted averages. A1 + A2 = 55.12 / 36.18, and **A1 + A2 + A4 = 55.51 / 37.01, a gain of +0.39** general (+0.83 focused), short of 1.0. A2 + A4 = 54.16 / 33.65 (no trained model); A1 + A4 = 53.48 / 34.85.
- *Hypotheses:*
  - **H1 fails.** Class-wise ECE is **0.030 for A4 against 0.018 for A1** (before temperature; measured per Inference direction, corrected by D081). Jev's Conflict (0.013) and Rephrase (0.007; A1 0.028) probabilities are well calibrated, but **Inference is over-confident (0.070**; at a stated ~0.9, 35% are right).
  - **H2 fails.** At matched density the source view scores 46.07 against the pair view's 47.22. Competition does what it was meant to, cutting invented relations to 67% (778 against 1,155), but source questions lose direction (76% against 91%). The two views complement each other: `both` scores 51.13.
  - **H3 fails.** Within-speaker Conflict recall is 12.1%, under the 15% bar, but the best of any system (A1 8.8%, A2 6.6%, A3 11%).
  - **H4 passes.** Inference direction is 86.6% right (A1 85.8%, A2 84.3%), with no reasoning step.
  - **H5 passes.** 0.83 s per map with a map's requests in parallel (1.28 s one after another); $0.62 per dev pass.
- *At matched density* (8.3 relations per map; A1 / A2 / A4):
  - neighbour-Inference recall 0.39 / 0.42 / **0.51**;
  - unrelated pairs given a relation 905 / **812** / 859;
  - ARI 50.02 / **54.32** / 51.13.
- *Per type* (tuned, pooled), P / R / F1:
  - Inference 0.34 / 0.42 / **0.38** (A1 0.32, A2 0.37 F1);
  - Conflict 0.52 / 0.12 / 0.19 (A2 0.19);
  - Rephrase 0.44 / 0.38 / 0.41 (A1 0.47, A2 0.51).
  - 8.6 relations per map (gold 8.3); 8 maps with an invented Conflict (A1 8, A2 8).
- *Complete dev maps (A4.4):* 225 / 225 valid, the labeller's turn types. **Honest dev Global 71.16 / 56.56** (A1 70.29 / 56.63, A2 72.53 / 57.63, A3 68.90 / 53.95).
- *Reading:*
  - With no training and no reasoning, Jev lands between the trained encoder and Claude. It has the best Inference F1 and the highest neighbour-Inference recall of the four systems: it finds the annotators' loose support between neighbouring turns better than even A1.
  - Its weaknesses are Rephrase (it calls Rephrases Inference) and over-confident Inference probabilities.
  - Asking the same relation two ways, independently per pair and competitively per proposition, is what makes it work: each view alone is at A1's level or below.
- **Test configuration, frozen now (before the test run):** run `test` (as `dev`), view `both`, the thresholds above, the labeller's links and types.

### D080 — A4.5 result: test Global 73.42 / 57.63 (ARI 54.21 / 30.98), 2nd of the four systems; all of A4 cost $1.08
- **2026-10-06 · A4.5 · run under D077's authorisation · Finding** (`results/a4/test_global.json`, `test_predictions/`, `test/responses.jsonl.gz`; `python -m a4_jev.runner --run=test --export`, `python -m a4_jev.predict`)
- *Protocol (D079):* 11 test inputs, the frozen configuration (view `both`; thresholds Inference 0.55, Conflict 0.75, Rephrase 0.30; the labeller's links and types). 22 requests, all valid, $0.038. The 11 maps were assembled, validated and saved before the gold was read, then scored once, with nothing changed afterwards.

  | Test, general / focused | ARI | ILO | Global |
  |---|---|---|---|
  | Best 2024 entries | 55.33 / 35.89 | 81.17 / 72.09 | 67.05 / 48.25 |
  | A1 + labeller (D050) | 51.38 / 30.77 | 92.64 / 84.28 | 72.01 / 57.53 |
  | A2 + labeller + LLM types (D064) | **55.77 / 33.34** | 91.56 / 83.20 | **73.67 / 58.27** |
  | A3 votes + labeller (D072) | 53.70 / 32.05 | 92.64 / 84.28 | 73.17 / 58.17 |
  | **A4 (Jev) + labeller** | **54.21 / 30.98** | 92.64 / 84.28 | **73.42 / 57.63** |

- *Against the 2024 field:* Global 1st / 1st; ARI general 2nd (DFKI 55.33); ARI focused 3rd (POKEMON 35.89, DFKI 30.40 below).
- *Per type (pooled), P / R / F1:*
  - Inference 0.38 / 0.46 / **0.42** (A2 0.44, A1 0.29), with direction right for 86% of the 35 found;
  - Rephrase 0.56 / 0.43 / 0.49 (A2 0.50, A1 0.53);
  - **Conflict 1 of 13** (3 predicted; A1 and A2 2 of 13, A3 0).
  - No map gets an invented relation type.
- *Dev → test:* +2.04 ARI general (52.17 → 54.21), against A1 +0.9, A2 +1.1, A3 +6.05. The order A1 < A4 < A2 is the same as on dev.
- *A4 totals:* 702 answered Jev requests (plus 1 refused for size, not charged) over 392 map passes (probe 5, pilot 3 × 50, dev 225, test 11, repeat 1). **$1.08** of the $10 budget (the gateway's `total_used`; balance $13.92); no training.

### D081 — A4 closes: iterative commitment not run; H1's measure corrected; report and discussion written
- **2026-10-06 · A4 · by Claude (under D076–D077's scope) · Done** (`reports/A4-jev.md`, `reports/A4-discussion.md`, `results/a4/discussion.json`; `python -m a4_jev.discuss`)
- **Iterative commitment (PLAN's 4b) is not run.**
  - D073 said it would run "only if H2 fails". H2 failed on its first half (source view −1.2 against the pair view), because source questions lose direction (76% against 91%).
  - Its second half, the over-linking that 4b was meant to fix, held: invented relations fell to 67%.
  - 4b doesn't address direction, and the user's scope (D076–D077: the core only, nothing bloating) leaves extensions out. *The user can overrule this;* a 4b pilot would cost about $0.15.
- **H1's calibration measure is corrected; the verdict is unchanged.**
  - The first version summed the two Inference directions. In A4's source view they come from two different questions, so the sum can exceed 1 (the top reliability bin averaged 1.05).
  - Calibration is now measured per Inference direction (each a probability from one question), then averaged, for A1 and A4 alike: **A4 0.030 against A1 0.018**. H1 still fails; Inference is over-confident (0.070).
  - Re-running `a4_jev.tune` (no calls, deterministic) left the thresholds and every score identical. The test predictions, made before, are untouched.
- **Discussion** (dev, matched density): A4 is the best of A1 / A2 / A4 at Inference (471 found with direction, against 396 and 301; neighbour recall 0.51) and at Conflict (29 found, against 23 and 19). It is the worst at Rephrase (201 found, against 377 and 358; 169 called Inference). It falls behind on maps of 26+ propositions. The two question forms are complementary.
- **A4 totals:** $1.08 of the $10 budget, no training, ~2 h of work from plan approval to test.

### D082 — Is A4's result solid? A second dev pass and a no-labels decode
- **2026-10-06 · A4 · by Claude, under the user's request to check critically whether Jev does well ("think critically … nothing bloating") · Finding** (`results/a4/robust.json`, `dev-repeat/`; `python -m a4_jev.runner --run=dev-repeat --export`, `python -m a4_jev.robust`; $0.62)
- *Why:*
  - Jev isn't deterministic, and A4.3 was one pass. Its +1.68 over A1 sat inside the bootstrap interval, so run-to-run noise had to be measured before claiming encoder-level performance.
  - "No training" also had to be checked against the labelled dev set the thresholds use.
- *Run to run* (the identical 402 requests sent again; same cross-fitted procedure):

  | Dev ARI, honest | General / focused | A4 − A1 general, 95% interval |
  |---|---|---|
  | Pass 1 (the A4.3 result) | 52.17 / 32.45 | +1.68 [−0.20, +3.65] |
  | Pass 2 | 51.75 / 32.24 | +1.25 [−0.66, +3.23] |
  | Mean of the two passes | 51.58 / 31.93 | +1.08 [−0.80, +3.04] |

  - Both passes choose the same thresholds. At those thresholds, 88.5% of decided relations are identical; per-map ARI differs by 2.0 on average (at most 28.8).
  - **The dev score is stable to about ±0.4.** A4's edge over A1 (+1.1 to +1.7) is always positive but never significant.
  - Averaging the passes doesn't help, so the second pass stays a check, not part of the system.
- *No labels* (no dev-tuned thresholds; dev ARI general / focused):

  | System | ARI |
  |---|---|
  | A1, argmax | 50.08 / 33.82 |
  | A2, every listed relation | 52.07 / 39.41 |
  | A3, greedy (D069) | 49.46 / 31.19 |
  | **A4, argmax** | **50.84 / 36.42** |

  **With no labelled data at all, Jev matches the trained encoder.** Tuned thresholds add +1.3 general and cost −4.0 focused, as they did for A1 and A2.

### D083 — Combination: the mean of A1 + A2 + A4 pair scores; pre-registered before its test run
- **2026-10-06 · Combination · by Claude, under the user's request for "an overall good project on 4 approaches" · Approved within that request** (`combination/run.py`; `results/combination/dev_report.json`, `thresholds.json`)
- *Form:* the simplest version of PLAN's Combination. Every pair's five scores are averaged over A1 (with its temperature), A2 (confidence-weighted votes of 3 answers) and A4 (two views). Then come A1's per-type thresholds, cross-fitted, and the labeller's turn types.
  - *Not run:* the confidence-routing variant (A2 only on A1's / A4's uncertain pairs), and any weighting. Averaging was the measured form, and routing would add a tuned band.
  - *Not included:* A3, because adding it lowered the A1 + A2 average on dev (D070).
- *Dev (honest):*
  - **A1 + A2 + A4: ARI 55.51 / 37.01, Global 72.83 / 58.84** (thresholds Inference 0.40, Conflict 0.55, Rephrase 0.30);
  - A1 + A2: 55.12 / 36.18, Global 72.64 / 58.43.
  - Both give 225 / 225 valid maps.
- **Pre-registered for the one test run (now, before the gold is read):** main = A1 + A2 + A4, the best on dev; A1 + A2 is reported beside it, to show what A4 adds on test.
  - Inputs: A1's saved test probabilities, A2's cached test-v1 answers, A4's test answers. No new calls; $0.

### D084 — Combination test: A1 + A2 + A4 = Global 74.51 / 59.47, above every single system; A1 + A2 is higher still on test
- **2026-10-06 · Combination · run once under D083's pre-registration · Finding** (`results/combination/test_global.json`, `test_predictions/`; `python -m combination.run --test`; $0, all inputs cached)
- *Protocol:* 22 maps (11 per combination) assembled, validated and saved before the gold was read; all 1,208 of A1's test rows aligned.

  | Test F1, general / focused | ARI | ILO | Global |
  |---|---|---|---|
  | Best 2024 entries | 55.33 / 35.89 | 81.17 / 72.09 | 67.05 / 48.25 |
  | A2 alone (best single system, D064) | 55.77 / 33.34 | 91.56 / 83.20 | 73.67 / 58.27 |
  | **A1 + A2 + A4 (pre-registered main)** | **56.39 / 34.65** | 92.64 / 84.28 | **74.51 / 59.47** |
  | A1 + A2 (reported beside it) | 57.69 / 36.37 | 92.64 / 84.28 | 75.17 / 60.32 |

- *Reading:*
  - **Combining helps:** the main combination beats every single system on test (A2 +0.6 ARI, +0.8 Global), as on dev (+0.9 over A2).
  - **A4's share doesn't hold up on test:** −1.30 / −1.72 against A1 + A2, after +0.39 / +0.83 on dev. Across 225 dev and 11 test maps, Jev adds about nothing to a combination that already has A1 and A2. The gain comes from A1's annotator habits plus A2's definitions (A2 discussion).
  - A1 + A2 was not the pre-registered choice; it is reported, not substituted.
  - *Against 2024:* ARI general 56.39 is above the best 2024 entry (55.33), and A1 + A2's 36.37 focused is above it too (35.89).

### D085 — The four-approach comparison is written; the project's planned phases are complete
- **2026-10-06 · Final write-up · by Claude, at the user's request ("a good report on how Jev compares and an overall good project on 4 approaches") · Done** (`reports/04-comparison.md`)
- *Content:*
  - test and honest dev results for A1–A4, the combination and the 2024 field;
  - paired bootstraps (dev ranks A2 > A4 ≈ A1 > A3);
  - cost, speed and data needs;
  - a four-way matched-density table of what each system is good at;
  - a critical answer to "is Jev impressive";
  - the 2024 failures against the lever that fixed each;
  - limits.
- *Added for it:* the A2 − A1 paired bootstrap on dev, **+4.14 [2.21, 6.12]**, in `results/a4/robust.json`. No other new computation.
- *Status of PLAN's success criteria:*
  1. at least one system beats 67.05 / 48.25: all four and the combination do;
  2. each approach has F1, time and cost recorded: yes, with ablations cut to the core for A3 and A4 by user decision (D071, D076);
  3. which 2024 failure each lever fixed: report §5. Conflict remains unsolved by every method.

### D086 — The final report merges every report into one end-to-end account; the per-approach reports are brought up to date
- **2026-10-06 · Final write-up · by Claude, at the user's request (one comprehensive but concise report, reviewed twice by an independent agent and revised after each review; approach reports edited "now that all testing is run") · Done** (`reports/05-final-report.md`)
- *Content:* the aim and success criteria; the task, IAT and the scorer; the data's shape; the 2024 field and why it scored as it did; the shared pipeline and the four approaches; how to read dev against test; results; each model's strengths and weaknesses; a critique of what the benchmark measures and misses; limits; conclusions.
- *New figures:*
  - *derived* (arithmetic on committed or published numbers): the split of the combination's Global margin over DFKI-MLST into +6.93 from ILO and +0.53 from ARI, since Global is the mean of the two (DFKI's ILO 78.78 and ARI 55.33 are from the overview paper);
  - *derived:* about 95% of the scorer's ordered proposition pairs are None (per-map mean 95.0% on dev, 96.1% pooled; EDA's 92% counted unordered pairs). The report states it as arithmetic on committed numbers: ~200 ordered pairs and ~8 relations per map;
  - *quoted:* QT30's inter-annotator agreement, 0.56 CASS (overview paper §3.1).
- *Updated reports:* "what came after" notes in A1–A4 and the discussions; A4's total cost is $1.70 with the D082 re-run; the ILO report records its test score (92.64 / 84.28) and D029's approval; the index starts with the final report.
- *Reviews:* two independent reviewer agents, each followed by a revision.
  - *Round 1* corrected the rare-label trap (a missed gold Conflict also scores 0, so the asymmetry is that a guess pays only on the exact pair), "decisions matter as much as models" (the labeller's model adds +13.5 ILO, thresholds +0.6), "direction fixed" (divergent structure is still wrong), the unsourced 2024-LLM explanation (now our hypothesis), the test-noise rule (about ±8 ARI) and the claim that no Conflict was invented on test (5 of 10 predictions were wrong).
  - *Round 2* corrected the main lesson against the matched-density data, the None share (92% → 95%), "missing context didn't hold 2024 back" (a no-context ablation wasn't run) and "Jev needs no labels" (no labelled pairs or tuned thresholds). It added the focused-score insight (one missed gold relation adds None as a zero-scoring class), and cut the report from ~6,500 to ~5,500 words.
  - Pre-existing errors it found in other reports were fixed: A1's pruning cut-off is 10 turns, not 8 (A2 discussion); A2's test precision gain is on Inference only (A2 report); A3 is close to A2 on new-word restatements (comparison report).
- *No new runs, no scoring, no spend.*

### D087 — A short report: the project in about half the length of the final report, with a critical reading
- **2026-10-06 · Final write-up · by Claude Science, at the user's request ("shorter … comprehensive … think critically throughout"; "commit it all") · Done** (`reports/06-short-report.md`)
- *Content:* the task, the four approaches, the result against 2024, dev against test, Task A against Task B, what the output maps look like, cost, the scoring and the task, what it means for argument mining, the models and the output, what went right and wrong, and what to try next.
- *New figures* (`reports/figures/short-*.png`), drawn from committed numbers: the Global margin over DFKI-MLST split into its illocution and relation parts for every system; dev paired-bootstrap differences beside per-map test ARI with 95% intervals; recall at matched density by relation kind, and what happens to the 1,060 dev gold Inferences; dev ARI against time per map, and the 2024 field and ours on the general/focused plane.
- *New derived numbers* (`python reports/short_report_numbers.py` → `results/short_report/derived.json`; arithmetic on committed files only, `data/evaluation/` not read):
  - each system's 11-map test ARI with a 95% bootstrap interval (20,000 resamples, seed 0): every interval contains DFKI's 55.33;
  - the combination's pooled test relations: 56 of 124 gold right, 108 predicted (precision 0.52, recall 0.45);
  - the share of the dev gap between predicting nothing (30.93) and the ceiling (98.64) that is closed: A1 0.29, combination 0.36;
  - ILO's share of the Global margin: 0.93 general, 0.81 focused;
  - per-map test ILO: 100 on 7 maps, 98.6 on map 6, about 73 on maps 1, 2 and 9 (bimodal; not diagnosed, since that would mean reading the test gold).
- *Not new:* every other number is quoted from the final report, the comparison and per-approach reports, or the committed result files they cite.
- *Reproduce:* `python reports/short_report_numbers.py`, then `python reports/short_report_figures.py` (numpy and matplotlib only; matplotlib is not in the pinned ML venv, so any Python with both works). The figure script reads only `results/` plus a few quoted constants, each with its source in the file: the 2024 field (overview paper), the dev floor and ceiling (D044) and the time per map (final report §7.3).
- *No new runs, no scoring, no spend.*

### D088 — Why relation mining barely moved since 2024, and the final README
- **2026-10-06 · Final write-up · by Claude Science, at the user's request** ("what has led to not a high increase in relation linking from 2024 … was it the testing checks? the data? training, dev, test? … add this to the reports … create a readme that will be final … commit all changes to main") **· Done**
- *Analysis* (§10 of `reports/05-final-report.md`, §6 of `reports/06-short-report.md`): each candidate cause of the relation plateau, with a verdict from committed evidence.
  - **Our safeguards: not the cause.** Cross-fitting costs A1 0.40 ARI against thresholds tuned on all of dev (`results/a1/v1/thresholds.json`). D029 concerns turn–proposition links. Freezing each test configuration removes test-set tuning, which can only inflate a score.
  - **The pipeline: not the cause.** Candidate ceiling 98.64 / 94.44 (D041); turn linking 98.3%.
  - **The split: not the cause.** Every system scores higher on test than on dev (A1 +0.88, A2 +1.14, A3 +6.05, A4 +2.04, combination +0.88).
  - **The test set's size: hides a real gain.** Every 11-map 95% interval contains DFKI's 55.33 (D087).
  - **Training data and model scale: contributes, not fully tested** (1,232 Conflicts; giant maps unused; 150M encoder and 4-bit 4B LLM; context +1.5 for A1, reasoning +0.8 for A2).
  - **The task and its labels: main cause.** At matched density on dev, **82–86% of each system's errors are missed or invented relations**, not a wrong type or direction (A1 86%, A2 82%, A4 84% of errors). Recall between adjacent turns is 41–46% for every system; two to ten turns apart the encoder falls to 16% / 12% while Claude keeps 37% / 30%. Type errors follow the conventions, within-speaker attacks stay at 7–12% recall, and annotators agree at 0.56 CASS. The metric dilutes gains: +17 points of pooled test Inference F1 (A1 → combination) moved ARI by 5.0.
- *New derived numbers* (`reports/short_report_numbers.py` → `results/short_report/derived.json`, from `results/a4/discussion.json`, `results/a1/v1/thresholds.json`, `results/a4/robust.json` and the test files): `relation_errors_dev_matched_density`, `relation_recall_by_distance_dev`, `protocol_cost_a1_ari_general`, `metric_dilution_test`, `dev_to_test_ari_general`. Earlier keys unchanged. New figure `reports/figures/short-why-relations.png`, drawn by `reports/short_report_figures.py`.
- *README* (`README.md`): the project's front page, written for a reader with a few minutes: question, results table, pipeline, safeguards, six findings, repository map, reproduction outline, how it was built, limits and references. Every number is copied from the reports or `results/`; the data sources (dialam.arg.tech, corpora.aifdb.org/qt30) and the QT30 citation come from the 2024 overview paper.
- *Structure:* kept as is. The layout already follows the work (shared code in `common/`, one folder per approach, every number under `results/`); moving folders would break the module commands and links that every report and log entry uses. `CLAUDE.md` stays at the root, where Claude Code reads it, and the README explains it.
- *Revised the same day (user request: short context on each model, plainer wording, a clearer account of why relations are hard):* a four-bullet "The four approaches" section (what each model is, how it was trained or prompted, what "typed questions" and "no training" mean for Jev, and A2's `xhigh` reasoning expectation against its +0.8 dev result); "pre-registered" replaced by plain wording; a "Why relations are hard, even for 2026 models" section (task, labels, data, models, score), built from numbers already in §10 of the final report. No new numbers.
- *Revised again (user request: "remove all mentions of the combination" from the README):* the README presents the four single systems only. Headline numbers are now ranges across them (test Global 72.01–73.67, ILO 91.56–92.64, ARI 51.38–55.77) and Claude's results (73.67 / 58.27; 53 of 124 test relations, 2 of 13 attacks; +15 Inference F1 over A1 moving ARI by 4.4), from the new `readme_single_systems` key in `results/short_report/derived.json`. The top figure is `reports/figures/readme-gain-vs-2024.png`, the same plot without the combination row. The reports, `combination/` and its results are unchanged.
- *Revised again (user request):* the README's "Why relations are hard" section opens with the two questions a system answers for every pair (related at all? and how?), a four-row error table for the encoder on dev (905 missed, 905 invented, 229 wrong type, 61 wrong direction; `relation_errors_dev_matched_density`), and closes with why it matters (Claude on test misses 71 of 124 gold links and 59 of its 112 predictions are wrong; `readme_single_systems.a2_test_pooled`). No new numbers.
- *No new runs, no scoring, no spend; `data/evaluation/` not read.*

### D089 — The write-ups are reframed as an exploration, and the combination is no longer presented
- **2026-10-10 · Final write-up · by Claude Science, at the user's request** (feedback that the study "seems directionless" with no clear research question; "pose it as an exploration with a direction … remove the A1+A2+A4 (pre-registered) bar from all images, don't use A1+A2+A4 as a concept") **· Done**
- *No method or result changed, and nothing was rerun.* Only the presentation changed.
- *The framing.* The README, the final report (§1, §12), the short report (summary, §2, §8, §10) and the comparison (§5) now open from where the 2024 systems fell short, as reported in the overview paper (relations as the bottleneck, missed Conflicts, misread direction, LLM entries losing to encoders), plus our own hypotheses about the causes. Each approach is presented as the probe of one shortfall: the ILO labeller for pair-by-pair illocutions, A1 for sentences judged in isolation, A2 for no consistency across a map, A3 for LLMs losing to encoders, A4 for LLM output too unreliable to threshold. The guiding question is *what do models that appeared since 2024 change, and where do they still fail?* The ending: illocutions were fixed by engineering, relations by no model, and §10's error analysis says where effort should go next.
- *The combination.* The A1 + A2 + A4 average (D083–D085) is removed from the README, every report and every figure. Headline numbers are now single systems: best Claude (A2), test Global 73.67 / 58.27; the gain over DFKI splits into +6.4 to +6.9 from ILO and −2.0 to +0.2 from ARI (general) per system; the pooled test example is A2 (53 of 124 relations right, 112 predicted); the metric-dilution example is A1 → A2 (Inference F1 0.29 → 0.437, ARI 51.38 → 55.77). The share of the dev gap closed is 25–35% (single systems). A3's and A4's fixed-in-advance success test 2 ("adding it to the A1 + A2 average gains ≥ 1.0") stays, reported by its dev result only.
- *Scripts.* `reports/short_report_numbers.py` drops the keys that included the combination or were superseded by per-system versions (`readme_single_systems`, `combo_test_pooled`, `headroom_share_combo`, `headroom_share_A1`, `ilo_share_of_global_gain_general`, `ilo_share_of_global_gain_focused`, `metric_dilution_test`) and adds `test_ranges`, `gain_split_vs_dfki`, `test_pooled_relations`, `metric_dilution_A1_to_A2_test`, `headroom_share_dev`; every carried-over value is unchanged. `reports/short_report_figures.py` draws single systems only; `reports/figures/readme-gain-vs-2024.png` is deleted, and the README uses `short-gain-vs-2024.png`.
- *Kept as a record:* `combination/`, `results/combination/`, decisions D083–D085, and A4's success-test code and results (`a4_jev/tune.py`, `results/a4/dev/dev_report.json`).

### D090 — Dev test: can Claude Sonnet 5.5 do Task B (linking and typing) better than the trained labeller?
- **2026-10-14 · Phase 9 · by Claude Science, at the user's request** ("run the cheaper first step"; "write the plan thoroughly in a folder, then I can tell the claude environment in my VS Code to run the plan") **· Approved by user 2026-10-14 · Done 2026-10-14 (D098): not better, 88.03 / 77.48 vs 90.08 / 80.64**
- *Question:* given dev maps in test format (no turn–proposition links), can Claude Sonnet 5.5 link each dialogue turn to its proposition and type it better than the ILO labeller, whose honest dev ILO is 90.08 / 80.64 (`results/ilo/v1/dev_report.json`)?
- *Brief:* [`plans/claude-ilo-dev-test/README.md`](plans/claude-ilo-dev-test/README.md) gives every setting, file, command and gate.
- *Protocol (fixed before any call):*
  - 225 dev maps, serialised with no links (similarity placement only);
  - Claude Sonnet 5.5 through `a2_llm/claude_cli.py`, 4 retrieved training maps as worked examples rendered in the same format, 3 answers voted, the labeller's 10 types;
  - effort chosen on the 20-map A2 pilot (tie, under 5 points, goes to `medium`), with at most one prompt revision, on the pilot only;
  - decision layer as `ilo/tune.py` (vote-share link threshold, one-to-one matching, per-type-group thresholds), with an honest score by leave-one-dev-episode-out cross-fitting over the same 5 episodes.
- *Stop rule:* Claude counts as better only if its honest dev ILO exceeds the labeller's by more than 0.5 on both general (> 90.58) and focused (> 81.14). A paired bootstrap over the 225 maps is reported either way.
- *Out of scope:* the 11 test maps (`data/evaluation/` is not read), and any change to `a2_llm/`, `ilo/` or committed results. Using Claude on test, if it wins, would be a separate decision.
- *Effort (step 4, D095):* **`medium`**. `xhigh` scored lower on the pilot (−2.97 general, −1.06 focused), so the D094 rule picks `medium`.
- *Why now:* the README says Task B was handled by one trained component, and the obvious reviewer question is whether a frontier LLM could do it better. This is the cheapest test, because Claude is the likeliest of the four models to beat the labeller. If it can't, the others are unlikely to.

### D091 — Definitions of the three added types: spec facts only
- **2026-10-14 · Phase 9, step 2 · by user** (chose "Spec facts only" when asked) **· Approved**
- *The gap:* the brief asks for definitions of Disagreeing, Arguing and Restating taken from `reference/` and not invented. None of the three reference files defines them. `annotation-details.pdf` only says where each attaches (Arguing, Restating: transition → relation; Agreeing, Challenging, Disagreeing: transition → proposition), and the overview paper and the data-format spec only name them.
- *Choice:* the system prompt says only what the spec and the training data say. Disagreeing is grouped with Agreeing and Challenging as a reply to an earlier claim, the opposite of Agreeing, so the proposition is the claim disagreed with. Arguing and Restating normally anchor the step that creates an inference or a rephrase, not a turn, so they are very rare as a turn's type. Their training counts on turn links are Disagreeing 18, Arguing 10 and Restating 4 in 1,231 maps (`results/ilo/dataset.json`).
- *Alternatives:* names only, with no description; standard IAT meanings written from general knowledge. The second was rejected because it breaks the brief's "don't invent" rule.
- *Numbering:* the brief reserved D091 for a possible prompt revision. A revision, if any, takes the next free number.

### D092 — Phase 9 step 2: `ilo_llm/` built; all four no-call checks pass
- **2026-10-14 · Phase 9, step 2 · by Claude · Done**
- *Result* (`results/ilo_llm/check.json`):
  1. gold round trip **98.70 / 96.28** on 225 dev maps (gate ≥ 98.6 / 94.7), 0 answers failing the schema, 0 invalid maps. It sits above the 03 report's 98.63 / 94.73 because that ceiling also includes the labeller's candidate misses; here only turns with two propositions are lost;
  2. prompts build for all 225 dev and 20 pilot maps;
  3. prompt sizes 13,990 / 19,780 / 27,303 characters (dev min / median / max), with the pilot max at 28,202, under the 35,149 limit (1.2 × A2's 29,291);
  4. the labeller's honest dev ILO is reproduced at **90.08 / 80.64** (temperature 1.05). Its per-map scores are in `results/ilo_llm/labeller_per_map.json`.
- *Differences from the brief, none of which changes the protocol:*
  - **Prompt wording.** The *Input* section no longer says propositions sit "under the turn that expresses them": with no links, a proposition is listed under the turn whose wording it closely matches. The conventions also gain the measured question cues from EDA §3 (almost no "?"; a question starts with a question word, and its proposition often contains "xxx").
  - **Runner.** `summary()` counts illocutions instead of relations. The pilot lists are read from `results/a2/pilot_maps.json`. `resolve()` no longer puts a list into a set, which crashed on the smoke run's map list; A2's copy has the same latent bug, left untouched. The test split is refused, and the `dev` run refuses to start until its effort and prompt are set in step 4.
  - **Examples.** Worked examples are serialised from `strip_to_input(gold)`, the same input the target gets.

### D093 — Phase 9 step 3: smoke test passed
- **2026-10-14 · Phase 9, step 3 · by Claude · Done**
- `results/ilo_llm/smoke/summary.json`: 3/3 answers valid, every token from `claude-sonnet-5-5`, no side models, 3 calls in total (no retries). Median 4.6 s per call, about 9.5k input tokens, 0 thinking tokens at `medium`, $0.13 API-equivalent in all (paid by Pro).
- Every answer lists each turn exactly once (8, 16 and 28 turns), so the prompt's "one entry per turn" is followed.

### D094 — Phase 9 step 4: the effort tie rule, made exact before the pilot is scored
- **2026-10-14 · Phase 9, step 4 · by Claude, fixed before any pilot score was computed · Approved** (within D090's tie rule)
- D090 says a tie (under 5 points on 20 maps) goes to `medium`, but it doesn't name the metric. **`xhigh` is chosen only if its pilot ILO beats `medium` by at least 5 points on general or on focused** (argmax decoding, the 20 pilot maps). Otherwise `medium` is chosen. Both metrics count because the stop rule uses both.

### D095 — Phase 9 step 4: pilot result; effort `medium`; one timing log slip repaired
- **2026-10-14 · Phase 9, step 4 · by Claude (pre-set rule D094) · Done** (`results/ilo_llm/pilot.json`; 20 pilot training maps, 1 answer each, argmax decoding)

  | Effort | ILO general / focused | Link recall / precision | Thinking tokens p50 | s per call p50 |
  |---|---|---|---|---|
  | **`medium`** | **88.61 / 81.39** | 98.56% / 99.27% | 0 | 4.6 |
  | `xhigh` | 85.64 / 80.33 | 99.64% / 100% | 265 | 5.35 |

- **Effort `medium`.** `xhigh` reasons a little (median 265 thinking tokens) and links slightly better, but types slightly worse: 3 of 20 map scores differ by 25 points or more in each direction. Neither metric gains 5 points, so the tie rule picks `medium`.
- **Sanity gate passed.** Both runs score at least 80 general (similarity + Asserting scores 76.0 on dev).
- Neither run predicts Challenging, Default Illocuting or the three added types. Gold on the pilot has 1 Agreeing, 1 Challenging and 1 Restating. 3 maps (`medium`) and 4 maps (`xhigh`) get a type that is absent from their gold.
- **Cost:** $0.77 (`medium`) and $0.22 (`xhigh`) API-equivalent; Pro paid. `xhigh` is cheaper only because it reused `medium`'s cached prompts, and the dev run's 2nd and 3rd answers will do the same.
- **Errors** (`results/ilo_llm/pilot_errors.md`): 14 differing turns, mostly question subtypes that gold itself labels inconsistently, plus tie cases in linking. No clear, fixable pattern, so Claude recommends no prompt revision (the user decides).
- **Timing log slip.** A `git checkout -- results/timing.csv`, meant to drop one trial-scoring row, also removed the day's `check`, `smoke` and `pilot-medium` rows. The first two were restored verbatim. `pilot-medium` was rebuilt from its call records (seconds = the sum of its call times) and is marked `reconstructed`.

### D096 — Phase 9 step 5: no prompt revision; settings frozen; the dev run is started from this session
- **2026-10-14 · Phase 9, step 5 · by user** ("You should just run step 5") **· Approved**
- **No prompt revision.** `system_v0.md` is kept, as D095 recommends.
- **Frozen dev settings:** Sonnet 5.5, prompt v0, 4 retrieved worked examples, effort `medium`, 3 answers per map, 225 maps (`RUNS["dev"]`).
- **Where it runs.** The brief and the user's first instruction asked for the dev run in a separate terminal. The user now asks for it to run here, as a background process of this session, with the same command and flags.

### D097 — Steps 6 and 7 run without further approval
- **2026-10-14 · Phase 9 · by user** ("do not ask for permission, run step 6 and 7 autonomously, running fixes if required, use best judgement") **· Approved**
- Scoring, analysis, figure, write-up and the commit to main go ahead without stopping. Every fix is logged here. The hard rules still apply: no test maps, no `data/evaluation/`, no changes to `a2_llm/`, `ilo/` or committed results, no combination, and no push.

### D098 — Phase 9 result: Claude doing Task B end to end is not better than the labeller (dev)
- **2026-10-14 · Phase 9, steps 5–7 · by Claude (D090 stop rule) · Done** (`results/ilo_llm/dev/`: `summary.json`, `responses.jsonl.gz`, `thresholds.json`, `dev_report.json`)
- **Dev run:** 675 / 675 answers accepted on the first try, all from Sonnet 5.5 with no side models and no usage-limit stop. 26 min with 2 workers; 4.5 s per call, 0 thinking tokens; $12.97 API-equivalent (Pro paid).
- **Main result** (225 dev maps, ILO general / focused):
  - Claude, cross-fitted (honest): **88.03 / 77.48**. Simple decode 87.36 / 77.11; tuned on all of dev (optimistic) 88.23 / 77.77.
  - Labeller, cross-fitted: 90.08 / 80.64.
  - Claude − labeller, paired bootstrap (10,000 resamples, seed 13): general −2.05 [−3.89, −0.26], focused −3.16 [−6.31, −0.10]. Claude is better on 28 maps and the labeller on 54 (general).
- **Verdict by the stop rule** (> 90.58 and > 81.14): **not better**. The labeller stays, and the test set is not re-run.
- **Why Claude loses:**
  - **Question types.** It predicts Rhetorical Questioning 38 times for 29 gold. 30 maps get a type absent from their gold (labeller: 18).
  - **Linking.** Recall / precision 97.89% / 98.69% (labeller 98.33% / 99.62%). Of 56 turns only Claude links, 21 are right; all 35 wrong ones take a proposition that gold credits to another turn, typically a relayed or quoted claim. Of 40 turns only the labeller links, 38 are right.
  - **Rare types.** Claude's answers propose Agreeing 45 times and Disagreeing 12 times, but the tuned thresholds switch all six rare types off in every fold. Both systems therefore score 0 on them.
  - **Thresholds.** The tuned link threshold sits at the grid floor (0.01): one vote in three is enough, because one-to-one matching already blocks most false links.
- **Fixes and choices made along the way (D097: no stops):**
  - `ilo_llm/evaluate.py` runs in `~/.venvs/a3`, as A4's evaluation did, because `a3_finetune.tune.bootstrap` imports `xgrammar`.
  - The figure is drawn with a throwaway numpy + matplotlib venv outside the repo. No project venv has matplotlib, and the existing figure script says any Python with both works.
  - The analyses (linking, per type, rare-label trap) use each system's *honest* links: every map is decoded with its held-out fold's thresholds. This is asserted to reproduce both honest scores exactly. The labeller's honest linking (98.33% / 99.62%) therefore differs slightly from its reported tuned-on-all-dev figure (98.27% / 99.62%).
  - `dev_report.json` also records Claude's raw type counts and, for each wrong link, the turn gold credits the proposition to.
- **Write-up:** `reports/03-ilo-labeller.md` §11 (with Figure 3, `reports/figures/ilo-claude-vs-labeller.png`, drawn by `reports/ilo_claude_figure.py`), one sentence in the README's findings, and CLAUDE.md §7.

### D099 — Dev test, part 2: can Jev do Task B better than the trained labeller? Protocol fixed before any call
- **2026-10-14 · Phase 10 · by user** ("Can you have Jev run this as well, very similar, do this end to end without much input from me") **· Approved; settings by Claude**
- *Question:* as D090, with Jev (A4's model) in place of Claude. Given dev maps in test format, can Jev link each turn to its proposition and type it better than the labeller (honest dev ILO 90.08 / 80.64)?
- *Design (`ilo_jev/`):*
  - **State.** A guide (the task, the 10 types with D091's spec-only definitions, and the annotators' measured habits, as Claude's `system_v0.md`) plus the map serialised with no links.
  - **Questions.** Two `choice` questions per turn, each asked in both option orders and averaged (A4, D076):
    - *which proposition does turn T express?* Every proposition is an option, plus `none`;
    - *how does it put it forward?* The 10 types.
  - **Probabilities.** P(T → P, type) = P(link P) × P(type), in `ilo.decode`'s 11-label format, so the labeller's decision layer applies unchanged.
  - **Calls.** One pass, since Jev gives probabilities, not votes.
- *Pilot:* the same 20 pilot maps, argmax decoding, comparing k = 0 against k = 4 retrieved worked examples (A2's pool minus the pilot episodes).
  - *Rule, fixed now (as D077):* keep k = 4 only if it gains ≥ 1.0 ILO general; otherwise k = 0.
  - *Sanity gate:* pilot general ≥ 80.
  - No wording revision.
- *Dev:* the 225 maps, the chosen k, the labeller's decision layer (no temperature, as for Claude), cross-fitted over the same 5 episodes.
- *Stop rule:* as D090. Jev is better only if its honest dev ILO exceeds 90.58 and 81.14. The paired bootstrap against the labeller and against Claude (88.03 / 77.48) is reported either way.
- *Budget:* paid gateway credit (balance $13.30; A4 has used $1.70 of D077's $10). The cost is projected before any call, and work stops and asks if the projection exceeds **$2**.
- *Out of scope:* test maps, `data/evaluation/`, and changes to `a4_jev/`, `ilo/`, `a2_llm/` or committed results.

### D100 — Jev pilot: k = 4 by rule; the sanity gate fails at plain argmax, and a diagnostic shows why
- **2026-10-14 · Phase 10 · by Claude (pre-set rule D099; diagnostic under D097's "use best judgement") · Done** (`results/ilo_jev/pilot.json`; smoke $0.004, pilot $0.044)
- *Smoke:* 3 / 3 maps complete and valid, model `typesafe-ai/jev`, 1.4 s per request.
- *Pilot* (20 maps, argmax, ILO general / focused; link recall / precision):

  | Arm | Argmax | Argmax, six rare types off (diagnostic) | Links | Maps with a type absent from gold |
  |---|---|---|---|---|
  | k = 0 | 68.96 / 57.49 | 85.67 / 79.29 | 98.92% / 100% | 14 |
  | k = 4 | 73.59 / 63.04 | 85.55 / 78.45 | 98.56% / 100% | 11 |

- **k = 4**, by D099's rule (+4.63 general ≥ 1.0).
- **The sanity gate (≥ 80) fails for both arms at plain argmax.**
- **Diagnostic, added after the gate failed:** score the same argmax with the six rare types switched off (falling back to Asserting), which is what the tuned layer did for both Claude and the labeller. Both arms then clear 80 (85.67 and 85.55), and linking is near perfect.
- *Reading:*
  - The pipeline is sound. Jev's weakness is typing: unthresholded, it invents rare types, Agreeing 7 times for 1 gold, which triggers the rare-label trap.
  - k = 4's gain comes almost entirely from inventing fewer rare types; with them off, the two arms tie.
  - The dev decision layer handles this as it does for every system. The dev run goes ahead at k = 4, about $0.31 projected.

### D101 — Phase 10 result: Jev doing Task B end to end is not better than the labeller (dev); the labeller gets a discussion report
- **2026-10-14 · Phase 10 · by Claude (D099 stop rule; write-up structure under the user's "whatever makes most sense structurally") · Done** (`results/ilo_jev/`: `check.json`, `pilot.json`, `smoke/`, `pilot-k0/`, `pilot-k4/`, `dev/` with `summary.json`, `responses.jsonl.gz`, `thresholds.json`, `dev_report.json`)
- **Gates:**
  - the gold round trip equals Claude's (98.70 / 96.28);
  - no node IDs in any request;
  - the cost was projected at $0.36 before any call, and $0.34 was spent: smoke $0.004, pilot $0.044, dev $0.288. The gateway balance went from $13.30 to $12.97;
  - the dev run had 247 / 247 requests answered, 225 / 225 maps valid, model `typesafe-ai/jev`, 0.51 s per request, 44 s in all, no retries.
- **Main result** (225 dev maps, ILO general / focused):
  - Jev, cross-fitted (honest): **85.35 / 75.76**. Simple decode 74.42 / 61.16; tuned on all of dev (optimistic) 86.10 / 76.74.
  - Labeller: 90.08 / 80.64. Claude: 88.03 / 77.48.
  - Jev − labeller, paired bootstrap: general −4.73 [−6.73, −2.77], focused −4.88 [−7.98, −1.83].
  - Jev − Claude: general −2.68 [−4.62, −0.86], focused −1.72 [−4.62, +1.17].
- **Verdict by the stop rule:** **not better**. Neither general model replaces the labeller, and the test set is not re-run.
- **Why:**
  - **Linking.** Jev links best of the three: 98.98% / 99.56%, against the labeller's 98.33% / 99.62%. 24 of its 32 extra links are right, mostly turns whose proposition was rebuilt from context.
  - **Typing.** Before thresholds, Jev makes a rare type its first choice on 147 linked turns, against 26 rare gold links. The tuned layer switches all rare types off, as it did for Claude and the labeller. Jev is then left under-calling questions: F1 Pure 0.746, Assertive 0.333, Rhetorical 0.098, against the labeller's 0.870 / 0.400 / 0.500.
  - **Rare-label trap.** 17 maps get a type absent from their gold (labeller 18, Claude 30).
- **Write-up:**
  - **03 report §11** now covers both models (Claude and Jev), with a three-way figure. `reports/figures/ilo-llms-vs-labeller.png`, drawn by `reports/ilo_llm_figure.py`, replaces the Claude-only figure and script, which are deleted.
  - **New `reports/03-ilo-discussion.md`:** a short companion about the labeller itself (why a small trained model holds Task B, what the two general models showed, where its remaining points are). It follows the discussion companions of A2–A4 and is indexed in `reports/README.md`.
  - The README's findings sentence and CLAUDE.md §7 now cover both models.
- **Code changes, outputs unchanged:**
  - `ilo_llm/tune.py`: `ScoredDev` and `tuned_report` were split out for reuse. Re-running Claude's tuning reproduced every score and threshold; only the 16th digit of one map's float differed, from set-order summation, so the committed file was kept.
  - `ilo_llm/evaluate.py`: `disagreements` takes the system's name; the default `"claude"` keeps the old output.

### D102 — The README explains why Task B uses a trained encoder
- **2026-10-15 · Write-up · by user** ("update the readme with the findings, very briefly, as a part of the reasoning as to why we chose the ILO labeller to be the encoder, and why it might be that the encoder was better") **· Done**
- **New README section *Why Task B uses a trained encoder*, after *How it works*.** It gives:
  - the original reasons (EDA: matching under a fixed convention; rare types weigh fully in the macro F1, so thresholdable probabilities were needed);
  - the D090 / D099 tests as a four-row table;
  - three data-backed reasons the encoder wins: linking is worth under 1 point; typing follows the annotators' habits; the decision is mostly local.
- Findings item 1 is shortened to point there, and the README's stale count of decisions is corrected. No result changed.
