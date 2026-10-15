# Materials: training

MUC32_main's run (`runs/train/full/MUC32_main/config.json`, `train_report.json`, `FROZEN.json`), and the lineage of
candidates that led to it (`docs/m5_training.md`, `docs/m7_authenticity.md`, `docs/m7_refinement.md`). Every
adoption decision was taken by a rule written before the candidate was measured.

## MUC32_main's recipe

| Item | Value | Source | Note |
|---|---|---|---|
| Data | r4 tokens (`data/tokens/full_r4`), train split of all 7 datasets | `config.json` | r4 = r3 + the ICU-day clock blocks (`docs/m7_refinement.md` amendment 1) |
| Objective | Next-token cross-entropy over every token except the goal block and the clock blocks (loss-masked) | `model/data.py` `labels_for` | The clock blocks are written by the automaton at sampling, never predicted |
| Passes per dataset | MIMIC 1.59, AmsterdamUMCdb 3.05, eICU 2.66, HiRID 2.42, SICdb 2.01, NWICU 1.96, INSPIRE 1.81 | `config.json` `passes` | Projected per dataset to a membership-inference AUC of 0.555 from the measured per-pass slopes (amendment 4); the fit each dataset may reach is bounded by the privacy gate |
| Training-time transforms | goal block in 25% of windows (random field subsets), header fields shuffled, dataset and site unknown in 10%, each header field unknown in 5% | `config.json` `transforms` | Unknown dataset gives the unconditional mode used by B2 |
| Optimizer | Muon on hidden weight matrices (Newton-Schulz orthogonalized momentum, Moonshot RMS matching), AdamW on embeddings and norms | `model/optim.py` | Amendment 2; same held-out NLL is reached in fewer steps, the membership signal depends on the fit, not the optimizer |
| Learning rate | 6e-4, 100 warmup steps, cosine decay to 10% | `config.json` | |
| Weight decay, betas, clipping | 0.1; (0.9, 0.95); 1.0 | idem | |
| Batch | 65,536 tokens per micro-batch and GPU (padding-free packing), 4 accumulation steps, 2 GPUs: 524,288 tokens per step | idem | |
| Steps, tokens | 1,990 steps, 997.5M tokens (target 986.7M) | `train_report.json` | |
| Hardware, time | 2 x RTX PRO 6000 Blackwell (96 GB), 2,843 s of training (47 min; 3,295 s wall), 350,800 tokens/s, peak 45 GB per GPU | idem | `torch.compile` (inductor) |
| Regularization | none (no dropout, no EMA, no input noise) | idem | Round 3 tested dropout and input noise; neither was adopted |
| Seed | 1234 | idem | |

## Result

| Measure | Value | Source |
|---|---|---|
| Tuning NLL (nats per target token, whole tuning split, 28.7M target tokens) | 1.5525 | `train_report.json` `final_eval` |
| By dataset | AmsterdamUMCdb 1.239, SICdb 1.342, INSPIRE 1.510, eICU 1.537, MIMIC 1.678, HiRID 1.696, NWICU 1.889 | idem |
| M7 privacy gate | PASS (P1-P5) | `runs/audit/m7_MUC32_main/report.json`; `docs/paper/evaluation_protocols.md` |

## The lineage (every candidate trained, the rule that decided it)

| Candidate | Size, context | Change | Tuning NLL | Privacy gate (P1) | Decision |
|---|---|---|---|---|---|
| S | 71.5M, 8k | first full run, r2 tokens | 1.614 | not audited | M better on all 7 datasets |
| M_main | 123.5M, 8k | the default size | 1.582 | FAIL (INSPIRE AUC 0.650 at 5.9 passes) | passes rebalanced |
| M32_main | 123.5M, 32k | context 32k, RoPE 1e6, 3 passes (2.5 for the two smallest), r3 tokens | 1.561 | PASS | the first candidate (round 2) |
| ML32_main | 269.5M, 32k | 2.2x larger at the same passes | 1.563 | PASS, with members-only gains (membership AUC +0.010-0.016; MIMIC upper bound 0.592) | not adopted: capacity at privacy-bound passes became memorization |
| MC32_main | 123.5M, 32k | ICU-day clock blocks (r4) | 1.564 | PASS | treatment-onset over-prediction barely moved |
| MCN32_main | idem | + input-noise augmentation | 1.607 | PASS | out (worse NLL) |
| MU32_main / MUH32_main | idem | Muon, lr 6e-4 / 1.2e-3 | 1.527 / 1.509 | INCONCLUSIVE / FAIL (MIMIC 0.604 / 0.623) | over the gate: Muon reaches the privacy bound in fewer steps |
| MUF32_main | idem | Muon, passes projected to AUC 0.565 | 1.5388 | INCONCLUSIVE (TPR at 1% FPR upper bound 0.031 > 0.03) | not adopted |
| **MUC32_main** | idem | Muon, passes projected to AUC 0.555 | **1.5525** | **PASS (AUC 0.549-0.555)** | **adopted** (study lead, amendment 5): authenticity A1-A5 met 30.5 of 35 against 25.0, calibration cells no worse |

Key findings from the lineage (each a measured result, `docs/m7_refinement.md` "Round 3 summary"):
1. The membership signal is a function of the fit reached, whatever the optimizer (AdamW and Muon checkpoints lie on
   one curve of P1 AUC against held-out NLL); so the privacy gate bounds each dataset's fit, and per-dataset passes
   can be set on that frontier.
2. A larger model at privacy-bound passes did not generalize better (ML32_main).
3. Treatment starts after the first ICU day are over-predicted step by step (teacher forcing) because rare-event
   timing is under-fitted; explicit ICU-day and treatment-status blocks barely help.
