# GPU-host job scripts

These are the scripts as they were run on the study's GPU hosts. They are kept as a record of how each run was
made. They are not a portable workflow.

## The hosts

Each script's header names the host it ran on. All three were rented cloud GPU hosts.

- **The two-GPU host**: 2 × RTX PRO 6000 Blackwell (96 GB each). It ran the full-scale data pipeline, training, the
  privacy audits and model selection.
- **The one-GPU host** (`westc`): RTX 4080 SUPER 32 GB, 16 CPUs, 62 GiB of memory. It ran the twin bank, the
  article's rollouts and analyses, amendment 16b and amendment 16a's comparators.
- **The RTX 5090 host** (`westd`): the toolkit's validation (amendment 11), the rest of amendment 16, and amendment 17.

## Conventions

- **Layout.** The scripts assume `BASE=/root/autodl-tmp`:
  - the repository is at `$BASE/icutwin`;
  - the logs are in `$BASE/logs`, as `<job>.log` with `<job>.exit`;
  - the project environment is `$BASE/icutwin/.venv` and vLLM's environment is `$BASE/vllm-env` (recipe:
    `setup_env_5090.sh`).
- **Running elsewhere.** Change `BASE`, or edit the paths at the top of a script.
- **Execution.** Each step runs detached through `run.sh`, and `status.sh` shows where the steps are.
- **The privacy gate.** Every script that samples from a model goes through `generate.producer.require_gate`, so it
  stops unless that model's privacy audit has passed.
- **What leaves a host.** Only aggregates are copied off the hosts. Twins, rollouts and per-stay outputs stay there
  and are deleted once scored, as each script states.

## The scripts by purpose

| Purpose | Scripts |
|---|---|
| Setup and job control | `run_setup.sh`, `setup_env_5090.sh`, `setup_r_env.sh` (the clinician study's pinned R environment), `run.sh`, `status.sh`, `export_pack.sh` |
| Data pipeline (S1-S5) | `orchestrate_s2.sh`, `orchestrate_s4.sh`, `clif_chain.sh`, `clif_groups.py`, `reconcile_parallel.sh`, `recon_status.py`, `orchestrate_r3.sh` |
| Training and model selection (M5, rounds 2-3) | `train.sh`, `m32_run.sh`, `m32_train.sh`, `ml32_run.sh`, `round3_*`, `round4_muon.sh`, `round5_frontier.sh` (Didymos 1.0, `MUC32_main`), `freeze_run.py`, `ckpt_average.py`, `eval_nll_at.py`, `nll_compare.py`, `nll_traj.py` |
| Calibration and treatment QA | `calib_*`, `cmp_calm.py`, `qa_after_authd.sh`, `qa_joint_summary.py`, `qa_summary.py`, `qa_tokens.py` |
| Goal-conditioned sampling | `goalfirst_m32.sh`, `gf_draws.py`, `gf_probe.py`, `gf_twin_stats.py`, `goal_diag.py`, `check_goalfirst.py` |
| Privacy gate (M7) | `m7_gate.sh`, `m7_run_m32.sh`, `m7_rerun_m32.sh`, `m7_p5v2_m32.sh`, `m7_summary.py`, `p1_compare.py`, `p1_curve.sh`, `p1_families.py`, `audit_validity.py`, `gate_record_v3.py` |
| Authenticity, pilots and the twin bank (M7-M8) | `authenticity_m32.sh`, `authd_finish.sh`, `auth_*.py`, `canon_rerun.sh`, `diversity_breakdown.py`, `target_margins.py`, `target_means.py`, `reaudit_events.sh`, `ref_audits.sh`, `pilot_m32.sh`, `pilot_m32b.sh`, `pilot_diagnose.py`, `forecast_m32.sh`, `forecast_run.sh`, `bank_*` |
| The article's analyses (`docs/p1_paper_protocol.md`) | `paper_*` (A1-C4 and amendments 2-17), `ext17_*` (amendment 17) |
| The toolkit's validation (amendment 11) | `toolkit_*` |
| The episode factory and the twin supply | `factory_chain.sh`, `factory_stores.sh`, `rollouts_bench.*`, `supply_*` |
