# Community MEDS ETLs: environments and commands

The S3 reconciliation compares our CLIF output with independent
MEDS ETLs run on the same raw subsets. `src/icutwin/clif/community.py` runs them unchanged, at the commits
pinned in `refs/REFS.lock`; this page records how to reproduce those runs exactly.

## Environment (WSL2)

The ETLs' stage runner (MEDS-Transforms) launches each stage as a shell command with POSIX quoting, so the
ETLs run under bash on Linux: in WSL2 on this Windows machine (below), directly on a Linux host such as the
server. On a Linux host the scripts get the paths as they are, and three environment variables replace the
WSL settings:
- `ICUTWIN_COMMUNITY_CACHE`: the per-ETL environments and the uv cache (default `$HOME/.cache/icutwin`);
- `ICUTWIN_PYPI_INDEX`: the index for the MEDS-Transforms installs (default `https://pypi.org/simple`; the
  study's WSL runs used the mirror below);
- `UV_PYTHON_PREFERENCE`: default `only-system` (a CPython 3.12 on the host); `managed` lets uv download one.

The provenance record of a run names its host (`linux` or `wsl:<distro>`) and its package index; the lock
check is the same on both.

| Item | Value |
|---|---|
| Distribution | `Ubuntu-24.04` (Ubuntu 24.04.4 LTS), WSL 2.6.3, the default distribution; default user `root`; systemd on (`/etc/wsl.conf`) |
| Virtual disk | `D:\wsl\Ubuntu-24.04\ext4.vhdx` (moved from `C:\Users\<user>\AppData\Local\wsl\...` on 2026-10-04; *WSL migration* below) |
| Python | the distribution's CPython 3.12.3 (`UV_PYTHON_PREFERENCE=only-system`) |
| uv | 0.11.2, `/root/.local/bin/uv` |
| Caches | `$HOME/.cache/icutwin/uv` (uv cache, about 1.6 GB) and `$HOME/.cache/icutwin/venvs/<ETL>` (one environment per ETL) |
| Package index | `https://pypi.tuna.tsinghua.edu.cn/simple` (MEDS-Transforms ETLs, set with `ICUTWIN_PYPI_INDEX`); PyPI's file host for `uv sync --frozen` (MEDS-Extract ETLs). pypi.org's index was not reachable from the study's WSL network; files.pythonhosted.org and GitHub were |
| Configuration | `src/icutwin/configs/clif.yaml`: `community_wsl_distro` (default `Ubuntu-24.04`), `community_root`, `community_proxy` (HTTP(S) proxy for the MIMIC concept-map download) |

## The seven ETLs

Every environment is locked by `src/icutwin/clif/community_envs/<ETL>.txt`: the exact package set of the
validated run (`uv pip freeze`, with a three-line header naming the commit, the environment and the use). Each
run writes its installed set to `<out>/_environment.txt` and **fails when it differs from the lock** (`diff`
in the run script under `set -euo pipefail`).

| Dataset | ETL (commit) | Runner | Install | Pins (deviation from the declared range) | Input staging | Entry point |
|---|---|---|---|---|---|---|
| MIMIC-IV | MIMIC_IV_MEDS (`7a2b35c608a4`) | MEDS-Extract 0.7 | `uv sync --frozen` from its `uv.lock` | – | raw subset (hard links) plus the 10 mimic-code v2.4.0 concept maps its spec pins by sha256 (downloaded once, verified) | `meds-extract-run spec=MIMIC-IV` |
| INSPIRE | INSPIRE_MEDS (`8002aa88cfc8`) | MEDS-Extract 0.7 | `uv sync --frozen` | – | raw subset | `MEDS_extract-INSPIRE` (wrapper: stages its subject side table, then `meds-extract-run`) |
| NWICU | NWICU_MEDS (`f637178e1c01`) | MEDS-Extract 0.7 | `uv sync --frozen` | – | raw subset | `meds-extract-run spec=NWICU` |
| eICU | eICU_MEDS (`d42deb031b78`) | MEDS-Transforms 0.2.4 | `uv pip sync` of the lock, then the ETL `--no-deps` | – (its own pins: meds-transforms 0.2.4, polars 1.26.0) | raw subset | `MEDS_extract-eICU` |
| SICdb | SICdb_MEDS (`d3ef9b9f2c41`) | MEDS-Transforms 0.2.4 | as eICU | – | raw subset | `MEDS_extract-SICdb` |
| HiRID | HiRID_MEDS (`5e1a21c8f247`) | MEDS-Transforms 0.2.4 | as eICU | **polars 1.26.0**: the declared `polars<1.30` resolves to 1.29.0, which panics in parquet predicate pushdown while sharding events; 1.26.0 is eICU_MEDS's pin | fresh tree per run (`stage_hirid`): the subset's CSV parts as the release's typed parquet variant under the PhysioNet processor names, header-only parts skipped, `reference_data.tar.gz` packed as released | `MEDS_extract-HIRID` |
| AmsterdamUMCdb | AUMCdb_MEDS (`209b521597f8`) | MEDS-Transforms 0.2.4 | as eICU | **meds-transforms 0.2.4**: the declared `meds-transforms>=0.1` resolves to 0.7, which has no `MEDS_transforms.utils.get_shard_prefix`; 0.2.4 is the release contemporary with the commit (2025-03-05) and the one the other 0.2-era ETLs pin | raw subset | `MEDS_extract-AUMCdb` (run from the ETL's directory) |

A pin is also asserted in code (`community.ETLS[...].pins`): `lock_file()` refuses a lock that lacks it
(`tests/test_community.py::test_a_lock_without_the_pins_is_refused`).

## Commands

Run the ETLs (all, or some) and then the reconciliation, from the repository root on Windows:

```bash
.venv/Scripts/python.exe -m icutwin.clif.cli task=community
```

```bash
.venv/Scripts/python.exe -m icutwin.clif.cli task=community "datasets=[hirid,aumc]"
```

```bash
.venv/Scripts/python.exe -m icutwin.clif.cli task=reconcile
```

`task=community` renders one script per ETL into `data/work/<mode>/community/<dataset>.sh` and runs it with
`wsl -d <community_wsl_distro> -- bash <script>`; the log goes to `data/community_meds/<mode>/<dataset>.log`.
The output directory is emptied first (MEDS-Extract skips stages whose outputs exist). The HiRID script, as
rendered for the subset (the repository at `D:\project`, `ICUTWIN_PYPI_INDEX` set to the mirror):

```
#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$HOME/.cache/icutwin/uv" UV_PYTHON_PREFERENCE=only-system
VENV="$HOME/.cache/icutwin/venvs/HiRID_MEDS" LOCK="/mnt/d/project/src/icutwin/clif/community_envs/HiRID_MEDS.txt"
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12
uv pip sync --python "$VENV/bin/python" --index-url https://pypi.tuna.tsinghua.edu.cn/simple "$LOCK"
uv pip install --python "$VENV/bin/python" --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps "/mnt/d/project/refs/HiRID_MEDS"
export PATH="$VENV/bin:$PATH" PYTHONIOENCODING=utf-8
mkdir -p "/mnt/d/project/data/community_meds/subset/hirid"
uv pip freeze --python "$VENV/bin/python" > "/mnt/d/project/data/community_meds/subset/hirid/_environment.txt"
diff <(grep -v '^#' "$LOCK" | LC_ALL=C sort) <(grep -v -e '^-e ' -e ' @ file://' "/mnt/d/project/data/community_meds/subset/hirid/_environment.txt" | LC_ALL=C sort)
MEDS_extract-HIRID root_output_dir="/mnt/d/project/data/community_meds/subset/hirid" raw_input_dir="/mnt/d/project/data/work/subset/community/hirid_input" do_download=False do_overwrite=True
```

MEDS-Extract ETLs replace the three install lines with
`UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --project <ETL>` and run, e.g. for MIMIC-IV,
`meds-extract-run spec=MIMIC-IV output_dir=<out> do_download=false input_dir=<staged input>`.

Each run writes `data/community_meds/<mode>/<dataset>_provenance.json`: ETL, commit, runner, WSL
distribution, lock file and its sha256, pins, input directory, start time, duration and exit code.

To change an environment: edit the lock (or regenerate it from a validated run's `_environment.txt`, keeping
the header), keep any pin listed in `ETLS`, and rerun `task=community` and `task=reconcile`.

## Verification (2026-10-04)

All seven ETLs ran on the subsets three times in this environment: on the original distribution (C:), on a
clone imported on D:, and on the distribution after the move to D:. Each run passed its environment check, and
the reconciliation of every run gave identical counts in all 28 tables and no unexplained row (subset run
times on the moved distribution: MIMIC 99 s, AUMC 80 s, eICU 109 s, HiRID 50 s, SICdb 19 s, NWICU 22 s,
INSPIRE 21 s).

## WSL migration (2026-10-04)

The distribution was moved from C: (nearly full) to D:, keeping its name, default status, default user and
files:

1. `wsl --export Ubuntu-24.04 D:\wsl\migrate\Ubuntu-24.04.vhdx --format vhd` and
   `wsl --import Ubuntu-24.04-migrate D:\wsl\Ubuntu-24.04-migrate <export> --vhd`: a clone on D:, verified
   (user, uv, Python, the seven environments, uv cache, systemd) and used for a full community run and
   reconciliation (identical results).
2. `wsl --shutdown` and `wsl --manage Ubuntu-24.04 --move D:\wsl\Ubuntu-24.04`: the original moved; checked
   again (registration: base path `D:\wsl\Ubuntu-24.04`, default distribution, uid 0) and used for another
   full community run and reconciliation (identical results).
3. The clone (`wsl --unregister Ubuntu-24.04-migrate`) and the export file were then removed.

C: free space, measured with WSL shut down before and after the move: 31,926,722,560 -> 35,869,290,496 bytes
(+3,942,567,936 bytes; the virtual disk was 3,815,768,064 bytes). An earlier reading taken while the WSL
virtual machine was running (27.35 GB free) is not comparable: the running VM holds a swap file on C:.
