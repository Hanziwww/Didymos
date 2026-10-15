# Reference checkouts

Shallow clones of the published reference implementations the plan aligns with. Exact commits are in
`REFS.lock` (`name commit url`). Re-create with the clone list in that file; nothing here is vendored
into `src/`.

License notes (checked 2026-10-03):

| Repo | License | How we use it |
|---|---|---|
| CLIF, CLIF_MEDS, ELF, clifpy | Apache-2.0 | mCIDE/ELF vocabularies, CLIF→MEDS ETL, validation |
| CLIF-MIMIC, MIMIC_IV_MEDS, INSPIRE_MEDS, SICdb_MEDS, MEDS-DEV, ethos-ares, BlendedICU, AmsterdamUMCdb, HIRID-ICU-Benchmark, YAIB | MIT | ETLs run as-is; mappings and task definitions as reference |
| mimic-code | MIT | concept SQL (KDIGO, NEE, ventilation) ported to `icutwin.meds.derived`; its DuckDB concepts run unchanged for the parity check (`icutwin_meds task=mimic_code`) |
| NWICU_MEDS, HiRID_MEDS, AUMCdb_MEDS | MIT | community MEDS ETLs run as-is for the S3 item reconciliation |
| tabicl | BSD-3-Clause | episode contract, compatibility test |
| ricu | GPL-3.0 | concept-dictionary JSON read as internal reference data only; never redistributed |
| CLIF-AmsterdamUMCdb, flair_baseline, synthetic_clif | no license file | run as external tools or read for reference only; no code copied |
