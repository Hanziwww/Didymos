# Materials: preprocessing (raw sources to harmonized MEDS)

One pipeline for all seven sources: raw archives -> CLIF 3.0 tables per dataset -> harmonized MEDS v0.4 with the ICU
stay as subject -> tokens. The stages are described in `docs/meds.md` and `docs/m4_tokenizer.md`;
every defect found and fixed was recorded in the project's data-quality log (#1-#119), which is not part of this release.

## Stages

| Stage | What | Reference implementation adopted | Our part |
|---|---|---|---|
| S1 staging | Archives extracted to the native PhysioNet / AMDS layouts; a Parquet mirror with row-count parity; deterministic 2% subsets for development | — | `icutwin.stage` |
| S2 CLIF | Per dataset: the CLIF 2.1 tables (+ 3.0 structures: RRT including IHD, input/output, MCS) and `ext_icu_stay` | MIMIC-IV: the official CLIF-MIMIC ETL (pinned, one idempotent patch); AmsterdamUMCdb: CLIF-AmsterdamUMCdb (run as an external tool) | Adapters for eICU, HiRID, SICdb, NWICU, INSPIRE driven by mapping tables with per-row provenance (ricu concept dictionaries, BlendedICU drug maps, mimic-code/eicu-code SQL, source dictionaries, manual for INSPIRE); all outputs normalized by clifpy's CLIF 2.1 -> 3.0 crosswalk |
| S3 QA | CLIFpy schema and mCIDE validation; cross-dataset distribution flags (median ratios, ranges, coverage); event-level reconciliation against the community MEDS ETLs | CLIFpy 2.1.0; community MEDS ETLs (MIMIC_IV_MEDS, eICU_MEDS, AUMC MEDS-Transforms example, SICdb_MEDS, INSPIRE_MEDS, NWICU and HiRID MEDS-DEV), pinned | Reconciliation with per-dataset bridges: every unpaired row needs a documented reason |
| S4 MEDS | ELF 1.0.0-beta codes (`{DOMAIN}//{l1}//{l2}//{l3}`), qualified codes (route, blood-pressure site, categorical value), canonical units; derived concepts | CLIF_MEDS (our engine reproduces it exactly: 0 differing events in all seven datasets) | ICU-stay subjects, windows, splits, derived concepts (below) |
| S5 tokens | `docs/paper/tokenization.md` | CoMET, ETHOS-ARES, the tokenization benchmark | Spec-driven tokenizer and grammar |

## Uniform rules (one rule each, applied to every dataset)

| Topic | Rule |
|---|---|
| Unit of analysis | An adult ICU stay (age at hospital admission at least 18), from 48 hours before ICU admission to ICU discharge; header and outcome events of its hospitalization |
| Splits | 80 / 5 / 15 (train / tuning / held-out) per dataset, grouped by patient through a hash of dataset and patient id |
| Monitor streams (HiRID, AmsterdamUMCdb, SICdb, eICU periodic) | Per (stay, category, clock hour): a single value keeps its time; several become their median, stamped at the hour's last measurement (the ricu / YAIB / ICareFM hourly semantics) |
| Continuous medications | One row per rate change; doses converted only with weights charted at or before them (0 violations in all seven datasets) |
| Binned demographics (AmsterdamUMCdb, HiRID, SICdb) | Bin midpoints with a flag |
| Billing diagnoses | Discharge-time events in the end block, never in a prediction input |
| Header fields documented after admission | Masked in prospective inputs until their documentation window ends (24 h for diagnosis group and surgical status in eICU, HiRID, AmsterdamUMCdb, SICdb; 1 h INSPIRE) |

## Derived concepts (labels only, never model inputs)

| Concept | Definition | Validation |
|---|---|---|
| KDIGO AKI stage (creatinine, urine output, CRRT) | mimic-code `kdigo_creatinine`, `kdigo_uo`, `kdigo_stages` | Logic parity with mimic-code exact (1.0 on every column) |
| Ventilation and other respiratory episodes | mimic-code `ventilation` with clifpy's device waterfall | Identical to mimic-code on its inputs |
| Vasopressor episodes, norepinephrine equivalents | infusion start to stop; mimic-code NEE (Goradia 2020) | Identical to mimic-code |
| RRT episodes | CRRT or IHD documentation, 24 h gap | — |
| SOFA (first 24 hours) | clifpy `compute_sofa_polars`; respiration as the worst score of concurrent P/F pairs (QA #66) | Decomposition against mimic-code documented (not equivalent: mimic-code lacks ventilated P/F 200-400 branches) |
| Adult Sepsis Event | clifpy `compute_ase` (blood culture + 4 antimicrobial days + organ dysfunction) | MIMIC and AmsterdamUMCdb only (others lack cultures or administered antimicrobials) |

## Full-scale gates (`docs/m5_training.md` "Server run")

| Gate | Result |
|---|---|
| CLIF 3.0 validation (clifpy) | 0 hard errors in all seven datasets |
| MEDS parity (our engine against CLIF_MEDS) | 0 differing events in all seven datasets |
| Weight causality | 0 violations in all seven datasets |
| mimic-code parity | KDIGO 1.0 on every column; ventilation and NEE episodes identical |
| Reconciliation with the community ETLs | every unpaired row with a documented reason, except two data defects of the training snapshot, fixed since (#95: 4 INSPIRE records; #98: HiRID infusion rates rounded to 6 decimals) |
| Tokenizer | 0 rejections or round-trip failures over all 409,297 stays |

Limits stated as such: medication reconciliation is presence-only where a source
has no dose, HiRID is compared on total dose only, SICdb's multi-case rate streams are excluded; it is not a complete
dose-and-timing validation.
