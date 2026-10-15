# S4: harmonized MEDS (ICU stay = subject)

`icutwin_meds task=build` turns each dataset's canonical CLIF 3.0 directory into one MEDS v0.4 dataset.

```
data/meds/<mode>/<dataset>/
  data/{train,tuning,held_out}/<shard>.parquet   subject_id, time, code, numeric_value, text_value,
                                                 patient_id, hospitalization_id, route_basis (extension
                                                 columns; route_basis on medication events only)
  metadata/dataset.json                          MEDS DatasetMetadataSchema (+ event_domains)
  metadata/codes.parquet                         every code with a description (and parent code if qualified)
  metadata/subject_splits.parquet                subject_id -> split
  metadata/stays.parquet                         subject_id -> stay_id, patient, hospitalization, ICU times,
                                                 window start, split, clock_valid
  metadata/_build_report.json
```

Every table is checked with the `meds` package schemas before it is written (`DataSchema`,
`CodeMetadataSchema`, `SubjectSplitSchema`).

## Events and codes

- **Codes** are ELF 1.0.0-beta codes, `{DOMAIN}//{level_1}//{level_2}//{level_3}`.
- **CLIF_MEDS** (the CLIF consortium's converter) produces its 15 domains. ECMO_MCS is skipped because it
  reads the CLIF 2.1 table.
- **Our ELF engine** (`meds/elf.py`) executes the same YAML config language as SQL and produces the
  extension domains in `meds/elf_ext/`:

| Domain | CLIF table | Code |
|---|---|---|
| ICU | ext_icu_stay | `ICU//admission//<unit_type>`, `ICU//admission_source//…`, `ICU//surgical_status//…`, `ICU//adm_dx_group//<group>` (when known), `ICU//stay_seq`, `ICU//discharge//<outcome>`; static `DATASET//<ds>`, `SITE//<ds>//<site>` |
| HEMO | invasive_hemodynamics | `HEMO//<measurement_category>` |
| INPUT / OUTPUT | input / output | `INPUT//<category>`, `OUTPUT//<category>` (mL) |
| IHD | intermittent_dialysis | `IHD//session`, `IHD//<flow>` |
| MCS | mcs | `MCS//<support>//<device>//<config>` (state), `MCS//<support>//<setting>` |
| AIRWAY / LINE / DRAIN | airway / line / drain | `AIRWAY//<category>//placed|removed`, … |
| MED_ORD | medication_orders | `MED_ORD//<category>//<unit>//start`, `MED_ORD//<category>//NA//stop`, each qualified by the route (below) |
| MICRO_ORDER, MICRO, MICRO_NC, MICRO_SUS | microbiology tables | collection, organism, PCR result, susceptibility |
| PT_DX | patient_diagnosis | `PT_DX//<format>//<code>` |

- **Qualified codes (M4, `meds/qualifiers.yaml`):** one level is appended to the ELF code of four domains,
  and `codes.parquet` gives the ELF code as the qualified code's `parent_codes`, so ELF-only consumers can
  collapse it. Our engine builds these domains from the exact concept configs of the CLIF_MEDS run
  (`<elf_dir>/_concept_config`, overlay included), the medication ones after the same clifpy dose conversion
  CLIF_MEDS's processors apply; the build uses them instead of CLIF_MEDS's unqualified files.

| Domain | Qualified code | Level |
|---|---|---|
| MED_INT, MED_CON | `MED_INT//<category>//<unit>//<action>//<route>` | `med_route_category` (`UNK` where the source states none, e.g. SICdb) |
| MED_ORD | `MED_ORD//<category>//<unit>//start|stop//<route>` | the declared extension column `medication_orders.med_route_category` |
| VITAL | `VITAL//sbp|dbp|map//arterial|cuff` (other vitals unqualified) | `meas_site_name` |
| PA | `PA//<category>//<categorical value>` for results without a score (CAM-ICU, SBT) | `categorical_value`, normalized |

- **Descriptions** mark categories outside CLIF 3.0 (`[not CLIF 3.0 vocabulary: <origin> extension]`, origin
  consortium practice, local or local concept; `clif/medication_extensions.csv`) and routes outside a table's
  CLIF 3.0 route list (`[route 'inhaled' not CLIF 3.0 vocabulary of the table: local extension]`).
- **Unit conversion:** CLIF_MEDS converts doses to its preferred units (`meds/clif_meds_overlay.yaml`); a
  category without a preferred unit gets clifpy's base unit (rates per minute, intermittent masses in mcg);
  units clifpy does not know (mmol, mEq, counting units) stay as recorded, counting units in one spelling
  across sources (`tab`, `cap`, `puff`, `neb`, `inh`, `drop`, `sachet`, `each`, ...; `clif/units.py`).
- **Route basis (M4, #78):** every medication event carries `route_basis`, what its route category rests on:
  - `reported`: the source names the route;
  - `reported_class`: a route class or order category (AmsterdamUMCdb 'Injecties ...' also covers subcutaneous
    injections);
  - `inferred`: a rule on drug and dose (NWICU 'Injection');
  - `assumed`: the record type alone (an infusion without a source route);
  - `verified`: reserved; no source supplies a clinically verified route.

  Rules: `clif/route_basis.csv`. The route level of the code is always the source's category, never corrected
  clinically; an analysis that needs a given route reads `route_basis`.

### Reference path and training MEDS

Two outputs share the engine (`meds/elf.py`):
- **Reference path:** CLIF_MEDS's own configs, row-by-row clifpy conversion with the weight charted at or
  before each dose, unqualified codes. `icutwin_meds task=parity` checks this path against the CLIF_MEDS run,
  event by event (every domain, every dataset), and it is exact. This is what lets the engine replace CLIF_MEDS's
  row-by-row Python at full scale.
- **Training MEDS** (`data/meds/<mode>/`, what the tokenizer reads): the reference path plus the documented
  build rules. The medication rules change events, so its medication domains are deliberately not
  CLIF_MEDS-equal:

| Rule (QA log) | What changes against the reference path | Subset counts |
|---|---|---|
| Qualified codes (D2) | one level appended (route, BP site, categorical value); `parent_codes` gives the reference code | every MED, BP and categorical PA event |
| Causal segment units (#76) | an infusion that starts before any weight keeps its unweighted unit to its stop; the reference converts its later rows | rows: SICdb 2,900, NWICU 737, HiRID 122, eICU 120, MIMIC 101 |
| One stream per code (#75) | the streams of a drug that land on one code are summed again | eICU 505 -> 486 events; elsewhere events unchanged |
| Window-start state (#77) | a `going` event at the window start for each infusion running then | MIMIC 116, eICU 83, INSPIRE 71, NWICU 6, SICdb 5 |
| Cohort (#73) | stays of patients under 18 are not subjects | eICU 12, SICdb 4 |

`_build_report.json` gives these counts per dataset (`infusion_streams`, `infusions_running_at_window_start`,
`stays_below_min_age`).

### The prediction timeline

The training MEDS is built for prospective use: no value at time t uses information recorded after t, with
the bounded exceptions listed here.
- **Doses** are converted only with weights charted at or before them (#76). `icutwin_meds task=weights`
  checks it on the data: a per-kg value before the hospitalization's first weight, or in an infusion segment
  that began before it, must be the source's own unit. Subsets: 0 violations among 121,495 per-kg events;
  6,245 such events, all charted per kg at the source (NWICU 6,184).
- **Infusions running at the window start** come from the events before the window (#77).
- **Header fields** describe the stay at ICU admission (#79). Where documentation of the admission may follow
  it, the grammar spec gives the window (`available_within_hours`: 24 h for diagnosis group and surgical status
  in eICU, HiRID, AmsterdamUMCdb and SICdb, 1 h INSPIRE, else 0). A prospective input before that window masks
  the field (`tokenization.encode.prospective`).
- **Derived concepts are labels**, never inputs of the token stream. NEE and KDIGO urine-output rates keep
  mimic-code's first-weight fallback, needed for parity with mimic-code, and flag each value it converts
  (`weight_charted_later`). A prospective task drops or recomputes such values when the weight was charted after
  its cutoff.
- **Bounded look-ahead of S2 conventions:** a task's gap between cutoff and target must be at least these
  horizons, or the task must recompute the events at its cutoff.
  - Bag or syringe exchanges: a stop followed by a restart within 15 min is not written as a stop
    (`canonical_continuous`).
  - eICU and NWICU infer an infusion's stop 60 min after its last confirmation when none follows within 6 h
    (ricu's interval rule): up to 6 h.
  - Hourly monitor medians are stamped at the hour's last measurement and use nothing later.

## Subjects, windows and splits

- **Subject:** an ICU stay of `ext_icu_stay` with ICU time (discharge after admission) of an adult: age at
  hospital admission at least `min_age_years` (18; `null` keeps every age; binned ages at their bin midpoint,
  so SICdb's 15–19 bin is out; a stay of unknown age is kept). Derived concepts use the same cohort, and
  `_build_report.json` counts the stays left out. Its `subject_id` is a 63-bit hash of the stay id, stable
  across extracts.
- **Static events:** the patient's static events (sex, race, ethnicity, dataset, site), plus birth and death.
- **Timeline:** the hospitalization's events from 48 h before ICU admission to ICU discharge.
- **Header and outcomes:** the hospitalization's header and outcome events, whatever their time: admission
  type and age at hospital admission, discharge disposition and billing diagnoses at hospital discharge.
- **ICU-stay events** belong to their own stay only. An earlier stay inside the pre-ICU window is history,
  which ADT carries.
- **Identical events** (same subject, time, code, value and ids) are kept once, as MEDS-Transforms keeps one
  row per distinct event: some sources repeat raw rows or carry one measurement under two items that map to
  one code. The count is `identical_events_dropped` in `_build_report.json`. Doses and volumes (MED_INT, INPUT,
  OUTPUT) are exempt: their codes carry the category, not the item, so two flushes of 10 mL through two lines
  at one time are two events.
- **Event order:** time (static first), then a fixed domain order, then code.
- **Splits:** 80/5/15 per dataset (train/tuning/held_out), grouped by patient through a hash of dataset and
  patient id, so a patient keeps its split in every extract.

## Derived concepts (`icutwin_meds task=derived`, never tokenized)

`data/derived/<mode>/<dataset>/`:

| File | Definition |
|---|---|
| `sofa_24h.parquet` | SOFA over the first 24 ICU hours (clifpy `compute_sofa_polars`, given CLIF 2.1 device names, which clifpy 2.1.0 ranks by). Deviation: the respiration score is the worst SOFA score of clifpy's concurrent P/F pairs (clifpy scores the lowest P/F only and gives a P/F < 200 without support no score) |
| `nee.parquet` | norepinephrine equivalents (mimic-code `norepinephrine_equivalent_dose`, Goradia 2020) at every vasopressor rate change; per-minute or per-hour rates use the weight in effect (mimic-code `weight_durations`: last weight at or before, else the first) |
| `episodes.parquet` | episodes overlapping the stay window (start, end): respiratory support per CLIF device (mimic-code `ventilation`: new episode at a status change or 14 h after the previous charting, ending at the next charting within 14 h, single chartings dropped; devices missing on ventilator rows inferred with clifpy's waterfall heuristics), `vasopressor` (start to stop of vasoactive infusions), `rrt` (CRRT or IHD documentation, 24 h gap) |
| `kdigo.parquet` | KDIGO AKI stage at every creatinine, urine-output and CRRT time (mimic-code `kdigo_creatinine`, `kdigo_uo`, `kdigo_stages` at the pinned commit): creatinine against the lowest of 48 h / 7 days; urine output (bladder irrigation inflow netted) per kg over 6/12/24 h, needing as many hours of documentation; CRRT = stage 3; `aki_stage_smoothed` = highest stage of the past 6 h. Deviations in `icutwin.meds.derived` |
| `fluid_balance.parquet` | fluid in (input, ml/hr infusions integrated, ml boluses) and out per ICU day |
| `ase.parquet` | CDC Adult Sepsis Event per blood culture (clifpy `compute_ase`: blood culture + >= 4 qualifying antimicrobial days + organ dysfunction within 2 days; `sepsis` with, `sepsis_wo_lactate` without the lactate criterion), with `stay_id` when the culture falls in a stay window; MIMIC and AmsterdamUMCdb only (eICU has no administered antimicrobials, the other sources no cultures) |

### Parity with mimic-code (`icutwin_meds task=mimic_code`, MIMIC only)

mimic-code's DuckDB concepts (`refs/mimic-code/mimic-iv/concepts_duckdb`, pinned) run unchanged on the MIMIC
parquet mirror. Two checks, reported in `data/qa/<mode>/mimic_code_parity.{json,md}`:
- **Logic parity:** our KDIGO, ventilation-episode and norepinephrine-equivalent code, fed mimic-code's own
  inputs, must reproduce mimic-code's tables exactly; the task fails otherwise.
- **End-to-end agreement:** our concepts from canonical CLIF against mimic-code's from raw MIMIC, per ICU
  stay (AKI stage, first-day urine output, admission weight, vasopressors and maximum NEE, IMV / NIV / HFNC,
  RRT, first-day SOFA). Differences come from the inputs (CLIF-MIMIC's device and drug mappings, our
  supplements) and from documented mimic-code limitations.
- **First-day SOFA, component by component** (`sofa_decomposition`): our SOFA and mimic-code's
  `first_day_sofa` differ in definition (window, lab time, renal urine-output criterion, P/F construction,
  MAP density, GCS of intubated patients), so they are not equivalent; each disagreeing stay is attributed to
  a definition, a reference-implementation defect or an input difference.
