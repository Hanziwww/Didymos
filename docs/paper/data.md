# Materials: data

Seven credentialed ICU databases, harmonized into one cohort of 409,297 adult ICU stays (`docs/paper/preprocessing.md`).
Table 1 is computed from every stay's tokens, decoded as the model sees them (`tools/paper_table1.py` ->
`runs/paper/materials/table1.json`; aggregates only).

## Sources

| Dataset (version) | Setting | Country | Admission period in the data (era token) | Access |
|---|---|---|---|---|
| MIMIC-IV (3.1) | Beth Israel Deaconess Medical Center, single centre | USA | 2008-2022 (3-year anchor groups) | PhysioNet credentialed |
| eICU-CRD (2.0) | 208 hospitals (multi-centre telehealth programme) | USA | 2014-2015 | PhysioNet credentialed |
| HiRID (1.1.1) | Bern University Hospital, single centre | Switzerland | not released | PhysioNet credentialed |
| AmsterdamUMCdb (1.0.2) | Amsterdam UMC, single centre | Netherlands | 2003-2016 (two 7-year groups) | Amsterdam Medical Data Science, data-use agreement |
| SICdb (1.0.8) | Salzburg University Hospital, single centre | Austria | 2013-2021 | PhysioNet credentialed |
| NWICU (0.1.0) | Northwestern Medicine, Chicago | USA | 2020-2022 (97.9%) | PhysioNet credentialed |
| INSPIRE (1.4.2) | Seoul National University Hospital, post-operative ICU stays | South Korea | not released | PhysioNet credentialed |

All processing ran locally or on a host approved under each dataset's data-use agreement; only aggregates leave it.

## Table 1: cohort by dataset (all splits)

| | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|
| ICU stays | 89,201 | 199,120 | 33,896 | 18,383 | 27,179 | 27,680 | 13,838 |
| Train / tuning / held-out | 71,434 / 4,544 / 13,223 | 159,293 / 9,896 / 29,931 | 27,194 / 1,691 / 5,011 | 14,748 / 888 / 2,747 | 21,708 / 1,373 / 4,098 | 22,178 / 1,402 / 4,100 | 11,107 / 659 / 2,072 |
| Age, mean (SD), years ^a | 65.2 (16.6) | 63.7 (17.0) | 66.0 (15.1) | 60.8 (15.1) | 68.5 (15.2) | 63.4 (16.8) | 61.8 (15.3) |
| Female, % | 43.5 | 46.0 | 35.8 | 32.3 | 37.6 | 43.8 | 42.6 |
| ICU length of stay, h, median (IQR) | 48.2 (27.1-94.8) | 37.6 (19.7-71.8) | 24.2 (19.3-53.0) | 31.1 (22.3-116.1) | 35.2 (19.8-75.8) | 46.7 (25.4-93.1) | 25.8 (20.6-50.6) |
| ICU death, % | 8.2 | 5.5 | 6.1 | 12.0 | 3.7 | 8.0 | 1.2 |
| In-hospital death, % ^b | 12.6 | 9.0 | 6.1 | 12.0 | 7.8 | 11.6 | 5.2 |
| Invasive ventilation in the ICU, % | 41.1 | 25.7 | 56.5 | 86.7 | 19.9 | 19.8 | 78.2 |
| Vasoactive infusion in the ICU, % | 32.0 | 11.1 | 35.5 | 69.8 | 33.5 | 22.4 | 9.4 |
| RRT in the ICU, % | 6.5 | 3.9 | 5.4 | 5.8 | 3.8 | 4.9 | 3.0 |
| Timeline events per stay, median | 693 | 471 | 539 | 796 | 583 | 270 | 513 |
| Admission type (largest) | ED 87.5% | ED 41.2% | not recorded | elective 50.4% | elective 34.3% | ED 69.7% | elective 77.3% |
| Race recorded | yes (white 67.7%, Black 11.2%) | yes (white 77.2%, Black 10.6%) | no | no | no | yes (white 67.1%, Black 13.7%) | single-centre Korean cohort |

^a Ages are 5-year bands decoded at their midpoints (the sources bin ages differently).
^b HiRID and AmsterdamUMCdb record mortality at ICU discharge only, so in-hospital death equals ICU death there.

Annotations:
- The sources differ by design: AmsterdamUMCdb and INSPIRE are dominated by post-operative ventilated patients,
  eICU by general ICUs with low treatment intensity, NWICU by the COVID era. This heterogeneity is what the dataset
  token conditions on and what the virtual-cohort analysis (C3) tests.
- Treatments are counted from the derived episodes' token sets (`factory.oracle.onsets`): any IMV device record, any
  vasoactive infusion start or running record, any CRRT or IHD record in the ICU window.

## Sets used by the experiments

| Set | Stays | Use |
|---|---|---|
| `real_test` (held-out split) | 2,000 per dataset (14,000) | A1, A2, A4, A5, B2, C1-C4: prompts and labels |
| `real_dev` (tuning split) | 659-3,000 per dataset (12,013) | A1's dry run, recalibration, early stopping of the baselines |
| `real_train` (train split) | 5,000 per dataset (35,000) | baselines' and probes' training rows; rare strata (C3) |
| `real_control` (train split) | 2,000 per dataset (14,000) | the real-against-real control (C3) |
| Twin bank (`bank_dev`) | 13,416 twins (eICU 3,975, MIMIC 2,608, HiRID 1,601, NWICU 1,492, SICdb 1,471, AmsterdamUMCdb 1,203, INSPIRE 1,066) | A4, C3 |

## External databases (amendment 16)

Two databases from countries not in the training data, used for the transfer analysis (16a). Neither was seen in
training, tuning or model selection, and the model is not trained on them.

| Dataset (version) | Setting | Country | Admission period | Access |
|---|---|---|---|---|
| RICD (3.0), Russian Intensive Care Dataset | Federal Research and Clinical Center of Intensive Care Medicine and Rehabilitology, Moscow; intensive care and neurorehabilitation, single centre | Russia | December 2017 to September 2025 | Open dataset (https://fnkcrr-dataset.ru/). The archive carries no licence file; the study lead confirmed its terms of use (2026-10-11) |
| zigongICU (1.1) | Zigong Fourth People's Hospital; ICU patients with infection, single centre | China | not released (times are hours from hospital admission) | PhysioNet credentialed (PhysioNet Credentialed Health Data License 1.5.0) |

- **Archives:**
  - `dataset/RICD/RICD-3.0.tar.zst`, sha256 `261350cf24283876261b02a37024167cedd9d3f64c2828edddf14d159c68e124`;
  - `dataset/zigongICU/zigongicu-1.1.tar.zst`, sha256 `c86a523479e45dab98d5a3f34c5b09ad898325c9f408643d1ae2101b8b4ce285`.
- **Count-only audit (before amendment 16):**
  - RICD: 10,938 hospitalizations of 8,135 patients, 4,526 of them with an ICU stay, and 5,942 ICU periods.
  - zigongICU: 2,790 hospitalizations, one per patient, of which 48 are of patients under 18.
  - Final cohort counts come from the ETL and its gate report (`docs/paper/external_gate.md`).
- **Harmonized cohort** (adapters `src/icutwin/clif/adapters/{ricd,zigong}.py`; adult ICU stays over the paper's
  window; every stay a test stay):

  | Dataset | ICU stays | MEDS events | Tokens | ICU deaths | In-hospital deaths |
  |---|---|---|---|---|---|
  | RICD | 5,382 | 9,648,056 | 12,958,156 | 579 | 13.3% |
  | zigongICU | 2,899 | 7,804,407 | 9,021,560 | 716 | 29.5% |
- **Processing:** the ETL runs locally. Per the study lead (2026-10-11), the tokenized stays may be processed on both
  approved GPU hosts. Only aggregates leave them.
