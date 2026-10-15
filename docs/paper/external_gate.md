# Amendment 16a: the external databases' pre-run gate

Recorded before any future was drawn on RICD or zigongICU (`tools/paper_external.py gate`; seed 16). Counts and shares only. The rule is pre-registered in `docs/p1_paper_protocol.md` (amendment 16a): an endpoint is scored on a database when its defining source has timing to the hour and at least 20 anchors of the design are positive.

## Encoding with the frozen vocabulary

| Database | Stays | Grammar rejections | Parser rejections | Round-trip failures | Re-encode failures | Goal mismatches |
|---|---|---|---|---|---|---|
| RICD | 5,382 | 0 | 0 | 0 | 0 | 0 |
| zigongICU | 2,899 | 0 | 0 | 0 | 0 | 0 |

**Out-of-vocabulary events** (a domain's OOV token, share of the domain's events; the training databases' held-out stays for reference):

| Domain | Training held-out | RICD | zigongICU |
|---|---|---|---|
| ADT | 0.0% of 199,798 | 12.7% of 16,825 | 0.1% of 30,840 |
| AIRWAY | 0.0% of 2,452 | none | none |
| CRRT | 0.0% of 461,597 | none | 0.0% of 487 |
| DRAIN | 0.0% of 9,460 | none | none |
| HEMO | 0.0% of 1,417,658 | none | none |
| ICU | 0.0% of 122,364 | 0.0% of 10,764 | 0.0% of 5,798 |
| IHD | 0.0% of 8,231 | none | none |
| INPUT | 0.0% of 1,913,110 | none | none |
| LAB | 0.0% of 11,345,038 | 0.0% of 1,563,346 | 0.0% of 919,042 |
| LINE | 0.0% of 18,769 | none | none |
| MCS | 3.7% of 62,723 | none | none |
| MED_CON | 0.0% of 2,006,431 | 85.1% of 6,258 | 55.5% of 14,373 |
| MED_INT | 0.1% of 1,883,584 | 31.6% of 660,433 | 6.5% of 211,183 |
| MED_ORD | 0.0% of 1,095,329 | none | none |
| MICRO | 0.9% of 35,949 | none | none |
| MICRO_NC | 0.0% of 826 | none | none |
| MICRO_ORDER | 0.0% of 57,780 | none | none |
| MICRO_SUS | 3.3% of 21,464 | none | none |
| OUTPUT | 0.0% of 1,825,850 | 14.5% of 312,278 | 0.0% of 751,681 |
| PA | 0.0% of 5,284,245 | 0.0% of 40,378 | 0.0% of 1,252,126 |
| POS | 0.0% of 566,892 | none | none |
| PROC | 0.1% of 317,405 | none | none |
| PT_DX | 0.0% of 81,238 | none | none |
| RESP | 0.0% of 9,500,813 | 0.0% of 40,457 | 0.0% of 1,677,678 |
| VITAL | 0.0% of 28,510,101 | 0.0% of 6,880,082 | 0.0% of 2,868,148 |

**Values against the training deciles:** for each numeric code with at least 200 values, the share in the lowest and the highest bin (each holds about one tenth of the training values). Codes with more than half in an end bin:

- RICD: 16 of 109 codes:
  - `MED_INT//ext_budesonide//mcg//given//inhaled`: 0% lowest, 100% highest bin (n = 1,173)
  - `MED_INT//ext_acetylcysteine//mcg//given//iv`: 100% lowest, 0% highest bin (n = 2,336)
  - `MED_INT//trimethoprim_sulfamethoxazole//mcg//given//iv`: 0% lowest, 100% highest bin (n = 6,401)
  - `MED_INT//acetaminophen//mg//given//iv`: 2% lowest, 98% highest bin (n = 871)
  - `MED_INT//ext_ketorolac//mcg//given//iv`: 5% lowest, 95% highest bin (n = 539)
  - `MED_INT//ext_thiamine//mcg//given//im`: 89% lowest, 0% highest bin (n = 353)
  - `MED_INT//ext_tranexamic_acid//mcg//given//iv`: 82% lowest, 0% highest bin (n = 643)
  - `OUTPUT//procedural_drain_other`: 0% lowest, 76% highest bin (n = 486)
  - `OUTPUT//urinary_other`: 0% lowest, 68% highest bin (n = 263,196)
  - `MED_INT//ext_acetylcysteine//mcg//given//inhaled`: 66% lowest, 0% highest bin (n = 28,772)
  - `MED_INT//ext_calcium//mcg//given//iv`: 66% lowest, 3% highest bin (n = 525)
  - `OUTPUT//chest_tube_temporary`: 0% lowest, 63% highest bin (n = 293)
  - `MED_INT//vancomycin//g//given//enteral`: 60% lowest, 0% highest bin (n = 9,185)
  - `MED_INT//cefepime//g//given//iv`: 0% lowest, 51% highest bin (n = 419)
  - `LAB//ext_creatine_kinase//u/l//misc`: 51% lowest, 0% highest bin (n = 11,162)
  - `LAB//so2_arterial//%//blood_gas`: 13% lowest, 50% highest bin (n = 5,630)
- zigongICU: 0 of 76 codes.

Intermittent doses are discrete: a standard dose at a bin edge fills one end bin without a unit error. Urine output is charted as totals over intervals (canonical CLIF; quartiles):

- RICD: 283,264 entries, 4.0 h between entries (IQR 3.0-6.0), 500 mL per entry (IQR 300-600).
- zigongICU: 758,063 entries, 0.3 h between entries (IQR 0.2-0.8), 100 mL per entry (IQR 50-150).

## Anchors

- **R** (A1's rolling design, 500 per database): 1,000 anchors, 2 left out by F1's clock check, 998 kept (RICD 500, zigongICU 498).
- **L24e** (the 24-hour landmark's case-cohort sample): 7,179 landmark anchors, 0 left out by the clock check; RICD: 350 of 677 in-hospital deaths and 350 of 4176 others (weight 11.93); zigongICU: 350 of 593 in-hospital deaths and 350 of 1733 others (weight 4.95).

**Alignment control** (each kept anchor's real stay decoded as a rollout member on its own clock reproduces the real store's label; the minimum agreement over binary programs without a clinical cut):

- R: 1.0000 over 998 anchors and 50 programs.
- L24e: 1.0000 over 1,400 anchors and 52 programs.

## Endpoints: prevalence and the availability decision

| Design | Endpoint | Database | Labelled anchors | Positive | Weighted prevalence | Source | Scored |
|---|---|---|---|---|---|---|---|
| R | decompensation@1440 | RICD | 500 | 103 | 2.3% | death at the last department discharge (hours) | yes |
| R | imv_onset@1440 | RICD | 498 | 0 | 0.0% | a daily ventilation flag on the SOFA record only | no |
| R | vasopressor_onset@1440 | RICD | 490 | 31 | 0.7% | vasopressor orders (hours) | yes |
| R | rrt_onset@2880 | RICD | 500 | 0 | 0.0% | no renal replacement source | no |
| R | aki_2@2880 | RICD | 475 | 51 | 2.7% | creatinine (hours) and urine output from monitoring (minutes) | yes |
| R | sepsis_ase@1440 | RICD | 500 | 0 | 0.0% | culture results not harmonized | no |
| R | hypotension@240 | RICD | 492 | 6 | 0.1% | blood pressure from monitoring (minutes) | no |
| R | hyperkalemia_top@1440 | RICD | 457 | 26 | 1.4% | laboratory potassium (hours) | yes |
| R | thrombocytopenia@2880 | RICD | 468 | 7 | 0.9% | laboratory platelets (hours) | no |
| R | decompensation@1440 | zigongICU | 498 | 171 | 15.5% | death at the hospital discharge (hours from admission) | yes |
| R | imv_onset@1440 | zigongICU | 225 | 24 | 4.1% | ventilation from the nursing chart, which records dates only | no |
| R | vasopressor_onset@1440 | zigongICU | 445 | 56 | 5.4% | vasopressor prescriptions (hours) | yes |
| R | rrt_onset@2880 | zigongICU | 491 | 12 | 1.6% | haemofiltration replacement-fluid prescriptions (hours) | no |
| R | aki_2@2880 | zigongICU | 491 | 54 | 6.8% | creatinine (hours) with urine output from the nursing chart (dates only) | no |
| R | sepsis_ase@1440 | zigongICU | 498 | 0 | 0.0% | no culture data | no |
| R | hypotension@240 | zigongICU | 473 | 16 | 3.0% | blood pressure from the nursing chart (dates only) | no |
| R | hyperkalemia_top@1440 | zigongICU | 455 | 23 | 3.0% | laboratory potassium (hours) | yes |
| R | thrombocytopenia@2880 | zigongICU | 448 | 21 | 3.9% | laboratory platelets (hours) | yes |
| L24e | icu_death@-1 | RICD | 700 | 289 | 11.5% | the patient pathway (hours) | yes |
| L24e | hospital_death@0 | RICD | 700 | 350 | 14.0% | the last department discharge (hours) | yes |
| L24e | remaining_los_gt_7d | RICD | 700 | 507 | 71.3% | the patient pathway (hours) | yes |
| L24e | decompensation@1440 | RICD | 700 | 8 | 0.3% | death at the last department discharge (hours) | no |
| L24e | icu_death@-1 | zigongICU | 700 | 274 | 20.0% | ICU transfers and the hospital discharge (hours) | yes |
| L24e | hospital_death@0 | zigongICU | 700 | 350 | 25.5% | the hospital discharge (hours) | yes |
| L24e | remaining_los_gt_7d | zigongICU | 700 | 184 | 25.3% | ICU transfers (hours) | yes |
| L24e | decompensation@1440 | zigongICU | 700 | 77 | 5.6% | death at the hospital discharge (hours from admission) | yes |

An endpoint not scored on a database is reported as not defined there; it is not imputed.
