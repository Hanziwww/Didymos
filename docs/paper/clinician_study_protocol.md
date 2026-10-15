# Clinician evaluation of the ICU digital twin: protocol (draft)

Status: agreed by the study lead (2026-10-10); it enters `docs/p1_paper_protocol.md` as amendment 12 and is registered
on OSF before the first rating. It replaces the protocol's sketch "Clinician study (prepared in this phase, run
later)", and keeps that sketch's design for module A. The existing ethics approval covers the study, and the centers
and raters are authorized. Sizes come from the pre-specified simulation `tools/clinician_study_power.py` (sections
5-7; its outputs `runs/toolkit/clinician_power*.json`).

## 1. Aims

Paper P shows statistically that MUC32_main's twins match real ICU stays, that its rollouts predict real outcomes and
that its what-if responses reproduce observational contrasts. This study asks critical care physicians three
questions the statistics cannot answer.

| Module | Question | Primary endpoint |
|---|---|---|
| A. Realism | Can intensivists tell a twin's first 24 hours from a real stay's? | Classification accuracy, tested for equivalence to chance |
| B. Plausibility | Are the twin's simulated responses to treatment and its explanations clinically sensible? | B1: plausibility of simulated courses against real ones (non-inferiority); B2: relevance of the model's top factors against random factors |
| C. Decision support | Does the twin's forecast card improve intensivists' prognosis? | Change in Brier score, aided minus unaided |

A teaching evaluation is not part of this protocol.

## 2. Design

A multicenter, blinded, web-based evaluation study with critical care physicians as raters, in two sessions of about
90 minutes each, all on retrospective, de-identified data.
- **Item order** is randomized per rater.
- **Versions** are counterbalanced across raters (B1's courses, B2's methods).
- **Order of modules** is fixed: A first, while raters are naive to the twin; C last, so exposure to the forecast card
  cannot inform the realism judgments.

| Session | Content | Time |
|---|---|---|
| 1 | Consent, tutorial (four practice items, not scored); A (80 items); B2 (20 items) | about 90 min |
| 2 | B1 (24 items); C (48 cases); the System Usability Scale and trust questions | about 100 min |

## 3. Participants

- **Raters:** 10-20 physicians from 2-4 centers.
  - *Eligible:* board-certified intensivists, or critical care fellows with at least two years of ICU practice.
  - *Recruitment target:* 16 raters, at least 12 for the primary analyses; at most 6 per center, at least a third
    fellows.
- **Excluded:** anyone who worked on the model, the toolkit or this study's stimuli. They know the twin's cues.
- **Credentialing (the data route):** each rater holds PhysioNet credentials for MIMIC-IV and eICU-CRD (CITI training,
  a signed data use agreement), verified by the center lead before any access.
- **Consent:** electronic informed consent at the start of session 1.
- **Anonymity:** centers and raters are anonymous in every output, as the approval requires. Centers are coded
  (center 1-4) and raters carry pseudonyms; only the center leads hold the keys. No name enters the repository, the
  host's records or the materials.
- **Compensation:** as the existing approval allows.

## 4. Data, stimuli and governance

### Data route
- **Real stays:** the held-out split only (stays the model never saw), from MIMIC-IV and eICU only.
- **Twins:** bank twins conditioned on those two datasets.
- **Governance:**
  - The bank is DUA-derived and internal. Showing it to credentialed raters under this study is recorded as the
    study lead's decision.
  - **Distribution (the study lead's decision, 2026-10-10, before any stimulus was built):** stimuli are generated on the
    approved 4080 host and given to each rater as one self-contained HTML packet (`toolkit.rating`), opened offline.
    No server, no network request, no third-party platform.
  - **Why this fits the agreements:** a packet holds renderings of MIMIC-IV and eICU stays, so it goes only to raters
    credentialed for both datasets, through the center's secure channel.
  - **The rater's undertaking:** at the start of the packet the rater attests the credentials and agrees to keep the
    packet confidential, not share it, and delete it and its exported files once the study team confirms receipt.
  - **What returns:** only the exported JSON (answers, times, a checksum), through the center lead. The key from
    pseudonyms to raters stays with the center leads.
  - **The answer key** (`key.json`: real or synthetic, versions, outcomes, comparators) and the item list never leave
    the approved host. Only aggregates leave it in the materials.
- **Blinding inside the packet:**
  - The file holds no answer: no real-or-synthetic label, no source reference, no dataset name, no outcome.
  - Item ids are opaque hashes with a secret salt kept on the host.
  - Module C's forecast cards are obfuscated, keyed by the packet, and decoded only after the unaided estimates are
    locked. This deters casual reading of the source; it is not encryption, and the instructions say so.

### Rendering parity
- Every stimulus, real or synthetic, is rendered from its token sequence through the same decoder (values drawn
  within their bins, with the vocabulary's rank persistence; `factory.twins.decode_features`). So neither the
  quantization nor the formatting can reveal the source.
- **The standardized case sheet** has three parts:
  - *Header:* the admission (age band, sex, unit type, admission source and diagnosis group).
  - *Time series:* small multiples over hours since ICU admission:
    - vital signs: heart rate, arterial pressures, respiratory rate, SpO2, temperature;
    - labs: sodium, potassium, creatinine, lactate, glucose, hemoglobin, platelets, white cells, bilirubin;
    - blood gases: pH, PaO2, PaCO2, FiO2;
    - urine output and fluids.
  - *Treatment lane:* vasoactive infusions with rates, ventilation (device, PEEP), sedation, renal replacement
    therapy.
- The dataset is not named. Items are matched on dataset, so its recording style cannot cue.
- The sheet extends the toolkit's blinded timeline (`toolkit.card.blinded`), and is fixed before the pilot.
- **Layout (2026-10-10, before the pilot):** every text has a zone of its own:
  - a left gutter for names, units and the legend (a name too long for it is truncated, with the full name as a
    tooltip);
  - the plot, for data only;
  - a right margin for the scale;
  - a top strip for the treatment-start label;
  - the axis rows.
- **Values inside the plot** (FiO2, PEEP, infusion rates) are written where they change and fit. Every record keeps
  a dot, so the charting rhythm stays visible. FiO2 is shown in percent.
- **Check:** a browser detector measured every text box of every sheet and card. The first build had collisions on
  every sheet; the released build has none. The same rules apply to real stays and twins.

### Pilot
Two physicians who are not raters run every module before the study, on the same stimuli. They check the timing, the
rendering and the instructions, and say whether anything other than the content gives the source away. Their ratings
are not analysed. Changes are recorded in the amendment before the first rater starts.
- **Materials:** packets P01 and P02 of the same build, and an eight-question feedback form
  (`pilot_feedback_form.txt`). The form asks about time per session, rendering faults, unclear instructions,
  non-content giveaways in A and B1, the card's clarity, and the kinds of cue used, which seed the codebook.
- **After the pilot:** a change to anything a rater sees makes a new build, and only the released build goes to
  raters. The build code is a hash of every packet's content and the page's code.
- **A short pilot set (the study lead, 2026-10-10):** five questions taken from P01 (`clinician_study.py pilot`;
  packets S01-S05, with their own build code).
  - *Contents:* two of module A's practice sheets (one real, one synthetic), and one item each of B2 (editing's
    factors), B1 (a twin course, not insulin) and C (both outcomes).
  - *Left out:* the practice round, the role and experience questions, the questionnaire and the session break.
  - *Feedback:* a five-question form.
  - *Scope:* it checks the flow, the display, the export and non-content giveaways in 10-15 minutes. It is not
    analysed.
- **Distribution:** `clinician_study.py archive` writes one zip per packet in number order, each with its
  instructions, together with the leads' README, the manifest and a tracking sheet without names.

## 5. Module A: realism

### Items
- 80 per rater (decided 2026-10-10, section 12), the same set for all raters, in a random order per rater:
  - 40 first 24-hour case sheets of real held-out stays (20 MIMIC-IV, 20 eICU);
  - 40 of twins, each matched to a real item on dataset and ICU length-of-stay band (`factory.oracle.los_band`).
- Four practice sheets (two real, two twins, outside the scored set) come first and are not scored.
- A stay shorter than 24 hours is shown to its end, with how it ended.
- **Selection:** random within strata (seeded), drawn by `tools/clinician_study.py select`. Stays used in modules B
  and C are excluded: module C tells raters its cases are real, so meeting the same stay there would reveal the
  answer. Nothing else is matched: the death deficit of short twins (F3 D4) is part of what is tested.

### Task
Raters are told that half the sheets are real and half synthetic. For each item they give:
- real or synthetic;
- confidence (1 guessing to 5 certain);
- optionally, the cue that decided them.

There is no feedback during the module.

### Endpoints
- **Primary:** the probability of a correct call, pooled.
  - Model: a mixed-effects logistic regression with random intercepts for raters and items.
  - Hypothesis: equivalence to chance. The twins count as indistinguishable when the 90% interval of the pooled
    accuracy lies within 0.40-0.60 (two one-sided tests at α = 0.05).
  - Reading: accuracy significantly above 0.60 means intensivists can tell; the cue analysis then shows how.
- **Secondary:**
  - the AUC of the 10-point confidence-weighted call (real 5 ... synthetic 5), by a bootstrap over raters and items;
  - accuracy by dataset, by seniority and by length-of-stay band;
  - each rater's accuracy against chance (exact binomial, Holm-adjusted);
  - inter-rater agreement (Fleiss' kappa);
  - item-level detection rates (the most-detected twins are inspected);
  - accuracy across item order (learning);
  - the cues, coded by two coders with a codebook fixed after the pilot (e.g. implausible values, charting rhythm,
    missing documentation, treatment logic, timing) and reconciled.

### Size
**The simulation:** 1,000 studies per cell. Calls are drawn from a logistic model with rater and item intercepts
(rater SD 0.3 on the logit scale; item SD 0.5 or 1.0, as heterogeneous as obvious twins and odd real stays make it).
The pooled accuracy is judged with a two-way cluster-robust interval.

| Raters × 60 items | True accuracy 0.50 | 0.52 | 0.55 |
|---|---|---|---|
| 12 | 0.76-0.91 | 0.75-0.85 | 0.44-0.52 |
| 16 | 0.82-0.96 | 0.77-0.90 | 0.46-0.62 |
| 20 | 0.85-0.98 | 0.80-0.94 | 0.50-0.66 |

Power to show equivalence; each range covers item SD 1.0 to 0.5.

**Reading:** with 60 items and 16 raters, the study shows indistinguishability with power of at least 0.8 if
intensivists are at chance, and about 0.8 if they are at 0.52. An accuracy of 0.55 would usually leave the result
undecided. Items count more than raters when items vary: 80 items with 16 raters give 0.91 at 0.50 and 0.85 at 0.52
(item SD 1.0), for about 15 more minutes in session 1.

**Decided:** 80 items (2026-10-10). The heterogeneity of items is the larger uncertainty, and the self-contained
packet lets a rater pause and resume.

## 6. Module B: plausibility

### B1: simulated responses to treatment

**Items:** 24 per rater, from F5's factual design (held-out MIMIC-IV and eICU starters, cut one minute before a real
start):
- 4 each for norepinephrine, fluid bolus, furosemide, invasive ventilation and CRRT;
- 4 for insulin, whose glucose response F5 found not reproduced: a known-failure stratum, which shows whether the
  ratings can detect an implausible simulation.

**Each item shows:**
- the history's case sheet up to the cut;
- the statement "<treatment> was started here";
- the next 6 hours of the response channels (MAP; urine output; glucose; potassium and creatinine; SpO2 and FiO2),
  either:
  - the twin's course under the give arm: one rollout drawn at random with a fixed seed, never selected; or
  - the stay's real observed course.

**Counterbalancing:** each rater sees every item once, in one version. Versions are counterbalanced so each item is
seen in both across raters.

**Task:** raters are told that each course is either observed or a model simulation, and rate:
- the course's physiological plausibility given the treatment (1-5);
- whether the response is in the expected direction (yes, no, unsure).

**Endpoints:**
- *Primary:* twin courses are non-inferior to real courses in mean plausibility, margin -0.5 points. Model: a linear
  mixed model with the source as a fixed effect, random intercepts for raters, items and stimuli, and the insulin
  stratum excluded. A stimulus is an item in one version (one drawn rollout, or the real course). Non-inferior
  when the lower bound of the 95% interval lies above -0.5.
- *Secondary:* per treatment; the expected-direction rate; insulin's twin courses against real (expected inferior).

**Size:** the simulation draws 1-5 ratings with item, rater and residual SDs of 0.6, 0.4 and 0.9, and analyses
per-item contrasts. With 20 items (insulin excluded), the power for non-inferiority is:
- 0.99-1.00 when twin and real courses are equally plausible (12-20 raters);
- 0.73, 0.84 and 0.92 with 12, 16 and 20 raters when twin courses are 0.2 points less plausible.

### B2: the model's explanations

**Items:** 20 per rater: held-out MIMIC-IV and eICU rolling anchors in the upper half of the Twin probe's risk of
death within 24 h (10) or of a new vasopressor within 24 h (10), as F4 selected them.

**Each item shows:**
- the case sheet;
- the model's risk;
- three "factors the model relied on", each an event group (a channel family × a time block before the anchor), with
  its three most recent events.

**The factors' source:** one of three methods, blinded and counterbalanced across raters:
- counterfactual editing (the paper's primary method);
- attention knockout (amendment 9);
- three random groups (the control).

**Task:** raters rate how relevant the factors are to the risk (1-5), and flag any factor they find irrelevant.

**Endpoints:**
- *Primary:* editing's and knockout's mean relevance against random, each a superiority contrast in a linear mixed
  model with random intercepts for raters, items and stimuli (an item with one method's factors). Two-sided tests,
  Holm-adjusted for the two; superior when the adjusted p is below 0.05 and the effect is positive.
- *Secondary:* editing against knockout; the irrelevant-factor rate.

**Size:** the same simulation, three methods counterbalanced, and the first Holm test at 0.025. A method rated 0.6
points above random is detected with power 0.92, 0.97 and 0.99 (12, 16, 20 raters); a 0.4-point difference with
0.56, 0.73 and 0.81.

## 7. Module C: decision support

### Cases
- 48 per rater: held-out MIMIC-IV and eICU stays at the 24-hour landmark, still in the ICU there, the same for all
  raters.
- **Sampling:** a case-cohort sample (stratified, seeded) that enriches ICU deaths to about a quarter of the cases.
  The weights restore the natural mix in the calibration analyses.
- **Outcomes:** from the real stores' labels (A1's):
  - death in the ICU;
  - acute kidney injury of stage 2 or more within 48 h, asked where the patient is not in that state at the
    landmark (39 of the 48 cases).
- **A new vasopressor within 24 h was dropped** (2026-10-10, after the stimuli were drawn, before the pilot and any
  rating). Of its 44 eligible cases, 1 had the event, so none of its secondary metrics could be estimated, and it
  would have cost each rater 44 estimates. The case-cohort sample enriched deaths only, and the outcome's rate at
  the landmark is 4.5%. Its labels stay in the key. AKI had 12 events in its 39 cases.

### Procedure, per case
1. **Unaided:** the case sheet of the first 24 hours; the rater gives each outcome's probability (0-100%).
2. **Aided:** the twin's forecast card is shown:
   - the risks;
   - E2's cumulative incidence curves;
   - the caveats.
   The rater gives final probabilities and whether the card changed the assessment.

**The card's risks** (fixed before any stimulus is built):
- *Rollouts:* the toolkit's standard mode (128 rollouts per case), capped at four times the 168-hour token quantile.
- *ICU death:* shown raw. At these caps F1 found it calibrated (O/E 0.99), and it has no recalibration map.
- *The other outcomes:* shown recalibrated where amendment 11's V2 produced a map that passed. Passing means the
  map keeps AUROC within 0.005 and moves both O/E and the calibration slope closer to 1 on V1's draws. The maps were
  fitted on rolling anchors and are applied at the landmark.
- *Otherwise:* raw, with the paper's O/E shown beside the risk.
- *Applied (V2's result, 2026-10-10):* AKI's map (`aki_2@2880`) passed: AUROC +0.0004, O/E 0.67 to 0.82, slope
  0.79 to 0.90 at 32 rollouts. It was adopted by the toolkit, so AKI is shown recalibrated. The packets apply the
  adopted maps to the stored forecasts as the toolkit does (`forecast.outputs_of`), never to ICU death, and the key
  records the risk each card shows.

The sequence is fixed, unaided before aided, the design of early-stage reader studies. Its anchoring is a stated
limit; a crossover with a washout was weighed and set aside for the raters' time.

### Endpoints
- **Primary:** the Brier score of ICU death, aided minus unaided.
  - Model: a linear mixed model of the per-estimate squared error, with random intercepts for raters and cases.
  - Hypothesis: a two-sided test at α = 0.05.
- **The second-opinion effect, and how it is separated.** Averaging a clinician's estimate with any advice whose
  errors are independent lowers the Brier score, whatever the advice is worth. In the simulation, a twin no better
  than the clinicians (AUC 0.80 for both) still gave a gain of about 0.017. So the primary endpoint shows that the
  card helps, not that the twin is the reason. The key secondary analysis separates the two:
  - *The counterfactual blend:* each estimate's own shift toward the card (its weight of advice) is replayed with
    LightGBM's risk (A1's pooled model) and, for ICU death, SAPS II's probability, on the same estimates.
  - *The twin-specific gain:* the twin's gain minus the counterfactual one, paired, with its interval.
  - *Reading:* this assumes the shift does not depend on whose advice it is. It is reported as an estimate, not a
    test (section 8's sizes).
- **Secondary:**
  - the twin-specific gain above;
  - the same for AKI;
  - AUC aided against unaided, and against the twin, LightGBM and SAPS II alone on the same cases (by a bootstrap
    over raters and cases, as an Obuchowski-Rockette analysis would);
  - calibration (slope and intercept, case-cohort weighted);
  - the weight of advice (the shift toward the card's risk, as a share of the distance to it);
  - over-reliance: movement toward the card where the card is wrong (on the wrong side of the outcome);
  - under-reliance: no movement where the card is right;
  - time per case;
  - trust and usability at the module's end: the System Usability Scale, and a short trust-in-automation scale
    (items from Jian et al.'s, fixed after the pilot).

### Size
**The simulation:** each case has a latent risk of ICU death, with the case-cohort's quarter of deaths. On top of it:
- the twin's risk is that risk with noise;
- each rater's unaided risk has more noise and a rater bias;
- the aided risk moves toward the card's by a weight of advice drawn per estimate.

Analysis: a two-way cluster-robust interval (raters and cases); 1,000 studies per cell; 48 cases.

| Scenario | AUC: raters unaided, twin | Gain (Brier) | Power (12 / 16 / 20 raters) |
|---|---|---|---|
| As the paper found (weight of advice 0.2-0.3) | 0.80-0.85, 0.93 | 0.015-0.036 | 0.98-1.00 / 0.98-1.00 / 0.99-1.00 |
| Pessimistic: a weaker twin, little reliance (weight 0.1-0.2) | 0.85, 0.875 | 0.007-0.012 | 0.74-0.77 / 0.77-0.81 / 0.77-0.82 |

**The twin-specific gain** has at most about 0.30 power, even when the comparator's AUC is 0.80 against the twin's
0.93, because raters move only a fifth of the way toward any advice. The study estimates it; it is not sized to show
it.

**A design weighed and set aside:** a real control card, half of each rater's cases shown with LightGBM's risks
(`--c-control-arm`). It had at most about 0.19 power for the contrast, and halved the primary endpoint's estimates.

## 8. Analysis

- **Pre-specification:** the code is fixed before the first rating and run once on the locked data.
  - **The code:**
    - `tools/clinician_study_analysis.py tables`: the tidy tables;
    - `tools/clinician_study_models.R`: the primary mixed models;
    - `tools/clinician_study_analysis.py secondary`: the cross-checks and the secondary endpoints.
    The chain was tested end to end on a simulated study (`fake`; `tests/test_clinician_analysis.py`).
  - **The environment:** R 4.3.3 with lme4 1.1-35.5, lmerTest 3.1-3 and Matrix 1.6-5, built on the approved host
    by `tools/server/setup_r_env.sh`. The responses never leave that host.
  - **Mixed models:** crossed random effects of raters and items (or cases). B's models add the stimulus.
  - **A's pooled accuracy:** the population-averaged probability of a correct call, the mean of the inverse logit
    over the rater and item effects. Its 90% interval comes from a parametric bootstrap (999 draws).
  - **Cross-checks:** A's and C's primary results are cross-checked by a two-way bootstrap that resamples raters
    and items or cases, as the paper's other intervals are computed (2,000 draws).
    - B's contrasts lie within items and within raters. There that bootstrap overstates the residual variance: on a
      simulated study its intervals were about 1.7 times the model's.
    - B's cross-check is therefore the per-item contrast with an item-level t interval, the analysis the sizing
      simulated. The bootstrap is reported beside it as a conservative bound.
- **Operational definitions** (secondary endpoints):
  - *Estimates:* C's estimates are the rater's percent over 100. The card's risk is the value shown, a whole
    percent.
  - *Weight of advice:* (aided - unaided) / (card - unaided), clipped to [0, 1], where the card differs from the
    unaided estimate by at least one point. Otherwise the estimate takes the rater's median weight.
  - *The counterfactual blend:* each estimate's weight is replayed toward the twin's card and toward the
    comparator. The twin-specific gain is the paired Brier difference of the two replays.
  - *Over-reliance:* among estimates where moving toward the card would raise the error, the share that moved at
    least one point toward it.
  - *Under-reliance:* among estimates where moving toward the card would lower the error, the share that moved
    less than one point.
  - *AUCs:* a mean of per-rater AUCs (the Obuchowski-Rockette reader average).
  - *Calibration:* the slope and the calibration-in-the-large, weighted by the case-cohort weights. A case's weight
    is its stratum's population over its sample count (dataset × ICU death).
  - *Questionnaire:* the standard 0-100 SUS score, and the mean of the six trust items with the two distrust items
    reversed.
- **Exports:** per packet, only the most complete returned export counts (final first, then the most answers). A
  file from another build or with a wrong checksum is listed and left out.
- **Primary endpoints:** one per module (A; B1; B2's two Holm-adjusted contrasts; C). The modules test different
  claims, so there is no adjustment across them. Secondary endpoints are descriptive with 95% intervals.
- **Missing data:** items without an answer are left out. A rater with less than 80% of a module answered is left out
  of that module's primary analysis, and included in a sensitivity analysis.
- **Interim analyses:** none. Stimuli are fixed before the first rating, and no result is inspected during
  collection.
- **Reporting:**
  - DECIDE-AI for module C (early-stage clinical evaluation of AI decision support);
  - CHERRIES for the web-based procedure;
  - the paper's materials for the rest (tables as CSV, aggregates only).

## 9. Ethics

The existing approval covers this study, as the study lead confirmed (2026-10-10). Its scope includes:
- the rater population and centers;
- the procedures (two online sessions);
- the viewing of de-identified PhysioNet data by credentialed raters;
- compensation;
- the storage of anonymous responses on the approved host;
- publication.

The centers and raters are authorized, and stay anonymous as the approval requires (section 3). Any later change to
these procedures is checked against the approval before it is made. Patient data are secondary, de-identified
research data under the datasets' agreements; no patient is contacted. Participation is voluntary, and a rater may
withdraw until the data are locked.

## 10. Implementation (the toolkit, on the approved host)

Built (2026-10-10):
- **`toolkit.casesheet`:** the standardized case sheet.
- **`toolkit.rating`:** the self-contained packet. It holds:
  - consent and the credential and confidentiality attestation;
  - the modules in order: practice, A, B2, a session break, B1, C, the questionnaire (the System Usability Scale
    and six trust items adapted from Jian et al.);
  - autosave after every answer;
  - export of progress or final results as JSON with a checksum, and resume from an exported file;
  - module C's two-step page, the card decoded only after the unaided estimates are locked.
- **`tools/clinician_study.py`:** the build, run on the approved 4080 host:

| Step | What it does |
|---|---|
| `select` | Every module's items, seeded (`items.json`, host only) |
| `simulate` | B1's 16 give-arm rollouts per item; B2's editing and knockout on the Twin probe, and its random groups; C's forecasts (128 rollouts, four times A1's 168-hour caps) |
| `packets` | The case sheets, packets R01-R20 and P01-P02 (the pilot), `key.json` and a manifest with each packet's SHA-256 |
| `collect` | Checks each returned file's packet, build and checksum; keeps each packet's most complete export; joins the answers to the key |

**Per packet:**
- each module's order is drawn from the rater's number;
- B1's version is twin or real by (rater + item) mod 2;
- B2's method is editing, knockout or random by (rater + item) mod 3.

**The analysis (fixed 2026-10-10, before the first rating; section 8):**
- `tools/clinician_study_analysis.py`: the tables, the cross-checks and the secondary endpoints;
- `tools/clinician_study_models.R`: the lme4 models;
- `tools/server/setup_r_env.sh`: the pinned R environment.

**GPU:** about 1-2 hours (B1's rollouts, B2's knockout, C's forecasts).

## 11. Timeline

| Weeks | Step |
|---|---|
| 1-2 | Confirm the approval's scope; credential the raters; build the stimuli and the rating server |
| 3 | Pilot; fix the case sheet, the codebook and the sizes; amendment and OSF registration |
| 4-7 | Sessions 1 and 2 at every center |
| 8 | Lock the data, run the pre-specified analysis, add the results to the materials |

## 12. Decisions (the study lead, 2026-10-10)

- **The approval:** it covers the study (section 9).
- **Centers and raters:** authorized, and anonymous in every output (section 3).
- **Module A's margin:** 0.40-0.60.
- **Module C's primary outcome:** ICU death.
- **The forecast card in C:** left to the design. ICU death is raw at four times the caps (calibrated in F1); the
  other outcomes are recalibrated where V2's maps pass, raw with the paper's O/E otherwise (section 7). V2's AKI
  map passed, so AKI is shown recalibrated.
- **C's outcomes:** ICU death and AKI. A new vasopressor was dropped before the pilot, with 1 event in 44 eligible
  cases (section 7).
- **Added by the design:** the counterfactual-blend analysis of the second-opinion effect (section 7). It came from
  the power simulation and needs no extra rater time.
- **Module A's size:** 80 items (section 5).
- **Distribution:** self-contained HTML packets per rater, built on the 4080 host, answers exported as JSON (section
  4).
