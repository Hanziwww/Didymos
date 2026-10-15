# Materials: results of the follow-up experiments (amendments 7-11, 16 and 17)

Experiments F1-F9 of `docs/p1_paper_protocol.md` amendment 7 (2026-10-09), fixed before they were run; amendment 9's exploratory diagnosis of attention (2026-10-10, after F4); amendment 10's comparators (2026-10-10: clinical scores, linear, tree and other learners, fixed before any was computed); and amendment 11's validation of the application toolkit (2026-10-10, fixed before it ran). The pre-registered results are in `docs/paper/results.md` and unchanged. Every table is computed from the aggregate JSON files under `runs/paper/` and `runs/toolkit/` on the host and rendered by `tools/paper_results_followup.py`. Held-out split throughout, except amendment 16a, where every stay of the two external databases is a test stay. Amendment 16 (2026-10-11, the study lead's request; fixed before it ran): 16a transfer to RICD and zigongICU, 16b the 48-hour landmark with the follow-up's settings, 16c attention knockout against the generator. Amendment 17 (2026-10-12, the study lead's request; fixed before any adapted model was trained): adapting Didymos to RICD and zigongICU on a locked patient split.

## F1. Rollout count to 128 (accuracy, inference only)

Anchors: R128 1050 (150 per dataset of test_R, inclusion weights kept); L24x 700 (346 in-hospital deaths and 354 controls weighted 8.91: a case-cohort sample). Left out as continued context windows: 1 (R), 0 (L24).

### R128 (`runs/paper/f1/R128`)

1050 anchors x 128 rollouts; 36.4% end at the token cap.

| Endpoint | Anchors | Positives | Twin, 128 rollouts | Twin, A1's 32 | LightGBM | TabPFN-3.5 | Twin probe | 128 - A1's 32 (95% CI) | 128 - LightGBM | 128 - TabPFN-3.5 | O/E (128) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 1045 | 221 | 0.917 | 0.903 | 0.956 | 0.955 | 0.954 | 0.015 (-0.011 to 0.038) | -0.038 (-0.062 to -0.018) | -0.038 (-0.062 to -0.016) | 1.74 |
| New IMV, 24 h | 742 | 77 | 0.855 | 0.837 | 0.805 | 0.797 | 0.788 | 0.018 (-0.022 to 0.066) | 0.050 (-0.027 to 0.145) | 0.057 (0.005 to 0.118) | 0.77 |
| New vasopressor, 24 h | 802 | 107 | 0.906 | 0.889 | 0.846 | 0.842 | 0.738 | 0.017 (-0.001 to 0.045) | 0.060 (0.021 to 0.098) | 0.064 (0.026 to 0.103) | 0.57 |
| New RRT, 48 h | 976 | 94 | 0.951 | 0.931 | 0.934 | 0.933 | 0.929 | 0.019 (0.002 to 0.040) | 0.018 (-0.005 to 0.043) | 0.019 (-0.012 to 0.051) | 0.56 |
| AKI stage >= 2, 48 h | 884 | 188 | 0.888 | 0.887 | 0.835 | 0.835 | 0.815 | 0.001 (-0.014 to 0.015) | 0.053 (0.022 to 0.085) | 0.053 (0.018 to 0.089) | 0.65 |
| Sepsis (ASE), 24 h | 1045 | 8 | 0.966 | 0.989 | 0.951 | 0.959 | 0.952 | -0.023 (-0.072 to 0.002) | 0.015 (-0.063 to 0.081) | 0.007 (-0.011 to 0.024) | 0.92 |
| Hypotension, 4 h | 914 | 70 | 0.874 | 0.875 | 0.900 | 0.894 | 0.861 | -0.001 (-0.019 to 0.022) | -0.025 (-0.064 to 0.005) | -0.020 (-0.057 to 0.007) | 0.96 |
| Hyperkalaemia (top bin), 24 h | 923 | 92 | 0.831 | 0.847 | 0.812 | 0.807 | 0.746 | -0.016 (-0.055 to 0.016) | 0.019 (-0.031 to 0.077) | 0.025 (-0.022 to 0.071) | 0.76 |
| Thrombocytopenia, 48 h | 890 | 42 | 0.932 | 0.829 | 0.918 | 0.910 | 0.865 | 0.103 (0.004 to 0.203) | 0.015 (-0.002 to 0.031) | 0.022 (0.003 to 0.044) | 0.79 |

AUROC against the number of rollouts (E0; the mean over 20 random subsets of each size):

| Endpoint | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 0.658 | 0.710 | 0.770 | 0.807 | 0.850 | 0.881 | 0.902 | 0.917 |
| AKI stage >= 2, 48 h | 0.709 | 0.763 | 0.826 | 0.853 | 0.870 | 0.883 | 0.886 | 0.888 |
| Sepsis (ASE), 24 h | 0.620 | 0.682 | 0.697 | 0.809 | 0.869 | 0.900 | 0.954 | 0.966 |
| New IMV, 24 h | 0.620 | 0.645 | 0.722 | 0.771 | 0.797 | 0.837 | 0.849 | 0.856 |
| New vasopressor, 24 h | 0.611 | 0.682 | 0.766 | 0.824 | 0.855 | 0.887 | 0.897 | 0.906 |
| New RRT, 48 h | 0.644 | 0.727 | 0.803 | 0.871 | 0.919 | 0.934 | 0.944 | 0.947 |
| Hypotension, 4 h | 0.675 | 0.740 | 0.796 | 0.842 | 0.868 | 0.878 | 0.877 | 0.874 |
| Hyperkalaemia (top bin), 24 h | 0.581 | 0.641 | 0.677 | 0.757 | 0.786 | 0.805 | 0.816 | 0.831 |
| Thrombocytopenia, 48 h | 0.677 | 0.724 | 0.775 | 0.830 | 0.863 | 0.886 | 0.923 | 0.937 |

### L24x (`runs/paper/f1/L24x`)

700 anchors x 128 rollouts; 8.5% end at the token cap.

| Endpoint | Anchors | Positives | Twin, 128 rollouts | Twin, A1's 32 | LightGBM | TabPFN-3.5 | Twin probe | 128 - A1's 32 (95% CI) | 128 - LightGBM | 128 - TabPFN-3.5 | O/E (128) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | 500 | 104 | 0.911 | 0.853 | 0.914 | 0.932 | 0.903 | 0.057 (0.022 to 0.096) | -0.003 (-0.030 to 0.021) | -0.022 (-0.047 to 0.001) | 0.99 |
| In-hospital death | 700 | 346 | 0.883 | 0.856 | 0.858 | 0.876 | 0.867 | 0.027 (0.012 to 0.044) | 0.025 (0.007 to 0.043) | 0.006 (-0.010 to 0.023) | 1.10 |
| ICU stay > 7 days | 696 | 148 | 0.864 | 0.843 | 0.850 | 0.864 | 0.852 | 0.021 (-0.021 to 0.062) | 0.013 (-0.016 to 0.044) | -0.000 (-0.035 to 0.031) | 0.97 |

AUROC against the number of rollouts (E0; the mean over 20 random subsets of each size):

| Endpoint | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| ICU death | 0.644 | 0.687 | 0.754 | 0.815 | 0.861 | 0.890 | 0.902 | 0.910 |
| In-hospital death | 0.629 | 0.687 | 0.755 | 0.805 | 0.839 | 0.865 | 0.875 | 0.883 |
| ICU stay > 7 days | 0.644 | 0.717 | 0.760 | 0.813 | 0.839 | 0.851 | 0.862 | 0.867 |

Annotations: the same anchors throughout; A1's 32-rollout Twin is an independent replicate (other seeds) of the same estimator. Weighted metrics (R: the design's inclusion weights; L24x: the case-cohort weights).

## F2. Estimators from the model's step hazards

### R128

| Endpoint | Anchors | Positives | AUROC E0 | E1 | E2 | O/E E0 | E1 | E2 | E2 - E0 (95% CI) | E2 - LightGBM | E2 - TabPFN-3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 1045 | 221 | 0.918 | 0.918 | 0.960 | 1.75 | 1.76 | 1.77 | 0.042 (0.025 to 0.062) | 0.004 (-0.006 to 0.015) | 0.005 (-0.005 to 0.015) |
| New IMV, 24 h | 744 | 77 | 0.870 | 0.870 | 0.886 | 0.55 | 0.55 | 0.55 | 0.016 (0.000 to 0.033) | 0.081 (0.029 to 0.152) | 0.089 (0.026 to 0.189) |
| New vasopressor, 24 h | 802 | 107 | 0.894 | 0.894 | 0.903 | 0.45 | 0.45 | 0.46 | 0.009 (0.002 to 0.019) | 0.057 (0.017 to 0.096) | 0.061 (0.021 to 0.102) |
| New RRT, 48 h | 979 | 96 | 0.948 | 0.949 | 0.959 | 0.53 | 0.58 | 0.56 | 0.011 (0.001 to 0.023) | 0.025 (0.005 to 0.048) | 0.026 (-0.002 to 0.056) |

AUROC against the number of rollouts, E0 and E2 on the same random subsets:

| Endpoint | Estimator | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | E0 (A1's share) | 0.658 | 0.710 | 0.770 | 0.808 | 0.850 | 0.881 | 0.903 | 0.918 |
| Decompensation, 24 h | E2 (hybrid) | 0.908 | 0.920 | 0.930 | 0.937 | 0.945 | 0.951 | 0.955 | 0.960 |
| New IMV, 24 h | E0 (A1's share) | 0.612 | 0.630 | 0.699 | 0.745 | 0.786 | 0.828 | 0.852 | 0.870 |
| New IMV, 24 h | E2 (hybrid) | 0.807 | 0.809 | 0.840 | 0.843 | 0.852 | 0.866 | 0.879 | 0.886 |
| New vasopressor, 24 h | E0 (A1's share) | 0.617 | 0.687 | 0.765 | 0.819 | 0.846 | 0.876 | 0.885 | 0.894 |
| New vasopressor, 24 h | E2 (hybrid) | 0.832 | 0.855 | 0.872 | 0.884 | 0.891 | 0.896 | 0.899 | 0.903 |
| New RRT, 48 h | E0 (A1's share) | 0.645 | 0.729 | 0.802 | 0.874 | 0.919 | 0.934 | 0.945 | 0.948 |
| New RRT, 48 h | E2 (hybrid) | 0.907 | 0.931 | 0.947 | 0.951 | 0.956 | 0.957 | 0.958 | 0.959 |

The self-check (sampled events against summed hazards within the rollouts; 1 expected):

| Endpoint | Sampled events | Summed hazards | Ratio |
|---|---|---|---|
| Decompensation, 24 h | 13436 | 13410.1 | 1.002 |
| New IMV, 24 h | 45242 | 44832.9 | 1.009 |
| New vasopressor, 24 h | 28489 | 28687.2 | 0.993 |
| New RRT, 48 h | 14589 | 14557.0 | 1.002 |

### L24x

| Endpoint | Anchors | Positives | AUROC E0 | E1 | E2 | O/E E0 | E1 | E2 | E2 - E0 (95% CI) | E2 - LightGBM | E2 - TabPFN-3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 698 | 48 | 0.981 | 0.981 | 0.982 | 1.30 | 1.30 | 1.31 | 0.001 (-0.003 to 0.005) | 0.025 (0.010 to 0.044) | - |
| New IMV, 24 h | 465 | 34 | 0.905 | 0.905 | 0.893 | 1.01 | 1.01 | 1.01 | -0.011 (-0.042 to 0.014) | 0.069 (-0.028 to 0.156) | - |
| New vasopressor, 24 h | 510 | 37 | 0.857 | 0.857 | 0.896 | 0.48 | 0.48 | 0.48 | 0.039 (0.002 to 0.103) | 0.130 (0.050 to 0.225) | - |
| New RRT, 48 h | 682 | 39 | 0.959 | 0.959 | 0.964 | 0.85 | 0.85 | 0.89 | 0.005 (-0.001 to 0.013) | 0.032 (0.013 to 0.057) | - |
| ICU death | 500 | 104 | 0.910 | 0.909 | 0.924 | 1.00 | 1.03 | 1.01 | 0.014 (0.005 to 0.025) | 0.010 (-0.013 to 0.031) | -0.008 (-0.028 to 0.010) |
| ICU death, at A1's cap | 500 | 104 | 0.902 | 0.896 | 0.920 | 1.10 | 1.38 | 1.38 | - | - | - |

AUROC against the number of rollouts, E0 and E2 on the same random subsets:

| Endpoint | Estimator | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | E0 (A1's share) | 0.641 | 0.712 | 0.782 | 0.849 | 0.895 | 0.939 | 0.969 | 0.981 |
| Decompensation, 24 h | E2 (hybrid) | 0.918 | 0.927 | 0.943 | 0.954 | 0.960 | 0.969 | 0.977 | 0.982 |
| New IMV, 24 h | E0 (A1's share) | 0.640 | 0.667 | 0.715 | 0.770 | 0.809 | 0.857 | 0.886 | 0.905 |
| New IMV, 24 h | E2 (hybrid) | 0.857 | 0.855 | 0.866 | 0.867 | 0.869 | 0.872 | 0.888 | 0.893 |
| New vasopressor, 24 h | E0 (A1's share) | 0.606 | 0.690 | 0.743 | 0.795 | 0.828 | 0.831 | 0.848 | 0.857 |
| New vasopressor, 24 h | E2 (hybrid) | 0.853 | 0.863 | 0.876 | 0.877 | 0.890 | 0.891 | 0.895 | 0.896 |
| New RRT, 48 h | E0 (A1's share) | 0.647 | 0.715 | 0.804 | 0.878 | 0.916 | 0.943 | 0.956 | 0.959 |
| New RRT, 48 h | E2 (hybrid) | 0.929 | 0.951 | 0.957 | 0.961 | 0.962 | 0.963 | 0.964 | 0.964 |
| ICU death | E0 (A1's share) | 0.655 | 0.709 | 0.770 | 0.815 | 0.853 | 0.884 | 0.902 | 0.910 |
| ICU death | E2 (hybrid) | 0.863 | 0.879 | 0.894 | 0.899 | 0.905 | 0.914 | 0.920 | 0.924 |

The self-check (sampled events against summed hazards within the rollouts; 1 expected):

| Endpoint | Sampled events | Summed hazards | Ratio |
|---|---|---|---|
| Decompensation, 24 h | 11640 | 11679.1 | 0.997 |
| New IMV, 24 h | 33012 | 33248.9 | 0.993 |
| New vasopressor, 24 h | 20970 | 21127.6 | 0.993 |
| New RRT, 48 h | 7022 | 6873.4 | 1.022 |
| ICU death | 11640 | 11679.1 | 0.997 |

### R128: the first (raw) and the corrected (renormalized) hazards (amendment 8)

| Endpoint | E2 AUROC, raw | renormalized | E2 O/E, raw | renormalized | Self-check, raw | renormalized |
|---|---|---|---|---|---|---|
| Decompensation, 24 h | 0.960 | 0.960 | 1.83 | 1.77 | 1.070 | 1.002 |
| New IMV, 24 h | 0.886 | 0.886 | 0.55 | 0.55 | 1.016 | 1.009 |
| New vasopressor, 24 h | 0.901 | 0.903 | 0.45 | 0.46 | 0.995 | 0.993 |
| New RRT, 48 h | 0.960 | 0.959 | 0.52 | 0.56 | 0.969 | 1.002 |

### The hazards against the sampler's exact probabilities (`runs/paper/f1/R128/hazard_check.json`)

| Dataset | Death (raw / renormalized) | Alive discharge | IMV | Vasopressor | RRT |
|---|---|---|---|---|---|
| AmsterdamUMCdb | 1.240 / 1.002 | 1.029 / 1.002 | 1.002 / 1.000 | 1.003 / 1.002 | 1.000 / 1.000 |
| eICU | 0.984 / 0.978 | 1.004 / 1.002 | 1.006 / 0.998 | 1.019 / 1.002 | 0.921 / 1.004 |
| HiRID | 1.050 / 0.984 | 1.017 / 1.004 | 1.005 / 1.001 | 0.994 / 1.002 | 0.907 / 1.000 |
| INSPIRE | 0.991 / 0.975 | 1.009 / 1.001 | 1.003 / 1.001 | 0.942 / 1.002 | 1.000 / 0.998 |
| MIMIC-IV | 1.010 / 1.001 | 1.002 / 0.997 | 1.007 / 0.998 | 1.009 / 0.999 | 1.017 / 1.002 |
| NWICU | 1.016 / 0.999 | 1.020 / 1.003 | 1.008 / 0.996 | 1.009 / 1.001 | 0.867 / 1.006 |
| SICdb | 1.056 / 1.004 | 1.016 / 0.998 | 1.000 / 1.000 | 1.052 / 1.003 | 1.000 / 1.000 |

Ratio of the exact masked probability (grammar and support, as the sampler draws) to the extracted hazard, summed over a 20% sample of the timeline positions of 40 rollouts per dataset (sequences of at most 6,000 tokens); 1 means exact. 'Raw': the first scoring; 'renormalized': amendment 8's.

Annotations: E0 = A1's share of positive rollouts among those whose outcome is known; E1 = Aalen-Johansen over the sampled events (competing causes, the cap censors); E2 = E1 with each rollout's events replaced by the model's hazard increments (Rao-Blackwellization). Treatment events are the first token of the engine's sets, death the ICU discharge 'dead', on the labels' decode clock. 'At A1's cap': the L24x rollouts censored where A1's (four times shorter) cap would have stopped them.

## Amendment 10. More comparators: clinical scores, linear models, trees and other learners (`runs/paper/f10/comparators.json`)

Every comparator on A1's held-out anchors (3,500 per design) and on F1's subsets. The clinical scores come from the canonical CLIF tables (the harmonized raw values), with only data recorded or resulted by the prediction time, from the 24 hours before it. The learned baselines use A1's features and training rows. AUROC weighted as A1; intervals by a bootstrap over stays within datasets (1,000 resamples shared by all predictors, so differences are paired). On the anchors the Twin scored.

(a) Inputs observed in the 24 hours before the 24-hour landmark (share of anchors; a missing component scores 0):

| Input | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|
| Heart rate | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.71 | 0.99 |
| Systolic pressure | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.71 | 0.99 |
| MAP | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.71 | 0.99 |
| Respiratory rate | 1.00 | 0.99 | 1.00 | 1.00 | 0.89 | 0.71 | 0.99 |
| SpO2 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 0.71 | 0.99 |
| Temperature | 0.97 | 0.96 | 0.62 | 1.00 | 0.94 | 0.71 | 0.99 |
| GCS | 1.00 | 0.76 | 0.99 | 0.64 | 0.00 | 0.00 | 0.55 |
| Platelets | 0.98 | 0.85 | 0.93 | 0.98 | 0.97 | 0.92 | 0.99 |
| Bilirubin | 0.46 | 0.41 | 0.38 | 0.60 | 0.36 | 0.56 | 0.95 |
| Creatinine | 0.99 | 0.90 | 0.88 | 0.99 | 0.97 | 0.95 | 1.00 |
| Lactate | 0.62 | 0.26 | 0.90 | 0.77 | 0.99 | 0.30 | 0.55 |
| WBC | 0.99 | 0.86 | 0.93 | 0.99 | 0.97 | 0.92 | 1.00 |
| PaCO2 | 0.50 | 0.39 | 0.82 | 0.99 | 0.99 | 0.00 | 0.86 |
| Urine output | 0.96 | 0.71 | 0.94 | 1.00 | 0.90 | 0.00 | 0.35 |
| PaO2/FiO2 under support | 0.38 | 0.19 | 0.59 | 0.88 | 0.20 | 0.00 | 0.72 |
| Respiratory device | 0.86 | 0.60 | 1.00 | 1.00 | 0.23 | 0.28 | 0.92 |

A check of the windowed SOFA: at the 24-hour landmark its window nearly coincides with the first ICU day, so it should match the pipeline's first-day SOFA (`meds.derived`, the same CLIF tables). INSPIRE's window also holds data from before ICU admission (the operating room):

| Dataset | Anchors | Correlation | Identical | Mean absolute difference | Median, window | Median, first day |
|---|---|---|---|---|---|---|
| AmsterdamUMCdb | 500 | 0.998 | 0.97 | 0.03 | 7.0 | 7.0 |
| eICU | 499 | 0.997 | 0.98 | 0.04 | 3.0 | 3.0 |
| HiRID | 500 | 0.999 | 0.99 | 0.01 | 4.0 | 4.0 |
| INSPIRE | 500 | 0.957 | 0.80 | 0.34 | 4.0 | 3.0 |
| MIMIC-IV | 500 | 0.999 | 0.99 | 0.02 | 4.0 | 4.0 |
| NWICU | 496 | 0.989 | 0.97 | 0.05 | 1.0 | 1.0 |
| SICdb | 500 | 0.992 | 0.96 | 0.07 | 5.0 | 5.0 |

(b) 24-hour landmark, AUROC (positives / anchors: ICU death 104 / 2904; In-hospital death 346 / 3500; ICU stay > 7 days 485 / 3465; Decompensation, 24 h 48 / 3488; New IMV, 24 h 85 / 2746; New vasopressor, 24 h 131 / 2936; New RRT, 48 h 94 / 3412):

| Group | Predictor | ICU death | In-hospital death | ICU stay > 7 days | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h |
|---|---|---|---|---|---|---|---|---|
| Twin | Twin (A1: 32 rollouts, share) | 0.858 | 0.858 | 0.808 | 0.929 | 0.838 | 0.880 | 0.939 |
| Tabular foundation models | TabPFN-3.5 | 0.932 | 0.881 | 0.856 | - | - | - | - |
| Tabular foundation models | TabPFN v2 | 0.914 | 0.869 | 0.839 | - | - | - | - |
| Gradient boosting | LightGBM, pooled | 0.915 | 0.867 | 0.855 | 0.958 | 0.829 | 0.806 | 0.922 |
| Gradient boosting | LightGBM, per dataset | 0.895 | 0.853 | 0.844 | - | - | - | - |
| Other trees | Random forest | 0.908 | 0.855 | 0.841 | 0.942 | 0.811 | 0.804 | 0.932 |
| Other trees | Extra trees | 0.893 | 0.850 | 0.840 | 0.949 | 0.838 | 0.811 | 0.921 |
| Other trees | Decision tree (CART) | 0.805 | 0.780 | 0.777 | 0.815 | 0.618 | 0.673 | 0.863 |
| Linear | Logistic regression, L2 | 0.864 | 0.830 | 0.787 | 0.868 | 0.674 | 0.717 | 0.871 |
| Linear | Logistic regression, L1 (LASSO) | 0.898 | 0.853 | 0.817 | 0.935 | 0.758 | 0.780 | 0.912 |
| Linear | Logistic regression, elastic net | 0.898 | 0.855 | 0.833 | 0.922 | 0.760 | 0.780 | 0.919 |
| Linear | Twin probe (linear, frozen states) | 0.906 | 0.870 | 0.837 | 0.974 | 0.822 | 0.826 | 0.935 |
| Other learners | Naive Bayes | 0.732 | 0.696 | 0.705 | 0.794 | 0.704 | 0.668 | 0.684 |
| Other learners | k-nearest neighbours | 0.831 | 0.794 | 0.810 | 0.866 | 0.727 | 0.765 | 0.844 |
| Other learners | Multilayer perceptron | 0.889 | 0.827 | 0.818 | 0.936 | 0.744 | 0.748 | 0.903 |
| Clinical scores | SOFA | 0.706 | 0.685 | 0.678 | 0.790 | 0.530 | 0.675 | 0.816 |
| Clinical scores | SAPS II | 0.778 | 0.745 | 0.636 | 0.879 | 0.539 | 0.655 | 0.757 |
| Clinical scores | OASIS | 0.730 | 0.701 | 0.639 | 0.841 | 0.584 | 0.608 | 0.680 |
| Clinical scores | NEWS | 0.805 | 0.712 | 0.633 | 0.849 | 0.709 | 0.614 | 0.646 |
| Clinical scores | MEWS | 0.787 | 0.688 | 0.586 | 0.861 | 0.655 | 0.588 | 0.603 |
| Clinical scores | qSOFA | 0.713 | 0.634 | 0.577 | 0.762 | 0.653 | 0.567 | 0.586 |
| Clinical scores | SIRS | 0.710 | 0.656 | 0.590 | 0.722 | 0.615 | 0.540 | 0.593 |
| Bedside indicators | Shock index | - | - | - | 0.721 | - | 0.627 | - |
| Bedside indicators | Lactate | - | - | - | 0.741 | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | - | - | - | 0.675 | - | - |
| Bedside indicators | MAP (lower worse) | - | - | - | - | - | 0.631 | - |
| Bedside indicators | KDIGO stage | - | - | - | - | - | - | 0.814 |
| Bedside indicators | Creatinine | - | - | - | - | - | - | 0.889 |

(b) 24-hour landmark: the Twin minus each predictor, AUROC (95% CI):

| Group | Predictor | ICU death | In-hospital death | ICU stay > 7 days | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h |
|---|---|---|---|---|---|---|---|---|
| Tabular foundation models | TabPFN-3.5 | -0.074 (-0.113 to -0.036) | -0.023 (-0.042 to -0.004) | -0.048 (-0.069 to -0.029) | - | - | - | - |
| Tabular foundation models | TabPFN v2 | -0.056 (-0.096 to -0.018) | -0.011 (-0.031 to 0.009) | -0.031 (-0.051 to -0.011) | - | - | - | - |
| Gradient boosting | LightGBM, pooled | -0.057 (-0.096 to -0.019) | -0.008 (-0.028 to 0.010) | -0.047 (-0.066 to -0.028) | -0.029 (-0.075 to 0.008) | 0.009 (-0.039 to 0.057) | 0.074 (0.045 to 0.103) | 0.016 (-0.008 to 0.042) |
| Gradient boosting | LightGBM, per dataset | -0.037 (-0.078 to 0.003) | 0.006 (-0.013 to 0.026) | -0.036 (-0.058 to -0.015) | - | - | - | - |
| Other trees | Random forest | -0.050 (-0.092 to -0.012) | 0.003 (-0.020 to 0.026) | -0.033 (-0.054 to -0.012) | -0.012 (-0.062 to 0.034) | 0.027 (-0.026 to 0.082) | 0.076 (0.045 to 0.108) | 0.007 (-0.019 to 0.030) |
| Other trees | Extra trees | -0.035 (-0.083 to 0.006) | 0.009 (-0.012 to 0.030) | -0.032 (-0.053 to -0.012) | -0.019 (-0.068 to 0.024) | 0.000 (-0.053 to 0.053) | 0.069 (0.039 to 0.100) | 0.017 (-0.009 to 0.042) |
| Other trees | Decision tree (CART) | 0.053 (-0.001 to 0.110) | 0.078 (0.049 to 0.106) | 0.031 (0.007 to 0.055) | 0.114 (0.033 to 0.194) | 0.221 (0.161 to 0.279) | 0.207 (0.158 to 0.260) | 0.076 (0.038 to 0.117) |
| Linear | Logistic regression, L2 | -0.006 (-0.054 to 0.043) | 0.028 (0.004 to 0.052) | 0.022 (-0.002 to 0.045) | 0.062 (0.008 to 0.120) | 0.164 (0.088 to 0.238) | 0.163 (0.121 to 0.209) | 0.068 (0.027 to 0.109) |
| Linear | Logistic regression, L1 (LASSO) | -0.040 (-0.085 to 0.004) | 0.005 (-0.016 to 0.025) | -0.008 (-0.031 to 0.013) | -0.005 (-0.046 to 0.029) | 0.080 (0.028 to 0.136) | 0.100 (0.068 to 0.133) | 0.027 (0.002 to 0.051) |
| Linear | Logistic regression, elastic net | -0.040 (-0.083 to 0.001) | 0.003 (-0.017 to 0.024) | -0.025 (-0.044 to -0.003) | 0.007 (-0.047 to 0.056) | 0.078 (0.026 to 0.134) | 0.099 (0.066 to 0.135) | 0.020 (0.000 to 0.040) |
| Linear | Twin probe (linear, frozen states) | -0.048 (-0.087 to -0.013) | -0.012 (-0.031 to 0.008) | -0.029 (-0.049 to -0.009) | -0.045 (-0.090 to -0.010) | 0.016 (-0.039 to 0.072) | 0.054 (0.025 to 0.082) | 0.004 (-0.023 to 0.027) |
| Other learners | Naive Bayes | 0.127 (0.070 to 0.184) | 0.162 (0.132 to 0.192) | 0.103 (0.079 to 0.128) | 0.135 (0.070 to 0.204) | 0.134 (0.073 to 0.193) | 0.211 (0.172 to 0.253) | 0.255 (0.214 to 0.298) |
| Other learners | k-nearest neighbours | 0.027 (-0.029 to 0.085) | 0.064 (0.038 to 0.090) | -0.002 (-0.025 to 0.021) | 0.064 (-0.004 to 0.134) | 0.111 (0.049 to 0.183) | 0.115 (0.079 to 0.156) | 0.095 (0.052 to 0.139) |
| Other learners | Multilayer perceptron | -0.031 (-0.076 to 0.010) | 0.032 (0.011 to 0.054) | -0.010 (-0.032 to 0.011) | -0.006 (-0.052 to 0.034) | 0.094 (0.037 to 0.153) | 0.131 (0.094 to 0.169) | 0.036 (0.005 to 0.065) |
| Clinical scores | SOFA | 0.153 (0.091 to 0.212) | 0.173 (0.139 to 0.208) | 0.131 (0.099 to 0.164) | 0.140 (0.060 to 0.229) | 0.309 (0.228 to 0.389) | 0.204 (0.159 to 0.254) | 0.122 (0.084 to 0.161) |
| Clinical scores | SAPS II | 0.080 (0.024 to 0.139) | 0.113 (0.082 to 0.147) | 0.172 (0.140 to 0.204) | 0.050 (-0.010 to 0.110) | 0.299 (0.209 to 0.381) | 0.225 (0.171 to 0.277) | 0.182 (0.130 to 0.239) |
| Clinical scores | OASIS | 0.128 (0.069 to 0.186) | 0.157 (0.122 to 0.195) | 0.170 (0.138 to 0.201) | 0.088 (0.014 to 0.172) | 0.254 (0.178 to 0.326) | 0.272 (0.220 to 0.330) | 0.259 (0.198 to 0.322) |
| Clinical scores | NEWS | 0.053 (-0.003 to 0.108) | 0.147 (0.112 to 0.184) | 0.175 (0.144 to 0.204) | 0.080 (0.021 to 0.139) | 0.129 (0.067 to 0.196) | 0.265 (0.210 to 0.322) | 0.292 (0.233 to 0.359) |
| Clinical scores | MEWS | 0.071 (0.012 to 0.125) | 0.171 (0.132 to 0.211) | 0.222 (0.187 to 0.255) | 0.068 (0.018 to 0.119) | 0.183 (0.111 to 0.261) | 0.292 (0.242 to 0.346) | 0.335 (0.272 to 0.405) |
| Clinical scores | qSOFA | 0.145 (0.082 to 0.202) | 0.224 (0.188 to 0.260) | 0.232 (0.200 to 0.264) | 0.167 (0.102 to 0.234) | 0.186 (0.118 to 0.257) | 0.312 (0.260 to 0.369) | 0.353 (0.292 to 0.418) |
| Clinical scores | SIRS | 0.148 (0.082 to 0.214) | 0.202 (0.167 to 0.237) | 0.218 (0.186 to 0.251) | 0.208 (0.124 to 0.294) | 0.223 (0.149 to 0.293) | 0.340 (0.286 to 0.393) | 0.345 (0.283 to 0.406) |
| Bedside indicators | Shock index | - | - | - | 0.208 (0.122 to 0.309) | - | 0.252 (0.200 to 0.307) | - |
| Bedside indicators | Lactate | - | - | - | 0.189 (0.097 to 0.285) | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | - | - | - | 0.163 (0.106 to 0.222) | - | - |
| Bedside indicators | MAP (lower worse) | - | - | - | - | - | 0.248 (0.193 to 0.300) | - |
| Bedside indicators | KDIGO stage | - | - | - | - | - | - | 0.125 (0.076 to 0.173) |
| Bedside indicators | Creatinine | - | - | - | - | - | - | 0.050 (0.014 to 0.085) |

(c) 48-hour landmark, AUROC (positives / anchors: ICU death 127 / 2643; In-hospital death 455 / 3500; ICU stay > 7 days 686 / 3449):

| Group | Predictor | ICU death | In-hospital death | ICU stay > 7 days |
|---|---|---|---|---|
| Twin | Twin (A1: 32 rollouts, share) | 0.882 | 0.830 | 0.797 |
| Tabular foundation models | TabPFN-3.5 | 0.913 | 0.858 | 0.835 |
| Tabular foundation models | TabPFN v2 | 0.907 | 0.847 | 0.829 |
| Gradient boosting | LightGBM, pooled | 0.903 | 0.859 | 0.834 |
| Gradient boosting | LightGBM, per dataset | 0.873 | 0.831 | 0.810 |
| Other trees | Random forest | 0.892 | 0.837 | 0.821 |
| Other trees | Extra trees | 0.891 | 0.824 | 0.820 |
| Other trees | Decision tree (CART) | 0.795 | 0.757 | 0.791 |
| Linear | Logistic regression, L2 | 0.864 | 0.798 | 0.791 |
| Linear | Logistic regression, L1 (LASSO) | 0.899 | 0.831 | 0.821 |
| Linear | Logistic regression, elastic net | 0.902 | 0.835 | 0.823 |
| Linear | Twin probe (linear, frozen states) | 0.919 | 0.848 | 0.825 |
| Other learners | Naive Bayes | 0.717 | 0.694 | 0.683 |
| Other learners | k-nearest neighbours | 0.844 | 0.792 | 0.782 |
| Other learners | Multilayer perceptron | 0.881 | 0.823 | 0.789 |
| Clinical scores | SOFA | 0.727 | 0.697 | 0.661 |
| Clinical scores | SAPS II | 0.804 | 0.747 | 0.629 |
| Clinical scores | OASIS | 0.773 | 0.702 | 0.659 |
| Clinical scores | NEWS | 0.785 | 0.675 | 0.615 |
| Clinical scores | MEWS | 0.764 | 0.633 | 0.545 |
| Clinical scores | qSOFA | 0.709 | 0.612 | 0.565 |
| Clinical scores | SIRS | 0.663 | 0.609 | 0.553 |

(c) 48-hour landmark: the Twin minus each predictor, AUROC (95% CI):

| Group | Predictor | ICU death | In-hospital death | ICU stay > 7 days |
|---|---|---|---|---|
| Tabular foundation models | TabPFN-3.5 | -0.032 (-0.064 to -0.004) | -0.028 (-0.047 to -0.011) | -0.038 (-0.057 to -0.020) |
| Tabular foundation models | TabPFN v2 | -0.026 (-0.058 to 0.005) | -0.017 (-0.038 to 0.001) | -0.032 (-0.051 to -0.015) |
| Gradient boosting | LightGBM, pooled | -0.021 (-0.053 to 0.010) | -0.029 (-0.049 to -0.010) | -0.036 (-0.055 to -0.019) |
| Gradient boosting | LightGBM, per dataset | 0.008 (-0.031 to 0.044) | -0.002 (-0.023 to 0.017) | -0.013 (-0.033 to 0.006) |
| Other trees | Random forest | -0.011 (-0.045 to 0.023) | -0.007 (-0.029 to 0.014) | -0.023 (-0.043 to -0.005) |
| Other trees | Extra trees | -0.010 (-0.043 to 0.022) | 0.006 (-0.016 to 0.026) | -0.022 (-0.042 to -0.005) |
| Other trees | Decision tree (CART) | 0.087 (0.042 to 0.132) | 0.073 (0.045 to 0.099) | 0.006 (-0.016 to 0.025) |
| Linear | Logistic regression, L2 | 0.018 (-0.016 to 0.050) | 0.031 (0.008 to 0.053) | 0.006 (-0.017 to 0.027) |
| Linear | Logistic regression, L1 (LASSO) | -0.017 (-0.047 to 0.012) | -0.002 (-0.023 to 0.017) | -0.024 (-0.045 to -0.006) |
| Linear | Logistic regression, elastic net | -0.020 (-0.051 to 0.010) | -0.006 (-0.025 to 0.013) | -0.026 (-0.044 to -0.008) |
| Linear | Twin probe (linear, frozen states) | -0.037 (-0.067 to -0.011) | -0.018 (-0.036 to -0.000) | -0.027 (-0.047 to -0.009) |
| Other learners | Naive Bayes | 0.165 (0.115 to 0.220) | 0.136 (0.105 to 0.166) | 0.114 (0.089 to 0.137) |
| Other learners | k-nearest neighbours | 0.037 (-0.006 to 0.081) | 0.038 (0.011 to 0.062) | 0.015 (-0.008 to 0.035) |
| Other learners | Multilayer perceptron | 0.001 (-0.038 to 0.039) | 0.006 (-0.016 to 0.026) | 0.008 (-0.013 to 0.028) |
| Clinical scores | SOFA | 0.155 (0.107 to 0.204) | 0.133 (0.103 to 0.162) | 0.136 (0.111 to 0.160) |
| Clinical scores | SAPS II | 0.078 (0.037 to 0.121) | 0.082 (0.054 to 0.109) | 0.169 (0.142 to 0.196) |
| Clinical scores | OASIS | 0.108 (0.061 to 0.158) | 0.128 (0.094 to 0.157) | 0.139 (0.114 to 0.166) |
| Clinical scores | NEWS | 0.097 (0.056 to 0.139) | 0.155 (0.124 to 0.182) | 0.183 (0.157 to 0.211) |
| Clinical scores | MEWS | 0.118 (0.065 to 0.173) | 0.197 (0.161 to 0.229) | 0.253 (0.223 to 0.285) |
| Clinical scores | qSOFA | 0.173 (0.125 to 0.223) | 0.218 (0.185 to 0.246) | 0.232 (0.206 to 0.261) |
| Clinical scores | SIRS | 0.219 (0.169 to 0.272) | 0.221 (0.190 to 0.253) | 0.245 (0.219 to 0.271) |

(d) Rolling anchors, AUROC (positives / anchors: Decompensation, 24 h 742 / 3486; New IMV, 24 h 236 / 2441; New vasopressor, 24 h 357 / 2730; New RRT, 48 h 296 / 3285; AKI stage >= 2, 48 h 648 / 2922; Sepsis (ASE), 24 h 28 / 3486; Hypotension, 4 h 274 / 3011; Hyperkalaemia (top bin), 24 h 283 / 3065; Thrombocytopenia, 48 h 150 / 3061):

| Group | Predictor | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h | AKI stage >= 2, 48 h | Sepsis (ASE), 24 h | Hypotension, 4 h | Hyperkalaemia (top bin), 24 h | Thrombocytopenia, 48 h |
|---|---|---|---|---|---|---|---|---|---|---|
| Twin | Twin (A1: 32 rollouts, share) | 0.893 | 0.826 | 0.859 | 0.939 | 0.895 | 0.941 | 0.906 | 0.804 | 0.939 |
| Tabular foundation models | TabPFN-3.5 | 0.956 | 0.814 | 0.813 | 0.944 | 0.853 | 0.947 | 0.922 | 0.822 | 0.954 |
| Tabular foundation models | TabPFN v2 | 0.952 | 0.782 | 0.808 | 0.922 | 0.836 | 0.946 | 0.912 | 0.810 | 0.932 |
| Gradient boosting | LightGBM, pooled | 0.955 | 0.817 | 0.803 | 0.932 | 0.858 | 0.933 | 0.925 | 0.826 | 0.953 |
| Gradient boosting | LightGBM, per dataset | 0.938 | 0.822 | 0.802 | 0.919 | 0.857 | 0.774 | 0.895 | 0.808 | 0.947 |
| Other trees | Random forest | 0.948 | 0.801 | 0.810 | 0.921 | 0.842 | 0.933 | 0.916 | 0.818 | 0.940 |
| Other trees | Extra trees | 0.943 | 0.800 | 0.801 | 0.920 | 0.824 | 0.950 | 0.905 | 0.773 | 0.908 |
| Other trees | Decision tree (CART) | 0.852 | 0.692 | 0.737 | 0.869 | 0.811 | 0.901 | 0.885 | 0.770 | 0.925 |
| Linear | Logistic regression, L2 | 0.936 | 0.750 | 0.812 | 0.880 | 0.824 | 0.929 | 0.894 | 0.783 | 0.947 |
| Linear | Logistic regression, L1 (LASSO) | 0.942 | 0.764 | 0.825 | 0.910 | 0.823 | 0.942 | 0.896 | 0.795 | 0.962 |
| Linear | Logistic regression, elastic net | 0.943 | 0.743 | 0.810 | 0.912 | 0.826 | 0.938 | 0.887 | 0.800 | 0.942 |
| Linear | Twin probe (linear, frozen states) | 0.944 | 0.769 | 0.738 | 0.933 | 0.813 | 0.931 | 0.851 | 0.716 | 0.884 |
| Other learners | Naive Bayes | 0.785 | 0.687 | 0.721 | 0.807 | 0.737 | 0.836 | 0.717 | 0.672 | 0.769 |
| Other learners | k-nearest neighbours | 0.874 | 0.676 | 0.721 | 0.821 | 0.797 | 0.837 | 0.829 | 0.734 | 0.788 |
| Other learners | Multilayer perceptron | 0.922 | 0.707 | 0.771 | 0.851 | 0.786 | 0.946 | 0.890 | 0.743 | 0.913 |
| Clinical scores | SOFA | 0.757 | 0.634 | 0.685 | 0.809 | 0.680 | 0.599 | 0.662 | 0.638 | 0.826 |
| Clinical scores | NEWS | 0.849 | 0.712 | 0.706 | 0.617 | 0.639 | 0.676 | 0.671 | 0.595 | 0.621 |
| Clinical scores | MEWS | 0.830 | 0.666 | 0.645 | 0.569 | 0.585 | 0.703 | 0.609 | 0.562 | 0.563 |
| Clinical scores | qSOFA | 0.773 | 0.639 | 0.646 | 0.604 | 0.580 | 0.662 | 0.634 | 0.552 | 0.613 |
| Clinical scores | SIRS | 0.692 | 0.637 | 0.644 | 0.583 | 0.568 | 0.603 | 0.551 | 0.567 | 0.663 |
| Bedside indicators | Shock index | 0.750 | - | 0.642 | - | - | - | - | - | - |
| Bedside indicators | Lactate | 0.701 | - | - | - | - | 0.572 | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | 0.654 | - | - | - | - | - | - | - |
| Bedside indicators | MAP (lower worse) | - | - | 0.633 | - | - | - | 0.863 | - | - |
| Bedside indicators | KDIGO stage | - | - | - | 0.817 | 0.648 | - | - | - | - |
| Bedside indicators | Creatinine | - | - | - | 0.874 | 0.600 | - | - | - | - |
| Bedside indicators | Potassium | - | - | - | - | - | - | - | 0.793 | - |
| Bedside indicators | Platelets (lower worse) | - | - | - | - | - | - | - | - | 0.919 |

(d) Rolling anchors: the Twin minus each predictor, AUROC (95% CI):

| Group | Predictor | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h | AKI stage >= 2, 48 h | Sepsis (ASE), 24 h | Hypotension, 4 h | Hyperkalaemia (top bin), 24 h | Thrombocytopenia, 48 h |
|---|---|---|---|---|---|---|---|---|---|---|
| Tabular foundation models | TabPFN-3.5 | -0.063 (-0.079 to -0.049) | 0.012 (-0.032 to 0.056) | 0.047 (0.017 to 0.078) | -0.005 (-0.021 to 0.011) | 0.042 (0.024 to 0.062) | -0.006 (-0.071 to 0.048) | -0.016 (-0.037 to 0.001) | -0.017 (-0.057 to 0.024) | -0.015 (-0.043 to 0.007) |
| Tabular foundation models | TabPFN v2 | -0.060 (-0.075 to -0.045) | 0.044 (-0.005 to 0.096) | 0.051 (0.023 to 0.080) | 0.016 (0.002 to 0.031) | 0.059 (0.036 to 0.080) | -0.005 (-0.074 to 0.047) | -0.006 (-0.028 to 0.014) | -0.006 (-0.046 to 0.036) | 0.006 (-0.032 to 0.049) |
| Gradient boosting | LightGBM, pooled | -0.062 (-0.079 to -0.048) | 0.009 (-0.043 to 0.065) | 0.056 (0.028 to 0.086) | 0.007 (-0.010 to 0.025) | 0.037 (0.020 to 0.055) | 0.008 (-0.058 to 0.058) | -0.019 (-0.041 to -0.001) | -0.021 (-0.061 to 0.017) | -0.014 (-0.047 to 0.010) |
| Gradient boosting | LightGBM, per dataset | -0.045 (-0.061 to -0.030) | 0.004 (-0.043 to 0.048) | 0.057 (0.029 to 0.083) | 0.020 (0.002 to 0.038) | 0.038 (0.022 to 0.055) | 0.109 (-0.011 to 0.254) | -0.001 (-0.027 to 0.023) | -0.003 (-0.047 to 0.041) | -0.008 (-0.036 to 0.016) |
| Other trees | Random forest | -0.056 (-0.073 to -0.040) | 0.025 (-0.032 to 0.083) | 0.049 (0.020 to 0.079) | 0.018 (-0.002 to 0.041) | 0.053 (0.035 to 0.072) | 0.008 (-0.060 to 0.063) | -0.010 (-0.031 to 0.008) | -0.014 (-0.058 to 0.030) | -0.001 (-0.033 to 0.024) |
| Other trees | Extra trees | -0.050 (-0.068 to -0.034) | 0.026 (-0.030 to 0.077) | 0.058 (0.028 to 0.088) | 0.019 (-0.004 to 0.039) | 0.071 (0.048 to 0.093) | -0.009 (-0.075 to 0.039) | 0.002 (-0.025 to 0.025) | 0.031 (-0.020 to 0.083) | 0.031 (-0.012 to 0.077) |
| Other trees | Decision tree (CART) | 0.041 (0.018 to 0.064) | 0.134 (0.078 to 0.203) | 0.123 (0.084 to 0.166) | 0.070 (0.021 to 0.124) | 0.084 (0.063 to 0.107) | 0.040 (-0.035 to 0.108) | 0.021 (-0.009 to 0.054) | 0.035 (-0.014 to 0.079) | 0.014 (-0.023 to 0.048) |
| Linear | Logistic regression, L2 | -0.044 (-0.061 to -0.026) | 0.075 (0.035 to 0.119) | 0.047 (0.018 to 0.080) | 0.059 (0.038 to 0.081) | 0.071 (0.047 to 0.096) | 0.012 (-0.049 to 0.066) | 0.012 (-0.014 to 0.036) | 0.022 (-0.031 to 0.075) | -0.008 (-0.041 to 0.021) |
| Linear | Logistic regression, L1 (LASSO) | -0.050 (-0.067 to -0.034) | 0.062 (0.015 to 0.116) | 0.034 (0.009 to 0.061) | 0.029 (0.001 to 0.059) | 0.072 (0.048 to 0.096) | -0.001 (-0.072 to 0.055) | 0.011 (-0.016 to 0.034) | 0.010 (-0.040 to 0.060) | -0.024 (-0.058 to 0.005) |
| Linear | Logistic regression, elastic net | -0.050 (-0.067 to -0.034) | 0.083 (0.035 to 0.141) | 0.050 (0.021 to 0.080) | 0.027 (0.014 to 0.041) | 0.069 (0.046 to 0.093) | 0.003 (-0.068 to 0.057) | 0.019 (-0.007 to 0.044) | 0.004 (-0.043 to 0.052) | -0.003 (-0.043 to 0.033) |
| Linear | Twin probe (linear, frozen states) | -0.052 (-0.068 to -0.037) | 0.057 (-0.001 to 0.116) | 0.121 (0.074 to 0.174) | 0.005 (-0.012 to 0.022) | 0.082 (0.059 to 0.106) | 0.010 (-0.046 to 0.064) | 0.055 (0.029 to 0.082) | 0.088 (0.037 to 0.140) | 0.055 (0.009 to 0.107) |
| Other learners | Naive Bayes | 0.107 (0.082 to 0.134) | 0.138 (0.078 to 0.198) | 0.138 (0.098 to 0.189) | 0.132 (0.097 to 0.164) | 0.158 (0.129 to 0.187) | 0.105 (0.021 to 0.211) | 0.190 (0.137 to 0.239) | 0.133 (0.065 to 0.201) | 0.169 (0.114 to 0.225) |
| Other learners | k-nearest neighbours | 0.019 (-0.002 to 0.041) | 0.150 (0.069 to 0.228) | 0.139 (0.091 to 0.192) | 0.118 (0.071 to 0.169) | 0.098 (0.072 to 0.124) | 0.104 (0.026 to 0.191) | 0.077 (0.040 to 0.113) | 0.071 (0.017 to 0.132) | 0.151 (0.068 to 0.244) |
| Other learners | Multilayer perceptron | -0.029 (-0.046 to -0.012) | 0.119 (0.048 to 0.197) | 0.089 (0.056 to 0.124) | 0.088 (0.048 to 0.136) | 0.109 (0.083 to 0.137) | -0.005 (-0.077 to 0.046) | 0.016 (-0.010 to 0.038) | 0.061 (0.005 to 0.115) | 0.026 (-0.011 to 0.068) |
| Clinical scores | SOFA | 0.136 (0.106 to 0.167) | 0.192 (0.109 to 0.289) | 0.174 (0.125 to 0.226) | 0.130 (0.079 to 0.174) | 0.214 (0.175 to 0.249) | 0.342 (0.164 to 0.483) | 0.244 (0.193 to 0.298) | 0.166 (0.103 to 0.233) | 0.113 (0.061 to 0.172) |
| Clinical scores | NEWS | 0.043 (0.018 to 0.066) | 0.114 (0.049 to 0.175) | 0.154 (0.107 to 0.210) | 0.322 (0.250 to 0.393) | 0.256 (0.220 to 0.292) | 0.265 (0.119 to 0.435) | 0.235 (0.180 to 0.291) | 0.210 (0.141 to 0.277) | 0.318 (0.220 to 0.422) |
| Clinical scores | MEWS | 0.063 (0.035 to 0.090) | 0.160 (0.086 to 0.237) | 0.214 (0.161 to 0.281) | 0.370 (0.299 to 0.434) | 0.310 (0.273 to 0.346) | 0.238 (0.117 to 0.381) | 0.297 (0.236 to 0.369) | 0.242 (0.170 to 0.314) | 0.375 (0.265 to 0.481) |
| Clinical scores | qSOFA | 0.120 (0.093 to 0.147) | 0.187 (0.110 to 0.257) | 0.214 (0.160 to 0.268) | 0.335 (0.274 to 0.395) | 0.315 (0.279 to 0.350) | 0.279 (0.135 to 0.424) | 0.273 (0.209 to 0.334) | 0.252 (0.185 to 0.322) | 0.326 (0.233 to 0.411) |
| Clinical scores | SIRS | 0.200 (0.168 to 0.231) | 0.189 (0.110 to 0.274) | 0.215 (0.164 to 0.275) | 0.356 (0.290 to 0.419) | 0.327 (0.286 to 0.361) | 0.338 (0.201 to 0.472) | 0.356 (0.300 to 0.418) | 0.237 (0.177 to 0.297) | 0.276 (0.192 to 0.362) |
| Bedside indicators | Shock index | 0.143 (0.109 to 0.175) | - | 0.218 (0.169 to 0.268) | - | - | - | - | - | - |
| Bedside indicators | Lactate | 0.191 (0.158 to 0.225) | - | - | - | - | 0.369 (0.198 to 0.532) | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | 0.172 (0.099 to 0.262) | - | - | - | - | - | - | - |
| Bedside indicators | MAP (lower worse) | - | - | 0.226 (0.166 to 0.280) | - | - | - | 0.043 (0.005 to 0.079) | - | - |
| Bedside indicators | KDIGO stage | - | - | - | 0.122 (0.062 to 0.188) | 0.247 (0.211 to 0.282) | - | - | - | - |
| Bedside indicators | Creatinine | - | - | - | 0.065 (0.043 to 0.087) | 0.295 (0.256 to 0.334) | - | - | - | - |
| Bedside indicators | Potassium | - | - | - | - | - | - | - | 0.012 (-0.039 to 0.071) | - |
| Bedside indicators | Platelets (lower worse) | - | - | - | - | - | - | - | - | 0.020 (-0.024 to 0.067) |

(e) R128 (the current Twin), AUROC (positives / anchors: Decompensation, 24 h 221 / 1045; New IMV, 24 h 77 / 742; New vasopressor, 24 h 107 / 802; New RRT, 48 h 94 / 976; AKI stage >= 2, 48 h 188 / 884; Sepsis (ASE), 24 h 8 / 1045; Hypotension, 4 h 70 / 914; Hyperkalaemia (top bin), 24 h 92 / 923; Thrombocytopenia, 48 h 42 / 890):

| Group | Predictor | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h | AKI stage >= 2, 48 h | Sepsis (ASE), 24 h | Hypotension, 4 h | Hyperkalaemia (top bin), 24 h | Thrombocytopenia, 48 h |
|---|---|---|---|---|---|---|---|---|---|---|
| Twin | Twin, current estimator (E2, F2) | 0.960 | 0.886 | 0.903 | 0.959 | - | - | - | - | - |
| Twin | Twin, 128 rollouts (share, E0) | 0.917 | 0.855 | 0.906 | 0.951 | 0.888 | 0.966 | 0.874 | 0.831 | 0.932 |
| Tabular foundation models | TabPFN-3.5 | 0.955 | 0.797 | 0.842 | 0.932 | 0.835 | 0.959 | 0.894 | 0.807 | 0.910 |
| Tabular foundation models | TabPFN v2 | 0.957 | 0.757 | 0.847 | 0.912 | 0.818 | 0.953 | 0.884 | 0.796 | 0.922 |
| Gradient boosting | LightGBM, pooled | 0.956 | 0.805 | 0.846 | 0.932 | 0.835 | 0.951 | 0.900 | 0.812 | 0.918 |
| Gradient boosting | LightGBM, per dataset | 0.942 | 0.792 | 0.828 | 0.905 | 0.839 | 0.700 | 0.853 | 0.816 | 0.906 |
| Other trees | Random forest | 0.949 | 0.776 | 0.812 | 0.917 | 0.818 | 0.926 | 0.881 | 0.811 | 0.889 |
| Other trees | Extra trees | 0.947 | 0.795 | 0.807 | 0.882 | 0.809 | 0.947 | 0.907 | 0.767 | 0.881 |
| Other trees | Decision tree (CART) | 0.864 | 0.643 | 0.766 | 0.800 | 0.797 | 0.864 | 0.875 | 0.725 | 0.903 |
| Linear | Logistic regression, L2 | 0.941 | 0.754 | 0.825 | 0.880 | 0.823 | 0.940 | 0.866 | 0.722 | 0.926 |
| Linear | Logistic regression, L1 (LASSO) | 0.943 | 0.782 | 0.840 | 0.875 | 0.816 | 0.946 | 0.864 | 0.738 | 0.942 |
| Linear | Logistic regression, elastic net | 0.942 | 0.766 | 0.830 | 0.901 | 0.815 | 0.940 | 0.848 | 0.745 | 0.942 |
| Linear | Twin probe (linear, frozen states) | 0.954 | 0.788 | 0.738 | 0.928 | 0.816 | 0.952 | 0.861 | 0.746 | 0.865 |
| Other learners | Naive Bayes | 0.774 | 0.692 | 0.724 | 0.835 | 0.748 | 0.863 | 0.745 | 0.658 | 0.800 |
| Other learners | k-nearest neighbours | 0.881 | 0.624 | 0.739 | 0.769 | 0.801 | 0.909 | 0.839 | 0.724 | 0.856 |
| Other learners | Multilayer perceptron | 0.926 | 0.810 | 0.813 | 0.797 | 0.773 | 0.963 | 0.870 | 0.703 | 0.875 |
| Clinical scores | SOFA | 0.747 | 0.681 | 0.661 | 0.807 | 0.653 | 0.523 | 0.696 | 0.642 | 0.785 |
| Clinical scores | NEWS | 0.877 | 0.748 | 0.749 | 0.679 | 0.607 | 0.660 | 0.745 | 0.603 | 0.685 |
| Clinical scores | MEWS | 0.855 | 0.619 | 0.687 | 0.612 | 0.599 | 0.719 | 0.710 | 0.588 | 0.662 |
| Clinical scores | qSOFA | 0.796 | 0.604 | 0.693 | 0.602 | 0.582 | 0.577 | 0.740 | 0.528 | 0.672 |
| Clinical scores | SIRS | 0.713 | 0.639 | 0.656 | 0.600 | 0.536 | 0.677 | 0.546 | 0.580 | 0.686 |
| Bedside indicators | Shock index | 0.756 | - | 0.694 | - | - | - | - | - | - |
| Bedside indicators | Lactate | 0.701 | - | - | - | - | 0.394 | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | 0.645 | - | - | - | - | - | - | - |
| Bedside indicators | MAP (lower worse) | - | - | 0.677 | - | - | - | 0.839 | - | - |
| Bedside indicators | KDIGO stage | - | - | - | 0.821 | 0.620 | - | - | - | - |
| Bedside indicators | Creatinine | - | - | - | 0.883 | 0.590 | - | - | - | - |
| Bedside indicators | Potassium | - | - | - | - | - | - | - | 0.788 | - |
| Bedside indicators | Platelets (lower worse) | - | - | - | - | - | - | - | - | 0.909 |

(e) R128 (the current Twin): the Twin minus each predictor, AUROC (95% CI):

| Group | Predictor | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h | AKI stage >= 2, 48 h | Sepsis (ASE), 24 h | Hypotension, 4 h | Hyperkalaemia (top bin), 24 h | Thrombocytopenia, 48 h |
|---|---|---|---|---|---|---|---|---|---|---|
| Tabular foundation models | TabPFN-3.5 | -0.038 (-0.062 to -0.016) | 0.057 (0.005 to 0.118) | 0.064 (0.026 to 0.103) | 0.019 (-0.012 to 0.051) | 0.053 (0.018 to 0.089) | 0.007 (-0.011 to 0.024) | -0.020 (-0.057 to 0.007) | 0.025 (-0.022 to 0.071) | 0.022 (0.003 to 0.044) |
| Tabular foundation models | TabPFN v2 | -0.040 (-0.064 to -0.018) | 0.098 (0.024 to 0.172) | 0.059 (0.025 to 0.099) | 0.039 (0.009 to 0.069) | 0.070 (0.033 to 0.110) | 0.013 (-0.019 to 0.042) | -0.010 (-0.047 to 0.019) | 0.035 (-0.023 to 0.105) | 0.011 (-0.036 to 0.056) |
| Gradient boosting | LightGBM, pooled | -0.038 (-0.062 to -0.018) | 0.050 (-0.027 to 0.145) | 0.060 (0.021 to 0.098) | 0.018 (-0.005 to 0.043) | 0.053 (0.022 to 0.085) | 0.015 (-0.063 to 0.081) | -0.025 (-0.064 to 0.005) | 0.019 (-0.031 to 0.077) | 0.015 (-0.002 to 0.031) |
| Gradient boosting | LightGBM, per dataset | -0.024 (-0.048 to -0.003) | 0.063 (0.005 to 0.119) | 0.078 (0.041 to 0.117) | 0.046 (0.017 to 0.074) | 0.049 (0.018 to 0.079) | 0.186 (0.047 to 0.343) | 0.010 (-0.034 to 0.047) | 0.015 (-0.038 to 0.077) | 0.027 (0.003 to 0.058) |
| Other trees | Random forest | -0.032 (-0.056 to -0.008) | 0.079 (0.010 to 0.158) | 0.094 (0.051 to 0.139) | 0.034 (0.005 to 0.062) | 0.070 (0.035 to 0.102) | 0.040 (0.019 to 0.067) | -0.007 (-0.028 to 0.011) | 0.021 (-0.024 to 0.066) | 0.044 (0.002 to 0.111) |
| Other trees | Extra trees | -0.029 (-0.053 to -0.006) | 0.060 (-0.032 to 0.158) | 0.099 (0.047 to 0.151) | 0.069 (0.031 to 0.109) | 0.079 (0.042 to 0.116) | 0.019 (-0.016 to 0.059) | -0.032 (-0.080 to 0.010) | 0.064 (-0.004 to 0.140) | 0.052 (0.007 to 0.135) |
| Other trees | Decision tree (CART) | 0.054 (0.015 to 0.091) | 0.212 (0.093 to 0.345) | 0.140 (0.086 to 0.206) | 0.151 (0.055 to 0.255) | 0.091 (0.054 to 0.129) | 0.102 (0.044 to 0.169) | -0.001 (-0.020 to 0.018) | 0.107 (0.033 to 0.184) | 0.029 (-0.053 to 0.099) |
| Linear | Logistic regression, L2 | -0.023 (-0.049 to -0.001) | 0.101 (0.031 to 0.166) | 0.081 (0.034 to 0.128) | 0.071 (0.035 to 0.101) | 0.065 (0.031 to 0.101) | 0.026 (-0.036 to 0.071) | 0.008 (-0.047 to 0.050) | 0.109 (0.018 to 0.220) | 0.007 (-0.019 to 0.030) |
| Linear | Logistic regression, L1 (LASSO) | -0.026 (-0.051 to -0.004) | 0.073 (-0.010 to 0.162) | 0.066 (0.029 to 0.104) | 0.076 (0.013 to 0.151) | 0.072 (0.038 to 0.106) | 0.020 (-0.010 to 0.044) | 0.010 (-0.038 to 0.051) | 0.094 (0.006 to 0.201) | -0.010 (-0.042 to 0.013) |
| Linear | Logistic regression, elastic net | -0.024 (-0.051 to -0.001) | 0.089 (-0.044 to 0.233) | 0.076 (0.036 to 0.117) | 0.049 (0.016 to 0.082) | 0.072 (0.039 to 0.107) | 0.026 (-0.007 to 0.064) | 0.027 (-0.029 to 0.075) | 0.087 (0.004 to 0.194) | -0.010 (-0.044 to 0.016) |
| Linear | Twin probe (linear, frozen states) | -0.037 (-0.060 to -0.016) | 0.067 (-0.066 to 0.190) | 0.168 (0.071 to 0.294) | 0.023 (-0.002 to 0.045) | 0.072 (0.031 to 0.113) | 0.014 (-0.051 to 0.067) | 0.013 (-0.051 to 0.070) | 0.085 (0.023 to 0.155) | 0.068 (0.007 to 0.173) |
| Other learners | Naive Bayes | 0.143 (0.103 to 0.184) | 0.162 (0.090 to 0.248) | 0.182 (0.138 to 0.232) | 0.115 (0.075 to 0.162) | 0.140 (0.092 to 0.186) | 0.103 (0.043 to 0.142) | 0.130 (0.020 to 0.239) | 0.173 (0.082 to 0.271) | 0.132 (0.065 to 0.242) |
| Other learners | k-nearest neighbours | 0.036 (0.006 to 0.071) | 0.231 (0.087 to 0.380) | 0.167 (0.088 to 0.261) | 0.181 (0.097 to 0.281) | 0.087 (0.045 to 0.126) | 0.057 (-0.025 to 0.158) | 0.036 (-0.029 to 0.110) | 0.107 (0.040 to 0.188) | 0.077 (0.041 to 0.135) |
| Other learners | Multilayer perceptron | -0.008 (-0.033 to 0.017) | 0.045 (-0.056 to 0.158) | 0.093 (0.042 to 0.151) | 0.154 (0.061 to 0.234) | 0.114 (0.061 to 0.166) | 0.003 (-0.045 to 0.052) | 0.005 (-0.029 to 0.043) | 0.128 (0.043 to 0.235) | 0.057 (0.024 to 0.095) |
| Clinical scores | SOFA | 0.174 (0.125 to 0.223) | 0.173 (0.055 to 0.312) | 0.246 (0.169 to 0.333) | 0.143 (0.083 to 0.207) | 0.234 (0.160 to 0.311) | 0.443 (0.275 to 0.565) | 0.178 (0.033 to 0.308) | 0.190 (0.102 to 0.284) | 0.147 (0.055 to 0.258) |
| Clinical scores | NEWS | 0.040 (-0.001 to 0.080) | 0.107 (0.019 to 0.218) | 0.157 (0.098 to 0.236) | 0.272 (0.189 to 0.357) | 0.281 (0.200 to 0.364) | 0.306 (0.007 to 0.751) | 0.129 (-0.003 to 0.255) | 0.228 (0.115 to 0.342) | 0.247 (0.064 to 0.467) |
| Clinical scores | MEWS | 0.063 (0.017 to 0.108) | 0.236 (0.110 to 0.400) | 0.219 (0.160 to 0.283) | 0.339 (0.238 to 0.443) | 0.288 (0.215 to 0.358) | 0.247 (0.072 to 0.576) | 0.164 (0.053 to 0.271) | 0.243 (0.119 to 0.378) | 0.270 (0.080 to 0.485) |
| Clinical scores | qSOFA | 0.122 (0.078 to 0.165) | 0.251 (0.131 to 0.387) | 0.214 (0.153 to 0.295) | 0.349 (0.254 to 0.449) | 0.306 (0.230 to 0.383) | 0.389 (0.031 to 0.687) | 0.134 (0.001 to 0.264) | 0.303 (0.188 to 0.412) | 0.261 (0.146 to 0.427) |
| Clinical scores | SIRS | 0.205 (0.152 to 0.260) | 0.216 (0.048 to 0.360) | 0.250 (0.180 to 0.335) | 0.351 (0.278 to 0.427) | 0.352 (0.285 to 0.430) | 0.289 (0.025 to 0.679) | 0.328 (0.199 to 0.446) | 0.251 (0.120 to 0.390) | 0.246 (0.091 to 0.437) |
| Bedside indicators | Shock index | 0.161 (0.104 to 0.221) | - | 0.212 (0.149 to 0.285) | - | - | - | - | - | - |
| Bedside indicators | Lactate | 0.217 (0.169 to 0.272) | - | - | - | - | 0.572 (0.468 to 0.626) | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | 0.210 (0.105 to 0.322) | - | - | - | - | - | - | - |
| Bedside indicators | MAP (lower worse) | - | - | 0.229 (0.134 to 0.335) | - | - | - | 0.036 (-0.075 to 0.123) | - | - |
| Bedside indicators | KDIGO stage | - | - | - | 0.130 (0.080 to 0.195) | 0.268 (0.209 to 0.325) | - | - | - | - |
| Bedside indicators | Creatinine | - | - | - | 0.068 (0.036 to 0.105) | 0.297 (0.219 to 0.371) | - | - | - | - |
| Bedside indicators | Potassium | - | - | - | - | - | - | - | 0.043 (-0.070 to 0.160) | - |
| Bedside indicators | Platelets (lower worse) | - | - | - | - | - | - | - | - | 0.023 (-0.016 to 0.069) |

(f) L24x (the current Twin), AUROC (positives / anchors: ICU death 104 / 500; In-hospital death 346 / 700; ICU stay > 7 days 148 / 696; Decompensation, 24 h 48 / 698; New IMV, 24 h 33 / 464; New vasopressor, 24 h 37 / 510; New RRT, 48 h 39 / 682):

| Group | Predictor | ICU death | In-hospital death | ICU stay > 7 days | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h |
|---|---|---|---|---|---|---|---|---|
| Twin | Twin, current estimator (E2, F2) | 0.924 | - | - | 0.982 | 0.893 | 0.896 | 0.964 |
| Twin | Twin, 128 rollouts (share, E0) | 0.911 | 0.883 | 0.864 | 0.981 | 0.929 | 0.864 | 0.958 |
| Tabular foundation models | TabPFN-3.5 | 0.932 | 0.876 | 0.864 | - | - | - | - |
| Tabular foundation models | TabPFN v2 | 0.911 | 0.867 | 0.834 | - | - | - | - |
| Gradient boosting | LightGBM, pooled | 0.914 | 0.858 | 0.850 | 0.957 | 0.825 | 0.766 | 0.933 |
| Gradient boosting | LightGBM, per dataset | 0.896 | 0.846 | 0.820 | - | - | - | - |
| Other trees | Random forest | 0.914 | 0.857 | 0.844 | 0.943 | 0.786 | 0.733 | 0.955 |
| Other trees | Extra trees | 0.894 | 0.847 | 0.831 | 0.948 | 0.851 | 0.764 | 0.936 |
| Other trees | Decision tree (CART) | 0.799 | 0.775 | 0.751 | 0.815 | 0.650 | 0.542 | 0.877 |
| Linear | Logistic regression, L2 | 0.857 | 0.825 | 0.759 | 0.863 | 0.800 | 0.706 | 0.915 |
| Linear | Logistic regression, L1 (LASSO) | 0.895 | 0.853 | 0.791 | 0.934 | 0.818 | 0.844 | 0.927 |
| Linear | Logistic regression, elastic net | 0.894 | 0.852 | 0.823 | 0.922 | 0.826 | 0.837 | 0.933 |
| Linear | Twin probe (linear, frozen states) | 0.903 | 0.867 | 0.852 | 0.974 | 0.897 | 0.847 | 0.915 |
| Other learners | Naive Bayes | 0.733 | 0.699 | 0.701 | 0.806 | 0.739 | 0.710 | 0.617 |
| Other learners | k-nearest neighbours | 0.830 | 0.791 | 0.812 | 0.866 | 0.654 | 0.708 | 0.889 |
| Other learners | Multilayer perceptron | 0.890 | 0.826 | 0.808 | 0.935 | 0.821 | 0.751 | 0.895 |
| Clinical scores | SOFA | 0.703 | 0.690 | 0.703 | 0.796 | 0.523 | 0.644 | 0.806 |
| Clinical scores | SAPS II | 0.780 | 0.750 | 0.662 | 0.884 | 0.609 | 0.639 | 0.804 |
| Clinical scores | OASIS | 0.731 | 0.709 | 0.691 | 0.845 | 0.702 | 0.562 | 0.732 |
| Clinical scores | NEWS | 0.803 | 0.715 | 0.651 | 0.848 | 0.787 | 0.511 | 0.644 |
| Clinical scores | MEWS | 0.792 | 0.697 | 0.615 | 0.862 | 0.774 | 0.499 | 0.635 |
| Clinical scores | qSOFA | 0.725 | 0.652 | 0.611 | 0.774 | 0.749 | 0.529 | 0.601 |
| Clinical scores | SIRS | 0.711 | 0.658 | 0.586 | 0.723 | 0.619 | 0.546 | 0.665 |
| Bedside indicators | Shock index | - | - | - | 0.716 | - | 0.558 | - |
| Bedside indicators | Lactate | - | - | - | 0.741 | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | - | - | - | 0.754 | - | - |
| Bedside indicators | MAP (lower worse) | - | - | - | - | - | 0.587 | - |
| Bedside indicators | KDIGO stage | - | - | - | - | - | - | 0.779 |
| Bedside indicators | Creatinine | - | - | - | - | - | - | 0.921 |

(f) L24x (the current Twin): the Twin minus each predictor, AUROC (95% CI):

| Group | Predictor | ICU death | In-hospital death | ICU stay > 7 days | Decompensation, 24 h | New IMV, 24 h | New vasopressor, 24 h | New RRT, 48 h |
|---|---|---|---|---|---|---|---|---|
| Tabular foundation models | TabPFN-3.5 | -0.022 (-0.047 to 0.001) | 0.006 (-0.010 to 0.023) | -0.000 (-0.035 to 0.031) | - | - | - | - |
| Tabular foundation models | TabPFN v2 | -0.000 (-0.029 to 0.027) | 0.015 (-0.003 to 0.033) | 0.030 (-0.002 to 0.061) | - | - | - | - |
| Gradient boosting | LightGBM, pooled | -0.003 (-0.030 to 0.021) | 0.025 (0.007 to 0.043) | 0.013 (-0.016 to 0.044) | 0.024 (0.010 to 0.041) | 0.104 (0.031 to 0.188) | 0.098 (0.014 to 0.203) | 0.025 (0.004 to 0.053) |
| Gradient boosting | LightGBM, per dataset | 0.015 (-0.015 to 0.045) | 0.036 (0.019 to 0.056) | 0.044 (0.004 to 0.082) | - | - | - | - |
| Other trees | Random forest | -0.003 (-0.034 to 0.024) | 0.025 (0.004 to 0.047) | 0.019 (-0.014 to 0.050) | 0.038 (0.014 to 0.066) | 0.143 (0.044 to 0.248) | 0.131 (0.022 to 0.249) | 0.003 (-0.013 to 0.019) |
| Other trees | Extra trees | 0.017 (-0.018 to 0.049) | 0.036 (0.015 to 0.059) | 0.033 (-0.001 to 0.071) | 0.032 (0.013 to 0.057) | 0.077 (-0.004 to 0.170) | 0.100 (0.014 to 0.207) | 0.022 (-0.001 to 0.050) |
| Other trees | Decision tree (CART) | 0.111 (0.063 to 0.163) | 0.108 (0.077 to 0.138) | 0.112 (0.061 to 0.171) | 0.166 (0.105 to 0.235) | 0.279 (0.175 to 0.376) | 0.322 (0.148 to 0.486) | 0.081 (0.029 to 0.158) |
| Linear | Logistic regression, L2 | 0.053 (0.013 to 0.095) | 0.058 (0.031 to 0.084) | 0.105 (0.050 to 0.162) | 0.117 (0.072 to 0.170) | 0.129 (0.044 to 0.230) | 0.158 (0.002 to 0.312) | 0.043 (-0.006 to 0.092) |
| Linear | Logistic regression, L1 (LASSO) | 0.016 (-0.016 to 0.049) | 0.029 (0.009 to 0.050) | 0.073 (0.030 to 0.121) | 0.046 (0.023 to 0.074) | 0.111 (0.030 to 0.190) | 0.020 (-0.062 to 0.098) | 0.031 (0.003 to 0.067) |
| Linear | Logistic regression, elastic net | 0.017 (-0.015 to 0.048) | 0.031 (0.011 to 0.052) | 0.041 (0.007 to 0.077) | 0.058 (0.024 to 0.103) | 0.103 (0.031 to 0.178) | 0.027 (-0.051 to 0.107) | 0.025 (-0.006 to 0.067) |
| Linear | Twin probe (linear, frozen states) | 0.008 (-0.023 to 0.035) | 0.015 (-0.002 to 0.035) | 0.012 (-0.032 to 0.047) | 0.006 (-0.003 to 0.018) | 0.032 (-0.044 to 0.129) | 0.017 (-0.100 to 0.108) | 0.043 (-0.001 to 0.106) |
| Other learners | Naive Bayes | 0.177 (0.122 to 0.232) | 0.183 (0.148 to 0.219) | 0.163 (0.104 to 0.222) | 0.175 (0.124 to 0.231) | 0.190 (0.079 to 0.301) | 0.154 (0.003 to 0.321) | 0.341 (0.216 to 0.482) |
| Other learners | k-nearest neighbours | 0.080 (0.033 to 0.124) | 0.092 (0.066 to 0.123) | 0.052 (0.018 to 0.087) | 0.114 (0.065 to 0.176) | 0.275 (0.135 to 0.413) | 0.156 (0.070 to 0.280) | 0.069 (0.033 to 0.108) |
| Other learners | Multilayer perceptron | 0.020 (-0.013 to 0.054) | 0.057 (0.032 to 0.084) | 0.056 (0.011 to 0.101) | 0.046 (0.023 to 0.072) | 0.108 (0.012 to 0.210) | 0.113 (0.051 to 0.180) | 0.063 (0.005 to 0.130) |
| Clinical scores | SOFA | 0.208 (0.145 to 0.268) | 0.192 (0.154 to 0.232) | 0.161 (0.081 to 0.239) | 0.185 (0.119 to 0.255) | 0.406 (0.230 to 0.561) | 0.220 (-0.004 to 0.433) | 0.152 (0.057 to 0.257) |
| Clinical scores | SAPS II | 0.131 (0.074 to 0.188) | 0.132 (0.099 to 0.167) | 0.202 (0.133 to 0.277) | 0.097 (0.052 to 0.150) | 0.320 (0.154 to 0.474) | 0.226 (-0.002 to 0.442) | 0.153 (0.080 to 0.228) |
| Clinical scores | OASIS | 0.180 (0.120 to 0.244) | 0.174 (0.134 to 0.216) | 0.173 (0.104 to 0.243) | 0.135 (0.075 to 0.207) | 0.227 (0.099 to 0.363) | 0.303 (0.096 to 0.492) | 0.225 (0.134 to 0.316) |
| Clinical scores | NEWS | 0.107 (0.050 to 0.163) | 0.167 (0.130 to 0.204) | 0.213 (0.141 to 0.285) | 0.133 (0.084 to 0.192) | 0.142 (0.031 to 0.279) | 0.353 (0.148 to 0.525) | 0.314 (0.161 to 0.455) |
| Clinical scores | MEWS | 0.119 (0.065 to 0.177) | 0.186 (0.149 to 0.224) | 0.249 (0.174 to 0.326) | 0.119 (0.074 to 0.170) | 0.155 (0.036 to 0.300) | 0.365 (0.161 to 0.542) | 0.323 (0.177 to 0.447) |
| Clinical scores | qSOFA | 0.185 (0.126 to 0.248) | 0.230 (0.188 to 0.272) | 0.252 (0.180 to 0.325) | 0.207 (0.146 to 0.275) | 0.180 (0.056 to 0.325) | 0.335 (0.150 to 0.482) | 0.356 (0.215 to 0.494) |
| Clinical scores | SIRS | 0.200 (0.139 to 0.261) | 0.225 (0.181 to 0.263) | 0.278 (0.204 to 0.354) | 0.257 (0.182 to 0.330) | 0.310 (0.177 to 0.452) | 0.318 (0.187 to 0.439) | 0.293 (0.159 to 0.434) |
| Bedside indicators | Shock index | - | - | - | 0.264 (0.179 to 0.365) | - | 0.307 (0.148 to 0.454) | - |
| Bedside indicators | Lactate | - | - | - | 0.239 (0.164 to 0.322) | - | - | - |
| Bedside indicators | ROX index (lower worse) | - | - | - | - | 0.175 (0.061 to 0.286) | - | - |
| Bedside indicators | MAP (lower worse) | - | - | - | - | - | 0.277 (0.129 to 0.426) | - |
| Bedside indicators | KDIGO stage | - | - | - | - | - | - | 0.179 (0.062 to 0.308) |
| Bedside indicators | Creatinine | - | - | - | - | - | - | 0.037 (0.011 to 0.075) |

(g) The datasets' own admission scores, on their datasets' anchors (the Twin and LightGBM on the same anchors; endpoints with fewer than 10 positives left out):

| Design | Endpoint | Score | Anchors | Positives | Score AUROC | Twin | LightGBM | Twin - score (95% CI) | Score O/E |
|---|---|---|---|---|---|---|---|---|---|
| L24 | In-hospital death | APACHE IVa, predicted hospital mortality (eICU) | 377 | 27 | 0.850 | 0.941 | 0.868 | 0.091 (0.032 to 0.163) | 0.57 |
| L24 | In-hospital death | SAPS 3 (SICdb) | 500 | 56 | 0.812 | 0.869 | 0.856 | 0.056 (-0.004 to 0.125) | 0.54 |
| L24 | In-hospital death | ASA class (INSPIRE) | 461 | 23 | 0.712 | 0.744 | 0.826 | 0.033 (-0.115 to 0.188) | - |
| L24 | ICU stay > 7 days | APACHE IVa, predicted ICU stay (eICU) | 401 | 35 | 0.763 | 0.704 | 0.836 | -0.059 (-0.165 to 0.048) | - |
| L24 | ICU stay > 7 days | SAPS 3 (SICdb) | 468 | 86 | 0.730 | 0.799 | 0.821 | 0.069 (-0.007 to 0.137) | 0.85 |
| L24 | ICU stay > 7 days | ASA class (INSPIRE) | 460 | 23 | 0.662 | 0.837 | 0.948 | 0.175 (0.020 to 0.330) | - |
| L48 | ICU death | SAPS 3 (SICdb) | 359 | 13 | 0.810 | 0.912 | 0.900 | 0.102 (-0.002 to 0.231) | 0.18 |
| L48 | In-hospital death | APACHE IVa, predicted hospital mortality (eICU) | 390 | 31 | 0.766 | 0.901 | 0.903 | 0.135 (0.066 to 0.212) | 0.56 |
| L48 | In-hospital death | SAPS 3 (SICdb) | 500 | 69 | 0.815 | 0.882 | 0.876 | 0.067 (0.014 to 0.119) | 0.58 |
| L48 | In-hospital death | ASA class (INSPIRE) | 445 | 33 | 0.602 | 0.818 | 0.834 | 0.215 (0.114 to 0.328) | - |
| L48 | ICU stay > 7 days | APACHE IVa, predicted ICU stay (eICU) | 415 | 50 | 0.764 | 0.806 | 0.847 | 0.042 (-0.051 to 0.141) | - |
| L48 | ICU stay > 7 days | SAPS 3 (SICdb) | 455 | 108 | 0.685 | 0.784 | 0.803 | 0.098 (0.028 to 0.162) | 0.98 |
| L48 | ICU stay > 7 days | ASA class (INSPIRE) | 444 | 60 | 0.573 | 0.799 | 0.900 | 0.225 (0.129 to 0.322) | - |

(i) By dataset: ICU death at the 24-hour landmark (AUROC; positives: MIMIC-IV 26, eICU 11, HiRID 13, AmsterdamUMCdb 27, SICdb 7, NWICU 19, INSPIRE 1):

| Predictor | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|
| Twin (A1: 32 rollouts, share) | 0.843 | 0.939 | 0.877 | 0.875 | 0.786 | 0.761 | 0.998 |
| TabPFN-3.5 | 0.901 | 0.896 | 0.943 | 0.931 | 0.971 | 0.866 | 0.986 |
| LightGBM, pooled | 0.875 | 0.882 | 0.943 | 0.909 | 0.951 | 0.849 | 1.000 |
| Random forest | 0.869 | 0.872 | 0.927 | 0.926 | 0.953 | 0.841 | 0.916 |
| Logistic regression, L2 | 0.847 | 0.838 | 0.915 | 0.816 | 0.842 | 0.862 | 0.745 |
| Logistic regression, L1 (LASSO) | 0.877 | 0.829 | 0.940 | 0.890 | 0.956 | 0.829 | 0.458 |
| Multilayer perceptron | 0.863 | 0.866 | 0.940 | 0.870 | 0.896 | 0.826 | 0.914 |
| SAPS II | 0.766 | 0.819 | 0.917 | 0.857 | 0.672 | 0.679 | 0.550 |
| OASIS | 0.752 | 0.818 | 0.752 | 0.845 | 0.719 | 0.612 | 0.519 |
| SOFA | 0.730 | 0.783 | 0.761 | 0.755 | 0.659 | 0.625 | 0.428 |
| NEWS | 0.779 | 0.920 | 0.773 | 0.865 | 0.889 | 0.670 | 0.500 |
| MEWS | 0.756 | 0.886 | 0.834 | 0.856 | 0.871 | 0.613 | 0.113 |
| qSOFA | 0.660 | 0.774 | 0.779 | 0.756 | 0.767 | 0.604 | 0.348 |

(j) By dataset: death within 24 h, rolling anchors (AUROC; positives: MIMIC-IV 125, eICU 125, HiRID 91, AmsterdamUMCdb 154, SICdb 79, NWICU 117, INSPIRE 51):

| Predictor | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|
| Twin (A1: 32 rollouts, share) | 0.927 | 0.828 | 0.885 | 0.892 | 0.873 | 0.896 | 0.912 |
| TabPFN-3.5 | 0.965 | 0.908 | 0.943 | 0.954 | 0.966 | 0.944 | 0.987 |
| LightGBM, pooled | 0.962 | 0.908 | 0.952 | 0.954 | 0.957 | 0.942 | 0.980 |
| Random forest | 0.953 | 0.889 | 0.948 | 0.945 | 0.963 | 0.936 | 0.977 |
| Logistic regression, L2 | 0.956 | 0.856 | 0.932 | 0.943 | 0.953 | 0.909 | 0.948 |
| Logistic regression, L1 (LASSO) | 0.955 | 0.908 | 0.927 | 0.945 | 0.966 | 0.913 | 0.947 |
| Multilayer perceptron | 0.925 | 0.858 | 0.932 | 0.937 | 0.951 | 0.830 | 0.939 |
| SOFA | 0.719 | 0.759 | 0.704 | 0.810 | 0.835 | 0.826 | 0.673 |
| NEWS | 0.839 | 0.806 | 0.827 | 0.867 | 0.887 | 0.834 | 0.821 |
| MEWS | 0.802 | 0.795 | 0.865 | 0.867 | 0.823 | 0.762 | 0.726 |
| qSOFA | 0.747 | 0.757 | 0.773 | 0.801 | 0.810 | 0.692 | 0.698 |

(h) Calibration of the published mortality equations (pooled over datasets):

| Design | Endpoint | Score | AUROC | O/E | Calibration slope | Brier |
|---|---|---|---|---|---|---|
| L24 | ICU death | SAPS II | 0.778 | 0.24 | 0.76 | 0.058 |
| L24 | ICU death | OASIS | 0.730 | 0.39 | 0.76 | 0.039 |
| L24 | In-hospital death | SAPS II | 0.745 | 0.54 | 0.59 | 0.099 |
| L24 | In-hospital death | OASIS | 0.701 | 0.87 | 0.57 | 0.085 |
| L48 | ICU death | SAPS II | 0.804 | 0.44 | 0.90 | 0.045 |
| L48 | ICU death | OASIS | 0.773 | 0.68 | 0.82 | 0.042 |
| L48 | In-hospital death | SAPS II | 0.747 | 0.90 | 0.62 | 0.102 |
| L48 | In-hospital death | OASIS | 0.702 | 1.38 | 0.54 | 0.106 |

Annotations: clinical scores need no training; a component not charted in the window scores 0, so scores are weaker where inputs are missing (table a; NWICU and SICdb have no GCS, NWICU no urine output or blood gases). SAPS II leaves out its chronic-disease items. Bedside indicators set a missing value to normal. Learned baselines: A1's features of real training stays; settings chosen on the tuning store. E2's per-anchor values were not kept, so its row has no paired difference; the subsets' differences are against the 128-rollout share (E0). Descriptive: no multiplicity adjustment.

## F3. Mortality: where the death deficit comes from

### D1. Step calibration along the real futures (all test_R and test_L24 anchors)

| Design | Endpoint | Observed | Expected (summed hazards) | O/E (95% CI) | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R | Decompensation, 24 h | 96.3 | 98.8 | 0.98 (0.86 to 1.11) | 0.91 | 1.27 | 1.08 | 0.90 | 0.56 | 1.52 | 1.37 |
| R | New IMV, 24 h | 68.8 | 104.1 | 0.66 (0.51 to 0.87) | 0.85 | 0.98 | 1.16 | 0.51 | 1.95 | 0.14 | 0.72 |
| R | New vasopressor, 24 h | 73.2 | 131.3 | 0.56 (0.45 to 0.69) | 0.81 | 0.73 | 0.43 | 0.90 | 0.78 | 0.24 | 0.30 |
| R | New RRT, 48 h | 84.0 | 108.8 | 0.77 (0.65 to 0.92) | 0.63 | 0.84 | 0.99 | 0.67 | 0.68 | 1.07 | 0.13 |
| L24 | Decompensation, 24 h | 48.0 | 63.6 | 0.76 (0.61 to 0.91) | 0.98 | 0.85 | 0.90 | 0.66 | 0.41 | 0.86 | 0.91 |
| L24 | New IMV, 24 h | 85.0 | 133.1 | 0.64 (0.54 to 0.77) | 0.53 | 0.87 | 0.80 | 0.84 | 1.27 | 0.39 | 0.47 |
| L24 | New vasopressor, 24 h | 131.0 | 191.0 | 0.69 (0.58 to 0.79) | 0.74 | 0.53 | 0.49 | 1.29 | 0.93 | 0.48 | 0.48 |
| L24 | New RRT, 48 h | 93.0 | 116.7 | 0.80 (0.65 to 0.96) | 0.99 | 0.75 | 0.93 | 0.73 | 1.16 | 0.55 | 0.24 |
| L24 | ICU death | 104.0 | 109.3 | 0.95 (0.82 to 1.12) | 1.00 | 1.18 | 0.78 | 0.83 | 0.67 | 1.43 | 0.91 |

Annotation: the model's next-token probabilities of each event summed along each anchor's real future (the compensator) against the events observed; O/E near 1 means calibrated step by step. Weighted (R inclusion weights). Real futures that continue a context window are left out (4 R, 5 L24).

Minutes per token (R128): rollouts 3.75, real futures 4.23 (medians).

Minutes per token (L24x): rollouts 3.57, real futures 4.00 (medians).

### D2. Generation drift (the same anchors' rollouts against their real futures)

| Subset | Endpoint | Observed | Real futures' summed hazards | Rollouts' summed hazards (mean) | Drift ratio |
|---|---|---|---|---|---|
| R128 | Decompensation, 24 h | 29.1 | 27.1 | 16.6 | 0.61 |
| R128 | New IMV, 24 h | 23.9 | 35.3 | 43.4 | 1.23 |
| R128 | New vasopressor, 24 h | 22.0 | 35.0 | 48.1 | 1.37 |
| R128 | New RRT, 48 h | 18.5 | 29.5 | 32.9 | 1.12 |
| L24x | Decompensation, 24 h | 48.0 | 64.3 | 36.7 | 0.57 |
| L24x | New IMV, 24 h | 136.8 | 144.9 | 135.8 | 0.94 |
| L24x | New vasopressor, 24 h | 100.3 | 186.2 | 208.6 | 1.12 |
| L24x | New RRT, 48 h | 109.2 | 103.2 | 123.4 | 1.20 |
| L24x | ICU death | 104.0 | 111.6 | 101.4 | 0.91 |

Hourly hazard of the rollouts over the real futures' (paths still at risk), by hours after the anchor:

| Subset | Cause | 0-6 h | 6-12 h | 12-24 h | 24-48 h |
|---|---|---|---|---|---|
| R128 | death | 0.56 | 0.34 | 0.31 | 0.28 |
| R128 | alive discharge | 0.89 | 0.88 | 0.84 | 0.99 |
| L24x | death | 0.52 | 0.42 | 0.34 | 0.49 |
| L24x | alive discharge | 1.08 | 0.99 | 0.93 | 1.05 |

### D3. Capped rollouts by the real outcome (mean share per anchor)

| Subset | Endpoint | Real outcome 0 | Real outcome 1 |
|---|---|---|---|
| R128 | Decompensation, 24 h | 0.343 | 0.405 |
| L24x | In-hospital death | 0.030 | 0.113 |
| L24x | ICU death | 0.018 | 0.053 |
| L24x | Decompensation, 24 h | 0.069 | 0.107 |

### D4. The bank's death gap (C3's twins against real held-out stays)

| Dataset | ICU death, real | ICU death, twins | Twins censored | Twins' ICU death among ended | ICU death, stay 0-24h (real / twins) | ICU death, stay 24-72h (real / twins) | ICU death, stay 72-168h (real / twins) | ICU death, stay 168+ h (real / twins) |
|---|---|---|---|---|---|---|---|---|
| MIMIC-IV | 0.088 | 0.057 | 0.000 | 0.057 | 0.115 / 0.039 | 0.050 / 0.035 | 0.089 / 0.075 | 0.201 / 0.165 |
| eICU | 0.048 | 0.039 | 0.000 | 0.039 | 0.053 / 0.042 | 0.038 / 0.028 | 0.047 / 0.041 | 0.084 / 0.091 |
| HiRID | 0.058 | 0.031 | 0.006 | 0.031 | 0.045 / 0.015 | 0.051 / 0.034 | 0.108 / 0.058 | 0.101 / 0.097 |
| AmsterdamUMCdb | 0.129 | 0.058 | 0.007 | 0.059 | 0.076 / 0.043 | 0.105 / 0.044 | 0.171 / 0.100 | 0.224 / 0.084 |
| SICdb | 0.040 | 0.037 | 0.079 | 0.040 | 0.031 / 0.020 | 0.024 / 0.024 | 0.045 / 0.045 | 0.100 / 0.146 |
| NWICU | 0.067 | 0.061 | 0.000 | 0.061 | 0.071 / 0.052 | 0.037 / 0.025 | 0.073 / 0.074 | 0.162 / 0.184 |
| INSPIRE | 0.012 | 0.007 | 0.002 | 0.007 | 0.007 / 0.011 | 0.002 / 0.002 | 0.016 / 0.000 | 0.112 / 0.019 |

Pre-specified reading: D1 near 1 with a drift ratio below 1 means the deficit is made in generation; D1 near the rollouts' O/E means per-step miscalibration; a gap between the caps means evaluation censoring.

## F4. C2: attributions against the generator's own edit effects (`runs/paper/f4/c2b.json`)

| Endpoint | Anchors | Reliability (split half) | Ceiling (Spearman-Brown) | Probe editing: rho / top-3 | IG | Attention rollout | Attention x grad | Sign agreement, editing | IG |
|---|---|---|---|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | 40 | 0.37 | 0.54 | 0.20 / 0.30 | 0.08 / 0.32 | 0.08 / 0.34 | 0.14 / 0.26 | 0.66 (n=98) | 0.58 (n=95) |
| Decompensation, 24 h | 40 | 0.32 | 0.48 | 0.19 / 0.32 | -0.10 / 0.21 | -0.03 / 0.25 | -0.09 / 0.24 | 0.64 (n=56) | 0.51 (n=55) |
| Hypotension, 4 h | 40 | 0.07 | 0.13 | 0.07 / 0.29 | 0.04 / 0.21 | -0.06 / 0.16 | -0.06 / 0.30 | 0.61 (n=41) | 0.56 (n=39) |
| New IMV, 24 h | 40 | 0.32 | 0.48 | 0.12 / 0.33 | -0.01 / 0.27 | -0.11 / 0.20 | -0.02 / 0.23 | 0.61 (n=61) | 0.38 (n=61) |

Sanity checks (Adebayo et al. 2018): the rank correlation of each method with itself after randomizing the model's weights (probe kept) or the probe's training labels; low means the method depends on what was learned:

| Endpoint | Random weights: editing | IG | Attention rollout | Attn x grad | Permuted labels: editing | IG | Attention rollout | Attn x grad |
|---|---|---|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | -0.03 | -0.06 | 0.97 | 0.61 | -0.05 | 0.04 | 1.00 | 0.95 |
| Decompensation, 24 h | -0.00 | -0.04 | 0.96 | 0.65 | -0.07 | -0.04 | 1.00 | 0.93 |
| Hypotension, 4 h | -0.06 | 0.07 | 0.97 | 0.53 | 0.12 | 0.22 | 1.00 | 0.95 |
| New IMV, 24 h | -0.02 | -0.08 | 0.97 | 0.53 | -0.03 | 0.24 | 1.00 | 0.94 |

Annotations: 40 anchors per endpoint drawn from the upper half of probe risk (valid labels); every event group edited, 48 rollouts per prompt; the generator's risk is the rollouts' mean summed hazard (death, IMV) or positive share (AKI, hypotension) over the horizon after the prompt's end. Agreement is bounded by the reliability ceiling.

## Amendment 9. Why attention rollout fails the randomization check, and remedies (`runs/paper/f4/attn_diag/attn_diag.json`; exploratory)

F4's anchors with prompts of at most 4,096 tokens (152 of 160), the trained model and F4's random initialization (seed 7, the trained probe kept). Medians over anchors of per-anchor Spearman correlations over the event groups (interquartile range in brackets). The structural null is the rollout of uniform causal attention, which involves no model. The rollout and attention x gradient computed here reproduce F4's stored scores (median correlation 1.00 trained, 1.00 random).

(a) Diagnosis:

| Measure | Median (IQR) |
|---|---|
| Rollout, trained against random (group sums: F4's check) | 0.97 (0.95 to 0.98) |
| ... with the groups' token counts held fixed (partial) | 0.96 (0.92 to 0.98) |
| Rollout, trained against random, token level | 0.94 (0.93 to 0.95) |
| Trained rollout against the structural null (group sums) | 0.97 (0.95 to 0.98) |
| Random rollout against the structural null (group sums) | 1.00 (1.00 to 1.00) |
| Total variation from the null, trained / random rollout | 0.19 (0.15 to 0.23) / 0.02 (0.02 to 0.03) |
| Rollout against the groups' token counts, trained / random | 0.67 (0.54 to 0.75) / 0.58 (0.47 to 0.70) |
| Rollout mass on the first token ([BOS], in no group), trained / random | 0.35 (0.27 to 0.44) / 0.38 (0.32 to 0.45) |
| Rollout mass on the statics, trained / random / null | 0.18 (0.16 to 0.19) / 0.24 (0.22 to 0.24) / 0.24 (0.23 to 0.25) |
| Largest over smallest group, in tokens | 84 (38 to 157) |
| Last layer's attention, trained against random: tokens / group sums / group means | 0.03 / 0.78 / 0.01 |
| Last token's attention entropy per head (share of the maximum), range over the 12 layers: trained / random | 0.68-0.81 / 0.92-0.95 |
| Probe editing (the reference) against the token counts: signed / absolute | 0.02 (-0.15 to 0.18) / 0.29 (0.10 to 0.44) |

(b) Methods (pooled over the four endpoints). A method passes if its sanity correlation is at most 0.3 and its agreement with editing is positive (amendment 9):

| Method | Trained against random (sanity; want low) | Sanity, token counts fixed | Agreement with probe editing | Editing, token counts fixed | With absolute editing | With token counts | With the null | Passes |
|---|---|---|---|---|---|---|---|---|
| Attention rollout (C2, F4: group sums) | 0.97 (0.95 to 0.98) | 0.96 | -0.08 (-0.22 to 0.11) | -0.09 | 0.21 | 0.67 | 0.97 | no |
| Attention rollout, group means | 0.92 (0.89 to 0.96) | 0.94 | -0.11 (-0.29 to 0.11) | -0.09 | 0.02 | 0.03 | 0.93 | no |
| Rollout less the structural null, sums | 0.00 (-0.20 to 0.24) | 0.14 | -0.04 (-0.25 to 0.12) | -0.08 | 0.10 | 0.55 | 0.27 | no |
| Rollout less the null, means | 0.08 (-0.10 to 0.30) | 0.07 | -0.08 (-0.28 to 0.08) | -0.08 | -0.10 | -0.18 | 0.11 | no |
| Rollout over the null (lift), means | 0.39 (0.00 to 0.62) | 0.37 | 0.10 (-0.18 to 0.26) | 0.10 | -0.03 | -0.39 | -0.78 | no |
| Last layer's attention, sums | 0.78 (0.71 to 0.86) | 0.06 | 0.07 (-0.09 to 0.26) | 0.15 | 0.31 | 0.80 | 0.30 | no |
| Last layer's attention, means | 0.01 (-0.27 to 0.34) | 0.05 | 0.08 (-0.07 to 0.27) | 0.13 | 0.04 | -0.43 | -0.59 | yes |
| Norm-weighted rollout (Kobayashi et al. 2020), sums | 0.96 (0.94 to 0.98) | 0.95 | -0.08 (-0.22 to 0.10) | -0.10 | 0.22 | 0.67 | 0.96 | no |
| Norm-weighted rollout, means | 0.92 (0.88 to 0.95) | 0.94 | -0.12 (-0.28 to 0.11) | -0.11 | 0.03 | 0.02 | 0.92 | no |
| Attention x gradient (C2, F4: sums) | 0.58 (0.46 to 0.71) | 0.08 | 0.19 (-0.03 to 0.42) | 0.23 | 0.41 | 0.58 | 0.05 | no |
| Attention x gradient, means | -0.03 (-0.25 to 0.14) | -0.00 | 0.14 (-0.04 to 0.33) | 0.20 | 0.09 | -0.60 | -0.55 | yes |
| Gradient-weighted rollout (Chefer et al. 2021), sums | 0.59 (0.47 to 0.71) | 0.05 | 0.20 (-0.03 to 0.40) | 0.25 | 0.41 | 0.60 | 0.06 | no |
| Gradient-weighted rollout, means | -0.04 (-0.27 to 0.15) | -0.03 | 0.14 (-0.02 to 0.33) | 0.22 | 0.10 | -0.59 | -0.54 | yes |
| Attention knockout (Geva et al. 2023) | -0.02 (-0.25 to 0.16) | -0.05 | 0.71 (0.53 to 0.80) | 0.70 | 0.11 | 0.03 | -0.00 | yes |
| The structural null (no model), sums | 1.00 (1.00 to 1.00) | 1.00 | -0.07 (-0.24 to 0.14) | -0.09 | 0.18 | 0.58 | 1.00 | - |

(c) By endpoint (sanity / agreement with editing):

| Endpoint | Anchors | Rollout (C2): sanity | editing | Rollout less null, means: sanity | editing | Gradient-weighted rollout, means: sanity | editing | Attn x grad, means: sanity | editing | Knockout: sanity | editing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | 39 | 0.97 | -0.08 | 0.01 | -0.13 | -0.02 | 0.15 | -0.02 | 0.15 | 0.01 | 0.72 |
| Decompensation, 24 h | 38 | 0.96 | 0.04 | 0.07 | -0.08 | -0.02 | 0.04 | 0.01 | 0.02 | -0.13 | 0.74 |
| Hypotension, 4 h | 37 | 0.97 | -0.20 | 0.01 | 0.01 | -0.10 | 0.25 | -0.07 | 0.25 | 0.00 | 0.69 |
| New IMV, 24 h | 38 | 0.97 | -0.07 | 0.15 | -0.11 | -0.06 | 0.21 | 0.02 | 0.21 | -0.07 | 0.60 |

Annotations: editing is the trained model's probe-level counterfactual (C2, F4's `attrib.jsonl`), signed (> 0: the group raised the risk); unsigned methods are also compared with its absolute value. Knockout blocks attention from every other token to the group's tokens in all 12 layers and reads the probe's logit with the group's positions left out of the mean pooling. Agreement with the generator (F4's ceiling 0.13-0.54) is not re-estimated here.

## F5. C1: validation of the what-if rollouts (`runs/paper/f5/c1b.json`)

### (a) The null of C1's read-outs (two withhold arms, seeds 7 and 8)

| Treatment | Response | Prefixes | Share of 'expected sign' (null 0.5) | Mean (null 0) | 95% CI | SD across prefixes |
|---|---|---|---|---|---|---|
| norepinephrine | map | 172 | 0.51 | 0.030 | -0.658 to 0.749 | 4.702 |
| insulin | glucose | 200 | 0.51 | 0.493 | -7.779 to 7.986 | 57.145 |
| crrt | potassium | 200 | 0.55 | -0.047 | -0.105 to 0.014 | 0.425 |
| crrt | creatinine_24h | 200 | 0.50 | -0.004 | -0.148 to 0.128 | 1.048 |
| furosemide | urine_output | 103 | 0.52 | 12.595 | -37.909 to 64.189 | 262.039 |
| imv | fio2_charted | 200 | 0.49 | 0.042 | -0.021 to 0.109 | 0.485 |
| imv | spo2 | 190 | 0.52 | -0.053 | -0.170 to 0.058 | 0.804 |
| fluid_bolus | map | 193 | 0.40 | -0.317 | -0.931 to 0.320 | 4.448 |
| fluid_bolus | urine_output | 185 | 0.46 | -31.263 | -70.923 to 6.955 | 262.907 |

### (b) The factual design: matching

| Treatment | Eligible starters | Matched pairs | Mean logit, starters | Mean logit, non-starters |
|---|---|---|---|---|
| norepinephrine | 1268 | 189 | -4.11 | -4.12 |
| insulin | 2084 | 187 | -0.90 | -0.90 |
| furosemide | 2065 | 195 | -3.16 | -3.16 |
| crrt | 285 | 180 | -6.78 | -6.80 |
| imv | 1339 | 158 | -2.61 | -2.65 |
| fluid_bolus | 1224 | 194 | -2.39 | -2.40 |

| Treatment | Response | Expected | Pairs | Observational contrast (real starters - matched) | Twin's version (starters' give - matches' withhold) | Twin give - withhold, starters | Share expected sign | Twin give - withhold, non-starters |
|---|---|---|---|---|---|---|---|---|
| norepinephrine | map | + | 160 | -3.31 (-6.53 to -0.20) | -3.58 (-5.85 to -1.07) | 0.79 (0.11 to 1.47) | 0.55 | -0.39 (-1.11 to 0.32) |
| insulin | glucose | - | 166 | 38.00 (6.36 to 72.23) | -0.32 (-14.45 to 12.26) | -22.55 (-32.76 to -13.38) | 0.70 | -22.68 (-31.00 to -15.44) |
| crrt | potassium | - | 101 | 0.24 (-0.11 to 0.59) | 0.44 (0.30 to 0.61) | -0.00 (-0.08 to 0.08) | 0.54 | 0.00 (-0.05 to 0.05) |
| crrt | creatinine_24h | - | 135 | 0.50 (-0.17 to 1.13) | 1.04 (0.75 to 1.34) | -0.28 (-0.43 to -0.14) | 0.60 | 0.01 (-0.12 to 0.15) |
| furosemide | urine_output | + | 80 | 337.05 (248.38 to 421.11) | 403.27 (303.95 to 510.87) | 439.29 (376.90 to 505.53) | 0.97 | 506.41 (453.50 to 567.51) |
| imv | fio2_charted | + | 158 | 2.82 (2.40 to 3.26) | 3.11 (2.75 to 3.51) | 2.39 (2.07 to 2.71) | 0.92 | 2.10 (1.71 to 2.49) |
| imv | spo2 | + | 127 | 1.08 (0.32 to 1.89) | 1.22 (0.74 to 1.67) | 0.49 (0.29 to 0.68) | 0.64 | 0.52 (0.37 to 0.69) |
| fluid_bolus | map | + | 182 | -5.84 (-9.55 to -2.54) | -4.30 (-6.87 to -1.88) | -0.79 (-1.40 to -0.23) | 0.43 | 0.01 (-0.54 to 0.56) |
| fluid_bolus | urine_output | + | 133 | -8.16 (-138.69 to 137.29) | -108.44 (-219.78 to -1.00) | -11.09 (-111.89 to 63.54) | 0.59 | 7.61 (-23.39 to 38.87) |

The factual check (each factual arm's 16 rollouts against the real response):

| Treatment | Response | Group | Anchors | 80% interval coverage | CRPS | Persistence error | PIT KS |
|---|---|---|---|---|---|---|---|
| norepinephrine | map | starter | 165 | 0.69 | 5.87 | 12.14 | 0.08 |
| norepinephrine | map | nonstarter | 160 | 0.69 | 5.74 | 10.20 | 0.09 |
| insulin | glucose | starter | 183 | 0.66 | 62.55 | 117.45 | 0.10 |
| insulin | glucose | nonstarter | 170 | 0.72 | 36.61 | 80.08 | 0.10 |
| crrt | potassium | starter | 125 | 0.73 | 0.46 | 0.76 | 0.08 |
| crrt | potassium | nonstarter | 132 | 0.79 | 0.31 | 0.62 | 0.07 |
| crrt | creatinine_24h | starter | 154 | 0.66 | 0.85 | 1.68 | 0.08 |
| crrt | creatinine_24h | nonstarter | 148 | 0.68 | 0.64 | 1.14 | 0.15 |
| furosemide | urine_output | starter | 86 | 0.71 | 213.36 | 450.00 | 0.08 |
| furosemide | urine_output | nonstarter | 89 | 0.78 | 104.36 | 293.71 | 0.11 |
| imv | fio2_charted | starter | 158 | 0.84 | 0.76 | 2.78 | 0.14 |
| imv | fio2_charted | nonstarter | 158 | 0.91 | 0.43 | 0.94 | 0.29 |
| imv | spo2 | starter | 142 | 0.71 | 1.09 | 2.22 | 0.10 |
| imv | spo2 | nonstarter | 131 | 0.73 | 1.35 | 2.63 | 0.07 |
| fluid_bolus | map | starter | 187 | 0.71 | 6.05 | 10.81 | 0.12 |
| fluid_bolus | map | nonstarter | 183 | 0.70 | 6.23 | 10.66 | 0.07 |
| fluid_bolus | urine_output | starter | 163 | 0.75 | 197.11 | 349.90 | 0.12 |
| fluid_bolus | urine_output | nonstarter | 152 | 0.72 | 168.55 | 276.12 | 0.10 |

Annotations: starters are cut 1 minute before their first start; non-starters are matched within dataset at the same ICU hour (or within 3-6 h) on the propensity of starting within 6 h (LightGBM on A1's features), caliper 0.2 SD of the logit. Contrasts carry the data's confounding by indication; the comparison is whether the twin reproduces the observational contrast, not whether either is causal.

## F6. C3: hospitals and intervals (`runs/paper/c3/c3b.json`)

| Hospital-level rate | Hospitals (twins) | Spearman twins vs real (95% CI) | Hospitals (control) | Spearman control vs real |
|---|---|---|---|---|
| icu_death | 17 | 0.45 (-0.04 to 0.82) | 15 | 0.61 (0.13 to 0.86) |
| ever_imv | 17 | 0.63 (0.15 to 0.93) | 15 | 0.52 (-0.08 to 0.94) |
| ever_vasopressor | 17 | 0.66 (0.15 to 0.90) | 15 | 0.85 (0.54 to 0.99) |
| los_median | 17 | 0.72 (0.29 to 0.92) | 15 | 0.71 (0.21 to 0.94) |

Inter-dataset structure: twins 0.73 (0.61 to 0.83), real control 0.95 (0.84 to 0.96).

Annotations: eICU hospitals with at least 30 held-out stays and 30 twins (control: 30 train stays); the real control's correlation is the ceiling sampling noise allows.

## F7. C4: neighbours as a predictor (`runs/paper/c4/c4b_<design>.json`)

### L24

| Endpoint | Neighbours | Positives | state (k=10) | random_init_state (k=10) | tfidf (k=10) | features (k=10) | state k=5 | state k=25 | state k=50 |
|---|---|---|---|---|---|---|---|---|---|
| ICU death | all | 104 | 0.723 | 0.588 | 0.685 | 0.720 | 0.695 | 0.764 | 0.808 |
| ICU death | within dataset | 104 | 0.723 | 0.588 | 0.685 | 0.735 | 0.695 | 0.764 | 0.808 |
| In-hospital death | all | 346 | 0.761 | 0.625 | 0.749 | 0.744 | 0.733 | 0.810 | 0.823 |
| In-hospital death | within dataset | 346 | 0.761 | 0.626 | 0.746 | 0.743 | 0.733 | 0.810 | 0.822 |
| ICU stay > 7 days | all | 485 | 0.780 | 0.620 | 0.781 | 0.755 | 0.749 | 0.795 | 0.803 |
| ICU stay > 7 days | within dataset | 485 | 0.780 | 0.620 | 0.783 | 0.760 | 0.749 | 0.796 | 0.803 |

### R

| Endpoint | Neighbours | Positives | state (k=10) | random_init_state (k=10) | tfidf (k=10) | features (k=10) | state k=5 | state k=25 | state k=50 |
|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | all | 742 | 0.861 | 0.628 | 0.845 | 0.841 | 0.822 | 0.885 | 0.891 |
| Decompensation, 24 h | within dataset | 742 | 0.861 | 0.631 | 0.844 | 0.844 | 0.822 | 0.885 | 0.890 |
| New IMV, 24 h | all | 236 | 0.640 | 0.500 | 0.668 | 0.638 | 0.624 | 0.684 | 0.686 |
| New IMV, 24 h | within dataset | 236 | 0.640 | 0.489 | 0.667 | 0.630 | 0.625 | 0.682 | 0.685 |
| New vasopressor, 24 h | all | 357 | 0.651 | 0.558 | 0.643 | 0.661 | 0.655 | 0.721 | 0.714 |
| New vasopressor, 24 h | within dataset | 357 | 0.651 | 0.557 | 0.644 | 0.649 | 0.656 | 0.721 | 0.715 |
| New RRT, 48 h | all | 296 | 0.845 | 0.666 | 0.776 | 0.760 | 0.755 | 0.865 | 0.874 |
| New RRT, 48 h | within dataset | 296 | 0.846 | 0.668 | 0.774 | 0.743 | 0.755 | 0.865 | 0.874 |
| AKI stage >= 2, 48 h | all | 648 | 0.722 | 0.665 | 0.674 | 0.721 | 0.689 | 0.739 | 0.752 |
| AKI stage >= 2, 48 h | within dataset | 648 | 0.722 | 0.662 | 0.675 | 0.706 | 0.689 | 0.739 | 0.752 |
| Sepsis (ASE), 24 h | all | 28 | 0.592 | 0.511 | 0.495 | 0.655 | 0.543 | 0.744 | 0.852 |
| Sepsis (ASE), 24 h | within dataset | 28 | 0.592 | 0.512 | 0.495 | 0.662 | 0.543 | 0.744 | 0.852 |
| Hypotension, 4 h | all | 274 | 0.707 | 0.664 | 0.711 | 0.692 | 0.643 | 0.746 | 0.760 |
| Hypotension, 4 h | within dataset | 274 | 0.707 | 0.665 | 0.710 | 0.702 | 0.642 | 0.745 | 0.761 |
| Hyperkalaemia (top bin), 24 h | all | 283 | 0.633 | 0.565 | 0.647 | 0.571 | 0.642 | 0.704 | 0.695 |
| Hyperkalaemia (top bin), 24 h | within dataset | 283 | 0.633 | 0.558 | 0.645 | 0.569 | 0.642 | 0.704 | 0.694 |
| Thrombocytopenia, 48 h | all | 150 | 0.750 | 0.603 | 0.702 | 0.713 | 0.703 | 0.804 | 0.814 |
| Thrombocytopenia, 48 h | within dataset | 150 | 0.749 | 0.604 | 0.702 | 0.715 | 0.703 | 0.804 | 0.814 |

Annotations: leave-one-out kNN risk (the weighted mean label of the k most similar anchors of other stays), its AUROC; 'random init state' is the same architecture with random weights (a negative control).

## F8. Augmentation with repetitions (`runs/paper/f8/aug_<design>.json`)

### L24: summary over datasets and endpoints (cells with at least 20 test positives)

| Real stays n | Twins added | Cells | Mean gain | Median gain | Cells with a gain | Cells whose test-anchor interval excludes 0 |
|---|---|---|---|---|---|---|
| 100 | 1 x n | 19 | 0.038 | 0.036 | 19/19 | 2/19 |
| 100 | 3 x n | 19 | 0.059 | 0.048 | 19/19 | 3/19 |
| 300 | 1 x n | 19 | 0.012 | 0.007 | 19/19 | 0/19 |
| 300 | 3 x n | 8 | 0.028 | 0.027 | 8/8 | 2/8 |
| 1000 | 1 x n | 8 | 0.007 | 0.006 | 6/8 | 0/8 |

### L24

| Endpoint | Dataset | Test positives | Real stays n | AUROC, n real | + n twins: difference (over 20 draws; over test anchors) | + 3n twins | Real equivalent of n + n twins | of n + 3n twins | 1,000 twins alone |
|---|---|---|---|---|---|---|---|---|---|
| ICU death | AmsterdamUMCdb | 82 | 100 | 0.792 | 0.009 (-0.030 to 0.050; -0.009 to 0.020) | 0.025 (-0.019 to 0.068; -0.012 to 0.026) | 137 | 219 | - |
| ICU death | AmsterdamUMCdb | 82 | 300 | 0.820 | 0.007 (-0.018 to 0.030; -0.011 to 0.016) | - | 443 | - | - |
| ICU death | eICU | 19 | 100 | 0.727 | 0.051 (-0.050 to 0.133; -0.011 to 0.051) | 0.032 (-0.065 to 0.137; -0.015 to 0.043) | 288 | 241 | 0.801 |
| ICU death | eICU | 19 | 300 | 0.782 | 0.020 (-0.071 to 0.104; -0.008 to 0.042) | 0.028 (-0.046 to 0.130; -0.033 to 0.037) | 884 | 1,240 | 0.801 |
| ICU death | eICU | 19 | 1000 | 0.810 | 0.022 (-0.029 to 0.066; 0.001 to 0.053) | - | >3000 | - | 0.801 |
| ICU death | HiRID | 29 | 100 | 0.817 | 0.025 (-0.041 to 0.090; -0.005 to 0.027) | 0.047 (-0.011 to 0.166; -0.008 to 0.027) | 170 | 379 | - |
| ICU death | HiRID | 29 | 300 | 0.861 | 0.006 (-0.045 to 0.048; -0.019 to 0.016) | - | 484 | - | - |
| ICU death | INSPIRE | 7 | 100 | 0.764 | -0.040 (-0.158 to 0.181; -0.041 to 0.022) | -0.152 (-0.383 to 0.013; -0.071 to 0.004) | - | - | - |
| ICU death | INSPIRE | 7 | 300 | 0.794 | -0.062 (-0.362 to 0.133; -0.060 to 0.017) | - | - | - | - |
| ICU death | MIMIC-IV | 45 | 100 | 0.703 | 0.068 (-0.024 to 0.208; -0.006 to 0.043) | 0.107 (-0.016 to 0.207; -0.015 to 0.064) | 171 | 339 | 0.807 |
| ICU death | MIMIC-IV | 45 | 300 | 0.808 | 0.007 (-0.046 to 0.054; -0.039 to 0.008) | 0.016 (-0.029 to 0.074; -0.068 to 0.000) | 429 | 587 | 0.807 |
| ICU death | MIMIC-IV | 45 | 1000 | 0.840 | -0.008 (-0.039 to 0.025; -0.034 to 0.010) | - | 790 | - | 0.807 |
| ICU death | NWICU | 31 | 100 | 0.674 | 0.093 (0.001 to 0.221; 0.009 to 0.083) | 0.123 (0.023 to 0.248; -0.001 to 0.094) | 224 | 294 | 0.806 |
| ICU death | NWICU | 31 | 300 | 0.799 | 0.009 (-0.031 to 0.057; -0.042 to 0.015) | 0.024 (-0.012 to 0.104; -0.053 to 0.022) | 407 | 591 | 0.806 |
| ICU death | NWICU | 31 | 1000 | 0.845 | 0.005 (-0.025 to 0.042; -0.039 to 0.010) | - | 1,302 | - | 0.806 |
| ICU death | SICdb | 24 | 100 | 0.828 | 0.008 (-0.053 to 0.058; -0.038 to 0.018) | 0.030 (-0.056 to 0.135; -0.047 to 0.013) | 127 | 211 | - |
| ICU death | SICdb | 24 | 300 | 0.874 | 0.001 (-0.062 to 0.045; -0.040 to 0.016) | - | 603 | - | - |
| In-hospital death | AmsterdamUMCdb | 82 | 100 | 0.792 | 0.005 (-0.029 to 0.039; -0.012 to 0.016) | 0.027 (-0.012 to 0.065; -0.008 to 0.029) | 121 | 268 | - |
| In-hospital death | AmsterdamUMCdb | 82 | 300 | 0.820 | 0.007 (-0.021 to 0.042; -0.013 to 0.015) | - | 427 | - | - |
| In-hospital death | eICU | 39 | 100 | 0.689 | 0.036 (-0.018 to 0.112; -0.016 to 0.034) | 0.041 (-0.031 to 0.104; -0.016 to 0.042) | 261 | 272 | 0.764 |
| In-hospital death | eICU | 39 | 300 | 0.742 | 0.013 (-0.036 to 0.077; -0.004 to 0.025) | 0.041 (-0.029 to 0.123; 0.004 to 0.047) | 505 | 1,856 | 0.764 |
| In-hospital death | eICU | 39 | 1000 | 0.774 | 0.024 (-0.012 to 0.073; -0.001 to 0.049) | - | 2,674 | - | 0.764 |
| In-hospital death | HiRID | 29 | 100 | 0.815 | 0.026 (-0.037 to 0.084; -0.012 to 0.024) | 0.047 (-0.016 to 0.127; -0.019 to 0.027) | 166 | 299 | - |
| In-hospital death | HiRID | 29 | 300 | 0.862 | 0.007 (-0.034 to 0.043; -0.026 to 0.013) | - | 537 | - | - |
| In-hospital death | INSPIRE | 27 | 100 | 0.696 | 0.020 (-0.064 to 0.169; -0.037 to 0.049) | 0.008 (-0.052 to 0.103; -0.114 to 0.025) | 151 | 120 | - |
| In-hospital death | INSPIRE | 27 | 300 | 0.740 | 0.007 (-0.056 to 0.072; -0.055 to 0.037) | - | 379 | - | - |
| In-hospital death | MIMIC-IV | 67 | 100 | 0.704 | 0.050 (-0.023 to 0.125; -0.007 to 0.044) | 0.077 (0.022 to 0.126; -0.019 to 0.060) | 179 | 314 | 0.791 |
| In-hospital death | MIMIC-IV | 67 | 300 | 0.780 | 0.020 (-0.008 to 0.048; -0.018 to 0.014) | 0.022 (-0.011 to 0.055; -0.040 to 0.007) | 480 | 498 | 0.791 |
| In-hospital death | MIMIC-IV | 67 | 1000 | 0.824 | -0.005 (-0.031 to 0.025; -0.038 to -0.003) | - | 828 | - | 0.791 |
| In-hospital death | NWICU | 46 | 100 | 0.695 | 0.050 (-0.042 to 0.178; -0.010 to 0.034) | 0.082 (-0.014 to 0.196; -0.008 to 0.065) | 209 | 275 | 0.791 |
| In-hospital death | NWICU | 46 | 300 | 0.789 | 0.002 (-0.053 to 0.076; -0.044 to 0.003) | 0.022 (-0.019 to 0.097; -0.049 to 0.004) | 343 | 779 | 0.791 |
| In-hospital death | NWICU | 46 | 1000 | 0.824 | 0.007 (-0.038 to 0.036; -0.034 to 0.005) | - | 1,950 | - | 0.791 |
| In-hospital death | SICdb | 56 | 100 | 0.742 | 0.044 (-0.004 to 0.131; -0.028 to 0.004) | 0.054 (-0.031 to 0.126; -0.046 to -0.002) | 244 | 288 | - |
| In-hospital death | SICdb | 56 | 300 | 0.799 | 0.004 (-0.040 to 0.046; -0.033 to 0.007) | - | 363 | - | - |
| ICU stay > 7 days | AmsterdamUMCdb | 142 | 100 | 0.774 | 0.028 (-0.013 to 0.081; -0.010 to 0.009) | 0.025 (-0.014 to 0.081; -0.033 to -0.002) | 235 | 220 | - |
| ICU stay > 7 days | AmsterdamUMCdb | 142 | 300 | 0.815 | 0.004 (-0.017 to 0.031; -0.021 to -0.000) | - | 374 | - | - |
| ICU stay > 7 days | eICU | 44 | 100 | 0.648 | 0.064 (-0.026 to 0.186; 0.017 to 0.075) | 0.095 (-0.045 to 0.280; 0.004 to 0.061) | 242 | 333 | 0.774 |
| ICU stay > 7 days | eICU | 44 | 300 | 0.741 | 0.017 (-0.026 to 0.052; -0.002 to 0.036) | 0.035 (-0.017 to 0.098; 0.001 to 0.049) | 785 | 2,380 | 0.774 |
| ICU stay > 7 days | eICU | 44 | 1000 | 0.763 | 0.016 (-0.024 to 0.073; -0.021 to 0.021) | - | 2,597 | - | 0.774 |
| ICU stay > 7 days | HiRID | 58 | 100 | 0.650 | 0.020 (-0.102 to 0.143; -0.016 to 0.035) | 0.048 (-0.019 to 0.122; -0.003 to 0.051) | 255 | 1,088 | - |
| ICU stay > 7 days | HiRID | 58 | 300 | 0.688 | 0.022 (-0.025 to 0.076; -0.011 to 0.034) | - | 1,828 | - | - |
| ICU stay > 7 days | INSPIRE | 29 | 100 | 0.776 | 0.056 (-0.048 to 0.172; -0.008 to 0.038) | 0.093 (0.021 to 0.193; -0.021 to 0.050) | 173 | 310 | - |
| ICU stay > 7 days | INSPIRE | 29 | 300 | 0.869 | 0.018 (-0.025 to 0.067; -0.024 to 0.007) | - | 610 | - | - |
| ICU stay > 7 days | MIMIC-IV | 61 | 100 | 0.704 | 0.022 (-0.040 to 0.127; -0.006 to 0.024) | 0.047 (-0.026 to 0.145; 0.008 to 0.054) | 171 | 355 | 0.751 |
| ICU stay > 7 days | MIMIC-IV | 61 | 300 | 0.749 | 0.021 (-0.012 to 0.065; -0.004 to 0.026) | 0.030 (-0.024 to 0.073; -0.012 to 0.033) | 659 | 761 | 0.751 |
| ICU stay > 7 days | MIMIC-IV | 61 | 1000 | 0.797 | 0.005 (-0.023 to 0.025; -0.014 to 0.014) | - | 1,414 | - | 0.751 |
| ICU stay > 7 days | NWICU | 67 | 100 | 0.656 | 0.045 (-0.042 to 0.215; -0.007 to 0.056) | 0.070 (0.002 to 0.229; 0.010 to 0.070) | 236 | 301 | 0.746 |
| ICU stay > 7 days | NWICU | 67 | 300 | 0.726 | 0.026 (-0.014 to 0.097; -0.018 to 0.022) | 0.037 (-0.014 to 0.113; -0.032 to 0.027) | 775 | 946 | 0.746 |
| ICU stay > 7 days | NWICU | 67 | 1000 | 0.766 | 0.010 (-0.023 to 0.040; -0.049 to 0.014) | - | 1,415 | - | 0.746 |
| ICU stay > 7 days | SICdb | 86 | 100 | 0.727 | 0.047 (-0.005 to 0.156; -0.018 to 0.026) | 0.067 (0.002 to 0.195; -0.020 to 0.028) | 228 | 401 | - |
| ICU stay > 7 days | SICdb | 86 | 300 | 0.786 | 0.024 (-0.005 to 0.060; -0.003 to 0.030) | - | 716 | - | - |

### R: summary over datasets and endpoints (cells with at least 20 test positives)

| Real stays n | Twins added | Cells | Mean gain | Median gain | Cells with a gain | Cells whose test-anchor interval excludes 0 |
|---|---|---|---|---|---|---|
| 100 | 1 x n | 34 | 0.021 | 0.022 | 25/34 | 4/34 |
| 100 | 3 x n | 34 | 0.033 | 0.037 | 28/34 | 1/34 |
| 300 | 1 x n | 34 | 0.012 | 0.007 | 26/34 | 1/34 |
| 300 | 3 x n | 34 | 0.027 | 0.023 | 28/34 | 3/34 |
| 1000 | 1 x n | 27 | 0.010 | 0.008 | 17/27 | 2/27 |
| 1000 | 3 x n | 5 | 0.054 | 0.045 | 5/5 | 0/5 |

### R

| Endpoint | Dataset | Test positives | Real stays n | AUROC, n real | + n twins: difference (over 20 draws; over test anchors) | + 3n twins | Real equivalent of n + n twins | of n + 3n twins | 1,000 twins alone |
|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | AmsterdamUMCdb | 154 | 100 | 0.844 | -0.011 (-0.089 to 0.071; -0.026 to -0.006) | -0.012 (-0.060 to 0.057; -0.035 to -0.011) | - | - | 0.858 |
| Decompensation, 24 h | AmsterdamUMCdb | 154 | 300 | 0.885 | -0.006 (-0.042 to 0.028; -0.017 to -0.004) | 0.000 (-0.026 to 0.047; -0.029 to -0.007) | 247 | 304 | 0.858 |
| Decompensation, 24 h | AmsterdamUMCdb | 154 | 1000 | 0.923 | -0.000 (-0.016 to 0.014; -0.015 to -0.000) | - | 920 | - | 0.858 |
| Decompensation, 24 h | eICU | 125 | 100 | 0.736 | 0.021 (-0.142 to 0.114; -0.014 to 0.024) | 0.046 (-0.016 to 0.157; -0.017 to 0.029) | 144 | 197 | 0.800 |
| Decompensation, 24 h | eICU | 125 | 300 | 0.795 | 0.006 (-0.040 to 0.062; -0.015 to 0.021) | 0.027 (-0.019 to 0.093; -0.008 to 0.035) | 393 | 1,198 | 0.800 |
| Decompensation, 24 h | eICU | 125 | 1000 | 0.814 | 0.022 (-0.014 to 0.059; 0.000 to 0.042) | 0.054 (0.016 to 0.100; -0.001 to 0.051) | 1,585 | 3,097 | 0.800 |
| Decompensation, 24 h | HiRID | 91 | 100 | 0.849 | -0.008 (-0.070 to 0.046; -0.020 to 0.001) | 0.001 (-0.087 to 0.079; -0.016 to 0.008) | - | 107 | 0.868 |
| Decompensation, 24 h | HiRID | 91 | 300 | 0.873 | 0.007 (-0.042 to 0.041; -0.001 to 0.015) | 0.009 (-0.027 to 0.039; -0.006 to 0.020) | 403 | 429 | 0.868 |
| Decompensation, 24 h | HiRID | 91 | 1000 | 0.910 | -0.002 (-0.022 to 0.033; -0.010 to 0.004) | - | 947 | - | 0.868 |
| Decompensation, 24 h | INSPIRE | 51 | 100 | 0.603 | 0.082 (-0.042 to 0.225; -0.071 to 0.018) | 0.105 (-0.152 to 0.420; -0.062 to 0.027) | 167 | 185 | - |
| Decompensation, 24 h | INSPIRE | 51 | 300 | 0.667 | 0.035 (-0.196 to 0.266; -0.039 to 0.001) | 0.007 (-0.140 to 0.193; -0.091 to -0.024) | 180 | 157 | - |
| Decompensation, 24 h | MIMIC-IV | 125 | 100 | 0.825 | 0.012 (-0.092 to 0.101; -0.011 to 0.015) | 0.035 (-0.050 to 0.128; -0.005 to 0.017) | 128 | 179 | 0.891 |
| Decompensation, 24 h | MIMIC-IV | 125 | 300 | 0.867 | 0.024 (-0.033 to 0.167; 0.002 to 0.021) | 0.037 (-0.012 to 0.152; 0.009 to 0.030) | 511 | 653 | 0.891 |
| Decompensation, 24 h | MIMIC-IV | 125 | 1000 | 0.923 | 0.015 (-0.003 to 0.046; 0.005 to 0.026) | - | 1,794 | - | 0.891 |
| Decompensation, 24 h | NWICU | 117 | 100 | 0.787 | 0.039 (-0.032 to 0.157; -0.017 to 0.007) | 0.059 (-0.019 to 0.203; -0.002 to 0.027) | 195 | 260 | 0.867 |
| Decompensation, 24 h | NWICU | 117 | 300 | 0.858 | 0.011 (-0.031 to 0.073; -0.017 to 0.004) | 0.026 (-0.020 to 0.090; -0.017 to 0.010) | 470 | 733 | 0.867 |
| Decompensation, 24 h | NWICU | 117 | 1000 | 0.897 | 0.013 (-0.013 to 0.038; -0.010 to 0.012) | - | 1,639 | - | 0.867 |
| Decompensation, 24 h | SICdb | 79 | 100 | 0.775 | 0.058 (-0.163 to 0.377; -0.016 to 0.006) | 0.070 (-0.092 to 0.322; -0.019 to 0.006) | 229 | 247 | 0.827 |
| Decompensation, 24 h | SICdb | 79 | 300 | 0.883 | 0.002 (-0.044 to 0.086; -0.014 to 0.002) | 0.007 (-0.048 to 0.144; -0.019 to 0.005) | 369 | 528 | 0.827 |
| Decompensation, 24 h | SICdb | 79 | 1000 | 0.912 | -0.005 (-0.057 to 0.049; -0.026 to -0.004) | - | 892 | - | 0.827 |
| New IMV, 24 h | AmsterdamUMCdb | 26 | 100 | 0.697 | 0.055 (-0.089 to 0.184; 0.002 to 0.112) | 0.087 (-0.079 to 0.222; -0.023 to 0.122) | 292 | 486 | - |
| New IMV, 24 h | AmsterdamUMCdb | 26 | 300 | 0.752 | 0.001 (-0.117 to 0.090; -0.082 to 0.019) | 0.091 (-0.033 to 0.263; -0.110 to 0.081) | 308 | 2,396 | - |
| New IMV, 24 h | eICU | 45 | 100 | 0.640 | 0.048 (-0.123 to 0.249; -0.008 to 0.109) | 0.066 (-0.127 to 0.252; -0.032 to 0.055) | 291 | 347 | 0.758 |
| New IMV, 24 h | eICU | 45 | 300 | 0.693 | 0.042 (-0.208 to 0.284; -0.071 to 0.033) | 0.092 (-0.033 to 0.341; -0.030 to 0.078) | 457 | 759 | 0.758 |
| New IMV, 24 h | eICU | 45 | 1000 | 0.801 | 0.005 (-0.143 to 0.135; -0.035 to 0.035) | 0.015 (-0.086 to 0.141; -0.076 to 0.022) | 1,170 | 1,551 | 0.758 |
| New IMV, 24 h | HiRID | 33 | 100 | 0.543 | 0.097 (-0.051 to 0.209; 0.006 to 0.154) | 0.152 (-0.057 to 0.283; -0.005 to 0.186) | 288 | 990 | 0.686 |
| New IMV, 24 h | HiRID | 33 | 300 | 0.641 | 0.016 (-0.149 to 0.111; -0.059 to 0.014) | 0.052 (-0.066 to 0.172; -0.078 to 0.071) | 425 | 947 | 0.686 |
| New IMV, 24 h | HiRID | 33 | 1000 | 0.696 | 0.037 (-0.065 to 0.156; -0.062 to 0.037) | - | 2,296 | - | 0.686 |
| New IMV, 24 h | INSPIRE | 24 | 100 | 0.549 | -0.009 (-0.176 to 0.221; -0.197 to -0.015) | -0.052 (-0.249 to 0.194; -0.246 to -0.046) | - | - | - |
| New IMV, 24 h | INSPIRE | 24 | 300 | 0.588 | -0.011 (-0.427 to 0.342; -0.075 to 0.025) | -0.055 (-0.371 to 0.167; -0.169 to 0.027) | 276 | - | - |
| New IMV, 24 h | MIMIC-IV | 37 | 100 | 0.661 | 0.039 (-0.074 to 0.186; -0.049 to 0.041) | 0.056 (-0.072 to 0.203; -0.055 to 0.048) | 188 | 372 | 0.724 |
| New IMV, 24 h | MIMIC-IV | 37 | 300 | 0.706 | 0.001 (-0.156 to 0.103; -0.020 to 0.025) | 0.020 (-0.071 to 0.107; -0.059 to 0.021) | 310 | 436 | 0.724 |
| New IMV, 24 h | MIMIC-IV | 37 | 1000 | 0.779 | -0.006 (-0.077 to 0.041; -0.046 to 0.009) | - | 910 | - | 0.724 |
| New IMV, 24 h | NWICU | 39 | 100 | 0.658 | 0.024 (-0.109 to 0.175; 0.002 to 0.063) | 0.058 (-0.029 to 0.193; 0.015 to 0.095) | 182 | 564 | 0.744 |
| New IMV, 24 h | NWICU | 39 | 300 | 0.694 | 0.033 (-0.035 to 0.108; -0.013 to 0.047) | 0.062 (-0.012 to 0.132; -0.017 to 0.085) | 773 | 1,326 | 0.744 |
| New IMV, 24 h | NWICU | 39 | 1000 | 0.736 | 0.042 (-0.013 to 0.103; -0.007 to 0.057) | - | 1,680 | - | 0.744 |
| New IMV, 24 h | SICdb | 36 | 100 | 0.660 | 0.024 (-0.187 to 0.203; -0.089 to 0.081) | 0.034 (-0.272 to 0.248; -0.056 to 0.080) | 211 | 228 | 0.762 |
| New IMV, 24 h | SICdb | 36 | 300 | 0.737 | -0.032 (-0.286 to 0.226; -0.156 to 0.034) | -0.006 (-0.282 to 0.301; -0.123 to 0.036) | 246 | 290 | 0.762 |
| New IMV, 24 h | SICdb | 36 | 1000 | 0.693 | 0.090 (-0.053 to 0.340; -0.052 to 0.054) | - | 1,983 | - | 0.762 |
| New vasopressor, 24 h | AmsterdamUMCdb | 49 | 100 | 0.748 | 0.003 (-0.061 to 0.092; -0.034 to 0.034) | 0.019 (-0.079 to 0.120; -0.065 to 0.026) | 108 | 149 | - |
| New vasopressor, 24 h | AmsterdamUMCdb | 49 | 300 | 0.789 | 0.003 (-0.064 to 0.112; -0.037 to 0.016) | 0.031 (-0.065 to 0.143; -0.041 to 0.073) | 334 | 700 | - |
| New vasopressor, 24 h | eICU | 48 | 100 | 0.664 | 0.009 (-0.117 to 0.150; -0.007 to 0.080) | 0.027 (-0.241 to 0.233; -0.035 to 0.033) | 121 | 160 | 0.620 |
| New vasopressor, 24 h | eICU | 48 | 300 | 0.698 | 0.019 (-0.097 to 0.163; -0.032 to 0.031) | 0.023 (-0.117 to 0.146; 0.015 to 0.088) | 735 | 761 | 0.620 |
| New vasopressor, 24 h | eICU | 48 | 1000 | 0.758 | 0.005 (-0.083 to 0.131; -0.015 to 0.047) | 0.045 (-0.061 to 0.193; -0.020 to 0.072) | 1,122 | 2,942 | 0.620 |
| New vasopressor, 24 h | HiRID | 57 | 100 | 0.631 | 0.017 (-0.100 to 0.116; -0.115 to 0.050) | 0.025 (-0.126 to 0.158; -0.176 to 0.051) | 170 | 1,346 | 0.697 |
| New vasopressor, 24 h | HiRID | 57 | 300 | 0.649 | -0.005 (-0.143 to 0.114; -0.049 to 0.004) | 0.012 (-0.143 to 0.105; -0.109 to 0.016) | 152 | 1,582 | 0.697 |
| New vasopressor, 24 h | HiRID | 57 | 1000 | 0.649 | 0.012 (-0.093 to 0.127; -0.033 to 0.053) | - | 1,548 | - | 0.697 |
| New vasopressor, 24 h | INSPIRE | 47 | 100 | 0.509 | 0.016 (-0.115 to 0.143; -0.093 to 0.122) | 0.015 (-0.133 to 0.142; -0.061 to 0.143) | 3,295 | 3,282 | - |
| New vasopressor, 24 h | INSPIRE | 47 | 300 | 0.517 | 0.037 (-0.071 to 0.123; -0.055 to 0.132) | 0.034 (-0.087 to 0.151; -0.099 to 0.115) | 3,930 | 3,866 | - |
| New vasopressor, 24 h | MIMIC-IV | 38 | 100 | 0.707 | 0.048 (-0.070 to 0.197; -0.001 to 0.068) | 0.061 (-0.014 to 0.180; -0.001 to 0.068) | 165 | 182 | 0.780 |
| New vasopressor, 24 h | MIMIC-IV | 38 | 300 | 0.792 | 0.016 (-0.062 to 0.071; -0.015 to 0.038) | 0.018 (-0.071 to 0.097; -0.023 to 0.036) | 499 | 528 | 0.780 |
| New vasopressor, 24 h | MIMIC-IV | 38 | 1000 | 0.790 | 0.030 (-0.021 to 0.134; -0.015 to 0.033) | - | 1,620 | - | 0.780 |
| New vasopressor, 24 h | NWICU | 39 | 100 | 0.665 | 0.036 (-0.167 to 0.332; -0.037 to 0.089) | 0.090 (-0.093 to 0.264; -0.012 to 0.133) | 287 | 707 | 0.749 |
| New vasopressor, 24 h | NWICU | 39 | 300 | 0.715 | 0.056 (-0.075 to 0.200; -0.060 to 0.039) | 0.047 (-0.099 to 0.224; -0.143 to 0.004) | 894 | 786 | 0.749 |
| New vasopressor, 24 h | NWICU | 39 | 1000 | 0.779 | 0.017 (-0.105 to 0.146; -0.041 to 0.012) | - | 1,547 | - | 0.749 |
| New vasopressor, 24 h | SICdb | 80 | 100 | 0.691 | -0.005 (-0.067 to 0.109; -0.022 to 0.027) | -0.008 (-0.089 to 0.101; -0.053 to 0.008) | - | - | 0.708 |
| New vasopressor, 24 h | SICdb | 80 | 300 | 0.668 | 0.028 (-0.044 to 0.114; -0.034 to 0.043) | 0.036 (-0.040 to 0.105; -0.045 to 0.028) | 465 | 508 | 0.708 |
| New vasopressor, 24 h | SICdb | 80 | 1000 | 0.705 | 0.011 (-0.049 to 0.075; -0.031 to 0.019) | - | 584 | - | 0.708 |
| New RRT, 48 h | AmsterdamUMCdb | 38 | 100 | 0.789 | 0.007 (-0.190 to 0.268; -0.011 to 0.031) | 0.017 (-0.283 to 0.287; -0.008 to 0.027) | 121 | 152 | 0.747 |
| New RRT, 48 h | AmsterdamUMCdb | 38 | 300 | 0.849 | -0.037 (-0.220 to 0.182; -0.028 to 0.018) | -0.022 (-0.256 to 0.193; -0.031 to 0.031) | 167 | 215 | 0.747 |
| New RRT, 48 h | AmsterdamUMCdb | 38 | 1000 | 0.889 | -0.025 (-0.205 to 0.057; -0.020 to 0.016) | - | 474 | - | 0.747 |
| New RRT, 48 h | eICU | 29 | 100 | 0.697 | -0.016 (-0.173 to 0.266; -0.082 to -0.006) | -0.086 (-0.391 to 0.284; -0.043 to 0.003) | - | - | 0.745 |
| New RRT, 48 h | eICU | 29 | 300 | 0.707 | 0.001 (-0.310 to 0.288; -0.013 to 0.027) | 0.050 (-0.272 to 0.380; -0.030 to 0.060) | 116 | 186 | 0.745 |
| New RRT, 48 h | eICU | 29 | 1000 | 0.756 | 0.056 (-0.125 to 0.248; -0.044 to 0.082) | 0.130 (-0.014 to 0.338; -0.014 to 0.124) | 1,786 | >4000 | 0.745 |
| New RRT, 48 h | HiRID | 104 | 100 | 0.731 | -0.003 (-0.103 to 0.071; -0.036 to 0.015) | 0.039 (-0.091 to 0.168; -0.005 to 0.054) | - | 202 | 0.750 |
| New RRT, 48 h | HiRID | 104 | 300 | 0.817 | -0.010 (-0.086 to 0.055; -0.017 to 0.007) | -0.030 (-0.124 to 0.024; -0.071 to -0.010) | 278 | 236 | 0.750 |
| New RRT, 48 h | HiRID | 104 | 1000 | 0.835 | 0.008 (-0.032 to 0.055; -0.012 to 0.029) | - | 1,267 | - | 0.750 |
| New RRT, 48 h | INSPIRE | 23 | 100 | 0.665 | 0.058 (-0.180 to 0.346; 0.001 to 0.163) | 0.081 (-0.167 to 0.430; -0.009 to 0.134) | >4000 | >4000 | - |
| New RRT, 48 h | INSPIRE | 23 | 300 | 0.599 | 0.108 (-0.119 to 0.426; -0.002 to 0.288) | 0.249 (0.029 to 0.566; 0.013 to 0.306) | 3,558 | >4000 | - |
| New RRT, 48 h | MIMIC-IV | 40 | 100 | 0.794 | -0.049 (-0.248 to 0.164; -0.045 to -0.004) | -0.069 (-0.416 to 0.157; -0.074 to 0.016) | - | - | 0.780 |
| New RRT, 48 h | MIMIC-IV | 40 | 300 | 0.867 | 0.007 (-0.083 to 0.215; -0.031 to 0.005) | -0.006 (-0.110 to 0.236; -0.064 to -0.003) | 351 | 227 | 0.780 |
| New RRT, 48 h | MIMIC-IV | 40 | 1000 | 0.915 | -0.016 (-0.058 to 0.013; -0.035 to 0.011) | - | 543 | - | 0.780 |
| New RRT, 48 h | NWICU | 41 | 100 | 0.897 | -0.018 (-0.212 to 0.065; -0.002 to 0.013) | 0.018 (-0.180 to 0.195; -0.013 to 0.007) | - | 191 | 0.920 |
| New RRT, 48 h | NWICU | 41 | 300 | 0.935 | 0.001 (-0.018 to 0.039; -0.004 to 0.012) | 0.007 (-0.026 to 0.072; -0.006 to 0.015) | 349 | 544 | 0.920 |
| New RRT, 48 h | NWICU | 41 | 1000 | 0.955 | -0.000 (-0.014 to 0.016; -0.018 to 0.000) | - | 983 | - | 0.920 |
| New RRT, 48 h | SICdb | 28 | 100 | 0.723 | -0.022 (-0.167 to 0.110; -0.034 to 0.001) | -0.049 (-0.267 to 0.131; -0.084 to -0.014) | - | - | 0.580 |
| New RRT, 48 h | SICdb | 28 | 300 | 0.811 | -0.090 (-0.302 to 0.046; -0.067 to 0.004) | -0.153 (-0.361 to 0.017; -0.066 to -0.014) | - | - | 0.580 |
| New RRT, 48 h | SICdb | 28 | 1000 | 0.798 | -0.063 (-0.179 to 0.186; -0.048 to 0.006) | - | 196 | - | 0.580 |
| AKI stage >= 2, 48 h | AmsterdamUMCdb | 110 | 100 | 0.652 | 0.038 (-0.052 to 0.113; -0.022 to 0.031) | 0.044 (-0.044 to 0.166; -0.047 to 0.026) | 223 | 250 | 0.728 |
| AKI stage >= 2, 48 h | AmsterdamUMCdb | 110 | 300 | 0.707 | 0.019 (-0.018 to 0.066; -0.035 to 0.019) | 0.032 (-0.016 to 0.079; -0.037 to 0.028) | 609 | 937 | 0.728 |
| AKI stage >= 2, 48 h | AmsterdamUMCdb | 110 | 1000 | 0.742 | 0.020 (-0.008 to 0.050; -0.018 to 0.040) | - | 1,964 | - | 0.728 |
| AKI stage >= 2, 48 h | eICU | 88 | 100 | 0.641 | 0.036 (-0.034 to 0.136; -0.037 to 0.038) | 0.061 (-0.008 to 0.175; -0.029 to 0.032) | 230 | 403 | 0.710 |
| AKI stage >= 2, 48 h | eICU | 88 | 300 | 0.698 | 0.025 (-0.016 to 0.068; -0.014 to 0.033) | 0.035 (-0.008 to 0.097; -0.029 to 0.029) | 958 | 1,279 | 0.710 |
| AKI stage >= 2, 48 h | eICU | 88 | 1000 | 0.724 | 0.015 (-0.016 to 0.055; -0.010 to 0.026) | 0.024 (-0.008 to 0.062; -0.019 to 0.032) | 1,497 | 1,774 | 0.710 |
| AKI stage >= 2, 48 h | HiRID | 142 | 100 | 0.722 | 0.019 (-0.057 to 0.082; -0.019 to 0.031) | 0.035 (-0.023 to 0.089; -0.037 to 0.034) | 175 | 245 | 0.767 |
| AKI stage >= 2, 48 h | HiRID | 142 | 300 | 0.768 | -0.001 (-0.060 to 0.068; -0.049 to -0.004) | 0.017 (-0.049 to 0.108; -0.063 to 0.006) | 295 | 408 | 0.767 |
| AKI stage >= 2, 48 h | HiRID | 142 | 1000 | 0.835 | -0.004 (-0.023 to 0.024; -0.051 to 0.001) | - | 926 | - | 0.767 |
| AKI stage >= 2, 48 h | INSPIRE | 33 | 100 | 0.672 | 0.029 (-0.099 to 0.162; -0.053 to 0.025) | 0.018 (-0.214 to 0.257; -0.078 to 0.029) | 144 | 128 | - |
| AKI stage >= 2, 48 h | INSPIRE | 33 | 300 | 0.622 | 0.092 (-0.180 to 0.406; -0.031 to 0.173) | 0.129 (-0.023 to 0.295; -0.074 to 0.154) | 164 | 1,698 | - |
| AKI stage >= 2, 48 h | MIMIC-IV | 116 | 100 | 0.649 | 0.026 (-0.061 to 0.134; -0.040 to 0.024) | 0.045 (-0.050 to 0.168; -0.043 to 0.040) | 160 | 208 | 0.688 |
| AKI stage >= 2, 48 h | MIMIC-IV | 116 | 300 | 0.707 | 0.000 (-0.050 to 0.070; -0.055 to 0.003) | 0.003 (-0.046 to 0.046; -0.074 to -0.007) | 308 | 396 | 0.688 |
| AKI stage >= 2, 48 h | MIMIC-IV | 116 | 1000 | 0.735 | 0.001 (-0.026 to 0.044; -0.033 to 0.017) | - | 1,070 | - | 0.688 |
| AKI stage >= 2, 48 h | NWICU | 15 | 100 | 0.607 | -0.034 (-0.447 to 0.290; -0.040 to 0.168) | 0.000 (-0.402 to 0.322; -0.040 to 0.173) | - | 101 | 0.708 |
| AKI stage >= 2, 48 h | NWICU | 15 | 300 | 0.731 | -0.015 (-0.397 to 0.157; 0.003 to 0.083) | 0.055 (-0.162 to 0.264; -0.014 to 0.112) | 274 | 1,946 | 0.708 |
| AKI stage >= 2, 48 h | NWICU | 15 | 1000 | 0.677 | 0.103 (-0.155 to 0.492; 0.014 to 0.103) | - | 1,898 | - | 0.708 |
| AKI stage >= 2, 48 h | SICdb | 150 | 100 | 0.698 | 0.023 (-0.033 to 0.083; -0.019 to 0.026) | 0.046 (-0.019 to 0.123; -0.025 to 0.038) | 163 | 384 | 0.753 |
| AKI stage >= 2, 48 h | SICdb | 150 | 300 | 0.736 | 0.014 (-0.026 to 0.045; -0.025 to 0.015) | 0.023 (-0.007 to 0.069; -0.040 to 0.021) | 443 | 546 | 0.753 |
| AKI stage >= 2, 48 h | SICdb | 150 | 1000 | 0.775 | -0.007 (-0.041 to 0.020; -0.041 to 0.009) | - | 735 | - | 0.753 |

Annotations: LightGBM per dataset (A1's features and settings, early stopping on the dataset's tuning rows), tested on the dataset's held-out A1 anchors. Two 95% intervals: over the 20 draws of the real stays (how much one small site's gain varies), and a paired bootstrap over test anchors of the draw-averaged predictions (the uncertainty of the expected gain). The real equivalent interpolates the real-only curve (100-4,000 stays, 10 draws each). Few test positives make a dataset's row noise.

## F9. Transfer to new institutions, characterized (`runs/paper/f9/transfer_<design>.json`)

### R

(a) Site prompts (R128, 32 rollouts each):

| Endpoint | Anchors | AUROC known | unknown | swapped | unknown - known | swapped - known | O/E known | unknown | swapped | Swap shift: slope | corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 1014 | 0.890 | 0.892 | 0.895 | 0.002 (-0.022 to 0.027) | 0.005 (-0.027 to 0.038) | 1.87 | 1.68 | 0.56 | -1.36 | -0.17 |
| New IMV, 24 h | 695 | 0.801 | 0.838 | 0.781 | 0.037 (-0.035 to 0.130) | -0.020 (-0.180 to 0.129) | 0.74 | 0.86 | 0.53 | 0.88 | 0.12 |
| New vasopressor, 24 h | 761 | 0.872 | 0.855 | 0.750 | -0.018 (-0.047 to 0.008) | -0.123 (-0.250 to -0.021) | 0.52 | 0.49 | 0.38 | -0.36 | -0.05 |
| New RRT, 48 h | 929 | 0.939 | 0.944 | 0.641 | 0.004 (-0.024 to 0.033) | -0.298 (-0.392 to -0.180) | 0.61 | 0.49 | 0.92 | 0.06 | 0.02 |
| AKI stage >= 2, 48 h | 835 | 0.869 | 0.869 | 0.817 | -0.001 (-0.023 to 0.021) | -0.052 (-0.106 to -0.005) | 0.64 | 0.63 | 0.98 | 0.34 | 0.25 |
| Sepsis (ASE), 24 h | 1014 | 0.977 | 0.745 | 0.500 | -0.232 (-0.470 to 0.003) | -0.477 (-0.514 to -0.380) | 1.12 | 1.05 | 1.27 | 0.17 | 0.10 |
| Hypotension, 4 h | 862 | 0.831 | 0.887 | 0.737 | 0.056 (0.008 to 0.112) | -0.094 (-0.217 to 0.014) | 0.85 | 0.75 | 1.03 | 0.16 | 0.09 |
| Hyperkalaemia (top bin), 24 h | 886 | 0.798 | 0.774 | 0.750 | -0.025 (-0.105 to 0.059) | -0.049 (-0.115 to 0.003) | 0.65 | 0.63 | 0.60 | 0.21 | 0.05 |
| Thrombocytopenia, 48 h | 850 | 0.881 | 0.796 | 0.845 | -0.084 (-0.204 to 0.003) | -0.036 (-0.086 to -0.010) | 0.76 | 0.78 | 0.51 | 0.22 | 0.07 |

(b) AUROC by the anchor's novelty quintile (q1 most familiar; Twin / LightGBM):

| Endpoint | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|
| Decompensation, 24 h | 0.875 / 0.978 | 0.917 / 0.962 | 0.884 / 0.956 | 0.920 / 0.946 | 0.855 / 0.929 |
| New IMV, 24 h | 0.784 / 0.751 | 0.907 / 0.933 | 0.745 / 0.706 | 0.864 / 0.790 | 0.779 / 0.828 |
| New vasopressor, 24 h | 0.841 / 0.780 | 0.850 / 0.831 | 0.809 / 0.730 | 0.902 / 0.826 | 0.874 / 0.817 |
| New RRT, 48 h | 0.962 / 0.955 | 0.951 / 0.967 | 0.889 / 0.894 | 0.948 / 0.939 | 0.942 / 0.897 |
| AKI stage >= 2, 48 h | 0.937 / 0.915 | 0.920 / 0.910 | 0.895 / 0.850 | 0.886 / 0.812 | 0.830 / 0.791 |
| Sepsis (ASE), 24 h | - / - | 0.943 / 0.959 | 0.950 / 0.916 | 0.905 / 0.892 | 0.955 / 0.914 |
| Hypotension, 4 h | 0.936 / 0.958 | 0.952 / 0.942 | 0.925 / 0.942 | 0.916 / 0.908 | 0.817 / 0.867 |
| Hyperkalaemia (top bin), 24 h | 0.785 / 0.892 | 0.792 / 0.845 | 0.815 / 0.739 | 0.802 / 0.800 | 0.820 / 0.889 |
| Thrombocytopenia, 48 h | 0.984 / 0.972 | 0.936 / 0.955 | 0.977 / 0.964 | 0.970 / 0.965 | 0.860 / 0.931 |

eICU hospitals by their number of train stays (tertiles; Twin / LightGBM, positives). Every eICU hospital has its own site token in this vocabulary, so no anchor comes from a hospital the model knows only as 'unknown':

| Endpoint | Low train volume | Mid | High |
|---|---|---|---|
| Decompensation, 24 h | 0.776 / 0.846 (42) | 0.910 / 0.970 (49) | 0.785 / 0.897 (34) |
| New IMV, 24 h | 0.884 / 0.897 (20) | 0.914 / 0.843 (12) | 0.649 / 0.869 (13) |
| New vasopressor, 24 h | 0.969 / 0.931 (13) | 0.825 / 0.562 (20) | 0.822 / 0.745 (15) |
| New RRT, 48 h | 0.962 / 0.889 (9) | 0.986 / 0.965 (18) | 0.975 / 1.000 (1) |
| AKI stage >= 2, 48 h | 0.906 / 0.864 (24) | 0.912 / 0.784 (26) | 0.784 / 0.687 (36) |
| Sepsis (ASE), 24 h | - / - (0) | - / - (0) | - / - (0) |
| Hypotension, 4 h | 0.862 / 0.895 (18) | 0.969 / 0.994 (20) | 0.841 / 0.859 (11) |
| Hyperkalaemia (top bin), 24 h | 0.683 / 0.851 (6) | 0.630 / 0.642 (7) | 0.984 / 0.874 (17) |
| Thrombocytopenia, 48 h | 0.983 / 0.999 (1) | 0.449 / 0.649 (3) | 0.926 / 0.918 (10) |

(c) Leave one dataset out (AUROC on that dataset's held-out anchors):

| Endpoint | Dataset | Positives | LightGBM within | pooled | without the dataset | Probe pooled | Probe without | Twin (zero-shot) |
|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | AmsterdamUMCdb | 154 | 0.951 | 0.954 | 0.938 | 0.907 | 0.817 | 0.892 |
| Decompensation, 24 h | eICU | 125 | 0.887 | 0.908 | 0.911 | 0.855 | 0.788 | 0.828 |
| Decompensation, 24 h | HiRID | 91 | 0.941 | 0.952 | 0.943 | 0.913 | 0.878 | 0.885 |
| Decompensation, 24 h | INSPIRE | 51 | 0.881 | 0.980 | 0.974 | 0.970 | 0.967 | 0.912 |
| Decompensation, 24 h | MIMIC-IV | 125 | 0.954 | 0.962 | 0.958 | 0.952 | 0.942 | 0.927 |
| Decompensation, 24 h | NWICU | 117 | 0.935 | 0.942 | 0.881 | 0.927 | 0.911 | 0.896 |
| Decompensation, 24 h | SICdb | 79 | 0.940 | 0.957 | 0.947 | 0.914 | 0.883 | 0.873 |
| New IMV, 24 h | AmsterdamUMCdb | 26 | 0.854 | 0.862 | 0.766 | 0.709 | 0.582 | 0.894 |
| New IMV, 24 h | eICU | 45 | 0.873 | 0.882 | 0.817 | 0.572 | 0.466 | 0.795 |
| New IMV, 24 h | HiRID | 33 | 0.805 | 0.767 | 0.760 | 0.807 | 0.770 | 0.698 |
| New IMV, 24 h | INSPIRE | 24 | 0.817 | 0.814 | 0.611 | 0.799 | 0.886 | 0.630 |
| New IMV, 24 h | MIMIC-IV | 37 | 0.869 | 0.819 | 0.770 | 0.762 | 0.578 | 0.935 |
| New IMV, 24 h | NWICU | 39 | 0.811 | 0.818 | 0.728 | 0.754 | 0.658 | 0.860 |
| New IMV, 24 h | SICdb | 36 | 0.872 | 0.833 | 0.696 | 0.783 | 0.661 | 0.945 |
| New vasopressor, 24 h | AmsterdamUMCdb | 49 | 0.888 | 0.849 | 0.706 | 0.780 | 0.622 | 0.884 |
| New vasopressor, 24 h | eICU | 48 | 0.805 | 0.800 | 0.796 | 0.719 | 0.550 | 0.910 |
| New vasopressor, 24 h | HiRID | 57 | 0.655 | 0.690 | 0.689 | 0.564 | 0.465 | 0.693 |
| New vasopressor, 24 h | INSPIRE | 47 | 0.563 | 0.654 | 0.623 | 0.655 | 0.516 | 0.701 |
| New vasopressor, 24 h | MIMIC-IV | 38 | 0.869 | 0.875 | 0.868 | 0.730 | 0.693 | 0.942 |
| New vasopressor, 24 h | NWICU | 39 | 0.824 | 0.861 | 0.699 | 0.737 | 0.657 | 0.874 |
| New vasopressor, 24 h | SICdb | 80 | 0.719 | 0.705 | 0.695 | 0.632 | 0.562 | 0.769 |
| New RRT, 48 h | AmsterdamUMCdb | 38 | 0.939 | 0.957 | 0.932 | 0.932 | 0.891 | 0.912 |
| New RRT, 48 h | eICU | 29 | 0.888 | 0.941 | 0.877 | 0.927 | 0.857 | 0.975 |
| New RRT, 48 h | HiRID | 104 | 0.879 | 0.861 | 0.869 | 0.862 | 0.726 | 0.907 |
| New RRT, 48 h | INSPIRE | 23 | 0.639 | 0.816 | 0.766 | 0.735 | 0.672 | 0.721 |
| New RRT, 48 h | MIMIC-IV | 40 | 0.944 | 0.961 | 0.948 | 0.886 | 0.887 | 0.971 |
| New RRT, 48 h | NWICU | 41 | 0.977 | 0.975 | 0.964 | 0.954 | 0.908 | 0.986 |
| New RRT, 48 h | SICdb | 28 | 0.928 | 0.954 | 0.932 | 0.886 | 0.865 | 0.944 |
| AKI stage >= 2, 48 h | AmsterdamUMCdb | 110 | 0.785 | 0.783 | 0.753 | 0.805 | 0.743 | 0.875 |
| AKI stage >= 2, 48 h | eICU | 88 | 0.777 | 0.779 | 0.707 | 0.645 | 0.489 | 0.863 |
| AKI stage >= 2, 48 h | HiRID | 142 | 0.869 | 0.871 | 0.848 | 0.723 | 0.644 | 0.866 |
| AKI stage >= 2, 48 h | INSPIRE | 33 | 0.844 | 0.896 | 0.715 | 0.826 | 0.831 | 0.864 |
| AKI stage >= 2, 48 h | MIMIC-IV | 116 | 0.765 | 0.783 | 0.759 | 0.696 | 0.633 | 0.842 |
| AKI stage >= 2, 48 h | NWICU | 15 | 0.856 | 0.833 | 0.788 | 0.855 | 0.736 | 0.911 |
| AKI stage >= 2, 48 h | SICdb | 150 | 0.801 | 0.770 | 0.639 | 0.715 | 0.647 | 0.854 |
| Sepsis (ASE), 24 h | AmsterdamUMCdb | 9 | 0.801 | 0.828 | 0.458 | 0.687 | 0.596 | 0.945 |
| Sepsis (ASE), 24 h | MIMIC-IV | 19 | 0.787 | 0.682 | 0.619 | 0.758 | 0.641 | 0.842 |
| Hypotension, 4 h | AmsterdamUMCdb | 40 | 0.886 | 0.900 | 0.897 | 0.732 | 0.694 | 0.904 |
| Hypotension, 4 h | eICU | 49 | 0.891 | 0.914 | 0.919 | 0.759 | 0.696 | 0.886 |
| Hypotension, 4 h | HiRID | 46 | 0.845 | 0.850 | 0.834 | 0.786 | 0.681 | 0.884 |
| Hypotension, 4 h | INSPIRE | 22 | 0.973 | 0.942 | 0.947 | 0.850 | 0.778 | 0.915 |
| Hypotension, 4 h | MIMIC-IV | 41 | 0.890 | 0.908 | 0.918 | 0.742 | 0.758 | 0.841 |
| Hypotension, 4 h | SICdb | 76 | 0.883 | 0.908 | 0.878 | 0.758 | 0.655 | 0.893 |
| Hyperkalaemia (top bin), 24 h | AmsterdamUMCdb | 52 | 0.874 | 0.906 | 0.887 | 0.735 | 0.548 | 0.901 |
| Hyperkalaemia (top bin), 24 h | eICU | 30 | 0.814 | 0.812 | 0.770 | 0.694 | 0.732 | 0.814 |
| Hyperkalaemia (top bin), 24 h | HiRID | 38 | 0.873 | 0.889 | 0.848 | 0.676 | 0.593 | 0.853 |
| Hyperkalaemia (top bin), 24 h | INSPIRE | 41 | 0.829 | 0.867 | 0.839 | 0.535 | 0.388 | 0.729 |
| Hyperkalaemia (top bin), 24 h | MIMIC-IV | 49 | 0.752 | 0.810 | 0.790 | 0.734 | 0.677 | 0.799 |
| Hyperkalaemia (top bin), 24 h | NWICU | 19 | 0.614 | 0.639 | 0.640 | 0.690 | 0.658 | 0.746 |
| Hyperkalaemia (top bin), 24 h | SICdb | 54 | 0.691 | 0.666 | 0.731 | 0.626 | 0.492 | 0.650 |
| Thrombocytopenia, 48 h | AmsterdamUMCdb | 22 | 0.985 | 0.969 | 0.975 | 0.920 | 0.913 | 0.995 |
| Thrombocytopenia, 48 h | eICU | 14 | 0.869 | 0.820 | 0.809 | 0.600 | 0.530 | 0.730 |
| Thrombocytopenia, 48 h | HiRID | 13 | 0.983 | 0.985 | 0.984 | 0.974 | 0.980 | 0.990 |
| Thrombocytopenia, 48 h | INSPIRE | 50 | 0.964 | 0.961 | 0.947 | 0.850 | 0.855 | 0.966 |
| Thrombocytopenia, 48 h | MIMIC-IV | 15 | 0.907 | 0.943 | 0.927 | 0.791 | 0.697 | 0.871 |
| Thrombocytopenia, 48 h | NWICU | 15 | 0.927 | 0.945 | 0.944 | 0.837 | 0.817 | 0.917 |
| Thrombocytopenia, 48 h | SICdb | 23 | 0.953 | 0.966 | 0.963 | 0.913 | 0.896 | 0.981 |

### L24

(b) AUROC by the anchor's novelty quintile (q1 most familiar; Twin / LightGBM):

| Endpoint | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|
| ICU death | 0.895 / 0.989 | 0.794 / 0.846 | 0.762 / 0.894 | 0.851 / 0.879 | 0.914 / 0.922 |
| In-hospital death | 0.874 / 0.913 | 0.798 / 0.826 | 0.856 / 0.830 | 0.853 / 0.867 | 0.887 / 0.872 |
| ICU stay > 7 days | 0.857 / 0.896 | 0.810 / 0.866 | 0.849 / 0.867 | 0.777 / 0.828 | 0.758 / 0.807 |

eICU hospitals by their number of train stays (tertiles; Twin / LightGBM, positives). Every eICU hospital has its own site token in this vocabulary, so no anchor comes from a hospital the model knows only as 'unknown':

| Endpoint | Low train volume | Mid | High |
|---|---|---|---|
| ICU death | 0.830 / 0.737 (4) | 1.000 / 0.964 (5) | 0.992 / 0.911 (2) |
| In-hospital death | 0.929 / 0.845 (15) | 0.952 / 0.870 (14) | 0.835 / 0.790 (10) |
| ICU stay > 7 days | 0.663 / 0.791 (12) | 0.715 / 0.933 (13) | 0.648 / 0.711 (19) |

(c) Leave one dataset out (AUROC on that dataset's held-out anchors):

| Endpoint | Dataset | Positives | LightGBM within | pooled | without the dataset | Probe pooled | Probe without | Twin (zero-shot) |
|---|---|---|---|---|---|---|---|---|
| ICU death | AmsterdamUMCdb | 82 | 0.867 | 0.868 | 0.839 | 0.841 | 0.817 | 0.875 |
| ICU death | eICU | 19 | 0.828 | 0.898 | 0.900 | 0.843 | 0.846 | 0.939 |
| ICU death | HiRID | 29 | 0.872 | 0.920 | 0.910 | 0.831 | 0.850 | 0.877 |
| ICU death | INSPIRE | 7 | 0.913 | 0.918 | 0.866 | 0.888 | 0.781 | 0.998 |
| ICU death | MIMIC-IV | 45 | 0.852 | 0.883 | 0.890 | 0.858 | 0.886 | 0.843 |
| ICU death | NWICU | 31 | 0.881 | 0.868 | 0.824 | 0.835 | 0.802 | 0.761 |
| ICU death | SICdb | 24 | 0.898 | 0.911 | 0.917 | 0.867 | 0.848 | 0.786 |
| In-hospital death | AmsterdamUMCdb | 82 | 0.867 | 0.856 | 0.840 | 0.853 | 0.833 | 0.850 |
| In-hospital death | eICU | 39 | 0.818 | 0.838 | 0.822 | 0.829 | 0.774 | 0.913 |
| In-hospital death | HiRID | 29 | 0.870 | 0.907 | 0.888 | 0.831 | 0.825 | 0.847 |
| In-hospital death | INSPIRE | 27 | 0.755 | 0.824 | 0.774 | 0.742 | 0.601 | 0.731 |
| In-hospital death | MIMIC-IV | 67 | 0.850 | 0.863 | 0.853 | 0.821 | 0.827 | 0.857 |
| In-hospital death | NWICU | 46 | 0.852 | 0.867 | 0.840 | 0.830 | 0.801 | 0.896 |
| In-hospital death | SICdb | 56 | 0.832 | 0.856 | 0.850 | 0.846 | 0.820 | 0.869 |

Annotations: the Twin saw every dataset's training stays, so none of this is transfer of the generative model to a new site (that needs a model trained without a dataset). (a) shows how much the predictions depend on the site tokens; (c) what a supervised model loses at a site without labels.

## Amendment 11. The application toolkit's validation

The toolkit (`src/icutwin/toolkit`, `docs/toolkit.md`) run on the RTX 5090 host; its rollouts are new draws (seeded per request), so V1 compares estimates within sampling error.

### V1. Conformance (`runs/toolkit/v1/`): the original rule not passed; the added check not passed; the study lead's adjusted criterion passed

Verdicts in the order they were made (amendment 11: the original rule fixed before the run; the added rollout-noise check and the study lead's adjusted criterion post hoc, after V1's result).

The toolkit against both paper runs on the same anchors (A1: 32 rollouts; F1: 128, its AUROC curve at 32 rollouts and its O/E). Limit: 1.96 x sqrt(2) x the toolkit's rollout-bootstrap SD.

| Endpoint | Toolkit AUROC | A1 | F1 (32) | Limit | Toolkit O/E | A1 O/E | F1 O/E | Limit | Original rule | Added check | Within the limit of |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 0.880 | 0.903 | 0.881 | 0.017 | 1.69 | 1.85 | 1.74 | 0.15 | yes | no | F1 |
| New IMV, 24 h | 0.843 | 0.838 | 0.837 | 0.065 | 0.83 | 0.80 | 0.76 | 0.09 | yes | yes | A1, F1 |
| New vasopressor, 24 h | 0.899 | 0.889 | 0.887 | 0.024 | 0.56 | 0.61 | 0.57 | 0.05 | yes | yes | A1, F1 |
| New RRT, 48 h | 0.948 | 0.932 | 0.934 | 0.016 | 0.56 | 0.57 | 0.56 | 0.06 | yes | no | F1 |
| AKI stage >= 2, 48 h | 0.893 | 0.887 | 0.883 | 0.018 | 0.67 | 0.66 | 0.65 | 0.03 | yes | yes | A1, F1 |
| Sepsis (ASE), 24 h | 0.963 | 0.989 | 0.900 | 0.152 | 0.80 | 0.83 | 0.92 | 0.15 | no | yes | A1, F1 |
| Hypotension, 4 h | 0.891 | 0.875 | 0.878 | 0.021 | 0.98 | 0.95 | 0.96 | 0.07 | yes | yes | A1, F1 |
| Hyperkalaemia (top bin), 24 h | 0.806 | 0.847 | 0.805 | 0.055 | 0.82 | 0.76 | 0.76 | 0.06 | yes | yes | A1, F1 |
| Thrombocytopenia, 48 h | 0.892 | 0.829 | 0.886 | 0.077 | 0.78 | 0.76 | 0.82 | 0.09 | yes | yes | A1, F1 |

The toolkit's forecast on F1's R128 anchors (32 rollouts, A1's caps) against A1's 32-rollout Twin on the same anchors:

| Endpoint | Anchors | A1 AUROC (95% CI) | Toolkit AUROC | A1 O/E (95% CI) | Toolkit O/E | Within |
|---|---|---|---|---|---|---|
| Decompensation, 24 h | 1045 | 0.903 (0.871 to 0.933) | 0.880 | 1.85 (1.49 to 2.26) | 1.69 | yes |
| New IMV, 24 h | 739 | 0.838 (0.723 to 0.919) | 0.843 | 0.80 (0.51 to 1.20) | 0.83 | yes |
| New vasopressor, 24 h | 802 | 0.889 (0.825 to 0.927) | 0.899 | 0.61 (0.40 to 0.82) | 0.56 | yes |
| New RRT, 48 h | 974 | 0.932 (0.896 to 0.962) | 0.948 | 0.57 (0.38 to 0.77) | 0.56 | yes |
| AKI stage >= 2, 48 h | 882 | 0.887 (0.842 to 0.924) | 0.893 | 0.66 (0.54 to 0.78) | 0.67 | yes |
| Sepsis (ASE), 24 h | 1045 | 0.989 (0.974 to 0.999) | 0.963 | 0.83 (0.27 to 1.48) | 0.80 | no |
| Hypotension, 4 h | 914 | 0.875 (0.793 to 0.951) | 0.891 | 0.95 (0.68 to 1.25) | 0.98 | yes |
| Hyperkalaemia (top bin), 24 h | 923 | 0.847 (0.772 to 0.913) | 0.806 | 0.76 (0.52 to 1.01) | 0.82 | yes |
| Thrombocytopenia, 48 h | 888 | 0.829 (0.664 to 0.978) | 0.892 | 0.76 (0.41 to 1.21) | 0.78 | yes |

E2 at 32 rollouts against F2's curve (within 0.02):

| Endpoint | Anchors | Toolkit E2 AUROC | F2 at k = 32 | Within |
|---|---|---|---|---|
| Decompensation, 24 h | 1045 | 0.940 | 0.951 | yes |
| New IMV, 24 h | 744 | 0.856 | 0.866 | yes |
| New vasopressor, 24 h | 802 | 0.894 | 0.896 | yes |
| New RRT, 48 h | 979 | 0.958 | 0.957 | yes |

F5's factual contrast (starters' give minus matched non-starters' withhold) and the A/A null:

| Treatment: response | Pairs | Toolkit (95% CI) | F5's Twin (95% CI) | A/A prefixes | A/A mean (95% CI) | Within |
|---|---|---|---|---|---|---|
| norepinephrine:map | 165 | -3.85 (-6.15 to -1.50) | -3.58 (-5.85 to -1.07) | 172 | 0.32 (-0.22 to 0.94) | yes |
| furosemide:urine_output | 95 | 405.16 (294.13 to 520.31) | 403.27 (303.95 to 510.87) | 104 | 32.64 (-18.43 to 90.76) | yes |

### V2. Recalibration (`runs/toolkit/v2/v2.json`)

Isotonic maps from the tuning split's R anchors (E2 for the four hazard programs, the share otherwise), applied to V1's draws (A1's labels); raw → recalibrated. A map is adopted by the toolkit where AUROC stayed within 0.005 at both 32 and 16 rollouts.

| Endpoint | Rollouts | Anchors | O/E | Calibration slope | ECE | Brier | AUROC | AUROC within 0.005 | Adopted |
|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 32 | 1045 | 1.71 → 0.75 | 0.82 → 0.81 | 0.012 → 0.014 | 0.0182 → 0.0192 | 0.940 → 0.939 | yes | yes |
| New IMV, 24 h | 32 | 744 | 0.55 → 1.16 | 0.65 → 0.65 | 0.031 → 0.013 | 0.0315 → 0.0274 | 0.856 → 0.863 | no | no |
| New vasopressor, 24 h | 32 | 802 | 0.46 → 1.00 | 0.78 → 0.41 | 0.030 → 0.004 | 0.0246 → 0.0230 | 0.894 → 0.885 | no | no |
| New RRT, 48 h | 32 | 979 | 0.55 → 0.84 | 1.33 → 0.69 | 0.016 → 0.006 | 0.0151 → 0.0152 | 0.958 → 0.940 | no | no |
| AKI stage >= 2, 48 h | 32 | 882 | 0.67 → 0.82 | 0.79 → 0.90 | 0.068 → 0.035 | 0.0805 → 0.0740 | 0.893 → 0.893 | yes | yes |
| Sepsis (ASE), 24 h | 32 | 1045 | 0.80 → 1.19 | 1.08 → 2.37 | 0.003 → 0.006 | 0.0066 → 0.0082 | 0.963 → 0.932 | no | no |
| Hypotension, 4 h | 32 | 914 | 0.98 → 0.90 | 0.55 → 0.78 | 0.014 → 0.017 | 0.0347 → 0.0355 | 0.891 → 0.890 | yes | yes |
| Hyperkalaemia (top bin), 24 h | 32 | 923 | 0.82 → 0.67 | 0.47 → 0.71 | 0.024 → 0.039 | 0.0435 → 0.0467 | 0.806 → 0.806 | yes | yes |
| Thrombocytopenia, 48 h | 32 | 888 | 0.78 → 0.85 | 0.67 → 1.08 | 0.010 → 0.010 | 0.0126 → 0.0123 | 0.892 → 0.839 | no | no |
| Decompensation, 24 h | 16 | 1045 | 1.86 → 0.84 | 0.79 → 0.78 | 0.013 → 0.011 | 0.0185 → 0.0191 | 0.939 → 0.938 | yes | yes |
| New IMV, 24 h | 16 | 744 | 0.55 → 1.18 | 0.59 → 0.59 | 0.031 → 0.017 | 0.0331 → 0.0267 | 0.842 → 0.841 | yes | no |
| New vasopressor, 24 h | 16 | 802 | 0.45 → 0.95 | 0.85 → 0.39 | 0.031 → 0.008 | 0.0250 → 0.0234 | 0.886 → 0.874 | no | no |
| New RRT, 48 h | 16 | 979 | 0.54 → 0.87 | 1.25 → 0.68 | 0.016 → 0.006 | 0.0156 → 0.0151 | 0.955 → 0.935 | no | no |
| AKI stage >= 2, 48 h | 16 | 883 | 0.67 → 0.81 | 0.54 → 0.69 | 0.069 → 0.037 | 0.0861 → 0.0792 | 0.882 → 0.884 | yes | yes |
| Sepsis (ASE), 24 h | 16 | 1045 | 0.73 → 1.37 | 0.85 → 2.11 | 0.004 → 0.007 | 0.0068 → 0.0082 | 0.949 → 0.928 | no | no |
| Hypotension, 4 h | 16 | 914 | 0.98 → 0.89 | 0.50 → 0.59 | 0.015 → 0.023 | 0.0358 → 0.0383 | 0.887 → 0.886 | yes | yes |
| Hyperkalaemia (top bin), 24 h | 16 | 923 | 0.82 → 0.65 | 0.29 → 0.62 | 0.033 → 0.043 | 0.0452 → 0.0475 | 0.772 → 0.773 | yes | yes |
| Thrombocytopenia, 48 h | 16 | 886 | 0.74 → 0.83 | 0.56 → 0.77 | 0.011 → 0.013 | 0.0146 → 0.0139 | 0.889 → 0.778 | no | no |

Annotations: the isotonic steps tie risks, so AUROC can fall (most for the rarer endpoints); a map fixes O/E only where the tuning split's miscalibration matches the held-out anchors' (death within 24 h: raw O/E 1.98 on tuning against 1.71 here, so its map overcorrects).

### V2b. Logistic recalibration of the treatment starts (`runs/toolkit/v2/v2b.json`)

Post hoc (the study lead's decision after V2, recorded before it ran): logit(p') = a + b logit(p) on E2, fitted on V2's tuning anchors; adopted when b > 0 and, at 32 and 16 rollouts, O/E is closer to 1 and ECE no larger. Raw → recalibrated on V1's draws:

| Endpoint | a | b | Rollouts | Anchors | O/E | ECE | Brier | AUROC | Adopted |
|---|---|---|---|---|---|---|---|---|---|
| New IMV, 24 h | -1.74 | 0.72 | 32 | 744 | 0.55 → 1.43 | 0.031 → 0.013 | 0.0315 → 0.0249 | 0.856 → 0.856 | yes |
| New IMV, 24 h | -1.74 | 0.72 | 16 | 744 | 0.55 → 1.44 | 0.031 → 0.011 | 0.0331 → 0.0254 | 0.842 → 0.842 | yes |
| New vasopressor, 24 h | -1.41 | 0.80 | 32 | 802 | 0.46 → 1.02 | 0.030 → 0.009 | 0.0246 → 0.0227 | 0.894 → 0.894 | yes |
| New vasopressor, 24 h | -1.41 | 0.80 | 16 | 802 | 0.45 → 1.04 | 0.031 → 0.009 | 0.0250 → 0.0221 | 0.886 → 0.886 | yes |
| New RRT, 48 h | 0.02 | 1.46 | 32 | 979 | 0.55 → 0.93 | 0.016 → 0.002 | 0.0151 → 0.0148 | 0.958 → 0.958 | yes |
| New RRT, 48 h | 0.02 | 1.46 | 16 | 979 | 0.54 → 0.90 | 0.016 → 0.004 | 0.0156 → 0.0154 | 0.955 → 0.955 | yes |

Maps in force: AKI stage >= 2, 48 h, Decompensation, 24 h, Hyperkalaemia (top bin), 24 h, Hypotension, 4 h, New IMV, 24 h, New RRT, 48 h, New vasopressor, 24 h.

### V3. Privacy of prompted and intervened sampling: INCONCLUSIVE (V3 at 500 per side: INCONCLUSIVE; the loss test at P1's sizes, V3x, in its place) (controls valid: yes)

| Test | Members | Non-members | AUC or rate (95% CI) | TPR at 1% FPR (95% CI) | Status |
|---|---|---|---|---|---|
| (a) loss, pooled | 15024 | 15027 | 0.531 (0.524 to 0.537) | 0.013 (0.011 to 0.017) | PASS |
| (a) loss, MIMIC-IV | 3000 | 3000 | 0.529 (0.515 to 0.543) | 0.012 (0.009 to 0.020) | PASS |
| (a) loss, AmsterdamUMCdb | 1365 | 1368 | 0.550 (0.528 to 0.572) | 0.016 (0.007 to 0.025) | PASS |
| (a) loss, eICU | 2997 | 2998 | 0.540 (0.525 to 0.554) | 0.017 (0.007 to 0.026) | PASS |
| (a) loss, HiRID | 2031 | 2035 | 0.555 (0.538 to 0.573) | 0.017 (0.010 to 0.027) | PASS |
| (a) loss, SICdb | 2087 | 2086 | 0.538 (0.521 to 0.556) | 0.008 (0.004 to 0.015) | PASS |
| (a) loss, NWICU | 2753 | 2750 | 0.534 (0.518 to 0.549) | 0.012 (0.006 to 0.021) | PASS |
| (a) loss, INSPIRE | 791 | 790 | 0.568 (0.538 to 0.597) | 0.018 (0.009 to 0.039) | INCONCLUSIVE |
| (b) extraction: first 32 tokens |  |  | 0.000 vs 0.000 (difference 0.000) |  | PASS |
| (b) extraction: event Jaccard | 1041 | 1039 | 0.516 (0.493 to 0.540) |  | PASS |
| (c) intervened: first 32 tokens |  |  | 0.000 vs 0.000 (difference 0.000) |  | PASS |
| (c) intervened: event Jaccard | 100 | 100 | 0.482 (0.400 to 0.553) |  | PASS |
| Control: copying generator |  |  | difference 1.000 |  | FAIL |
| Control: private reference (loss) | 7513 | 7514 | 0.493 (0.483 to 0.502) | 0.013 (0.009 to 0.017) | PASS |

Annotations: P1's rule (PASS: AUC upper bound at most 0.60 and TPR upper bound at most 0.03; FAIL: a lower bound above 0.60 or 0.05); P2's limit for the exact match (member minus non-member at most 0.01). Written also to `runs/audit/m7_MUC32_main/prompted.json`.

The M7 gate report records conditional and counterfactual sampling as PASS (recorded 2026-10-11, the study lead's decision; the rule's verdict INCONCLUSIVE; stated: INSPIRE inconclusive (TPR at 1% FPR 0.018, 0.009 to 0.039; 791 per side, every eligible held-out stay)).

V3's own loss test at up to 500 per side (replaced by V3x's above, the study lead's decision):

| Dataset | Members | Non-members | AUC (95% CI) | TPR at 1% FPR (95% CI) | Status |
|---|---|---|---|---|---|
| pooled | 3482 | 3483 | 0.525 (0.512 to 0.539) | 0.011 (0.007 to 0.015) | PASS |
| MIMIC-IV | 500 | 500 | 0.504 (0.468 to 0.541) | 0.008 (0.000 to 0.024) | PASS |
| AmsterdamUMCdb | 499 | 500 | 0.551 (0.513 to 0.588) | 0.022 (0.006 to 0.042) | INCONCLUSIVE |
| eICU | 498 | 497 | 0.527 (0.490 to 0.561) | 0.020 (0.004 to 0.036) | INCONCLUSIVE |
| HiRID | 493 | 493 | 0.553 (0.519 to 0.591) | 0.014 (0.004 to 0.032) | INCONCLUSIVE |
| SICdb | 496 | 495 | 0.524 (0.486 to 0.559) | 0.006 (0.000 to 0.026) | PASS |
| NWICU | 498 | 499 | 0.520 (0.485 to 0.553) | 0.014 (0.004 to 0.028) | PASS |
| INSPIRE | 498 | 499 | 0.538 (0.502 to 0.572) | 0.012 (0.000 to 0.028) | PASS |

### V4. Trial designer (`runs/toolkit/v4/v4.json`)

Checks: type I error within 0.04-0.06 (both analyses) in 92 of 108 score cells; unadjusted simulated size within 5% of the analytic in 92 of 108.

Relative risk 0.8 (sizes in patients, both arms; 80% power, two-sided 0.05):

| Endpoint | Score | AUROC | Event rate | N, all | N, top 50% | N, top 25% | N adjusted / unadjusted | Type I (unadjusted) | Type I (adjusted) |
|---|---|---|---|---|---|---|---|---|---|
| ICU death (L24x) | Twin | 0.878 | 0.338 | 1454 | 513 | 307 | 0.64 | 0.052 | 0.047 |
| ICU death (L24x) | SOFA | 0.733 | 0.338 | 1454 | 839 | 469 | 0.87 | 0.063 | 0.059 |
| ICU death (L24x) | SAPS II | 0.774 | 0.338 | 1454 | 709 | 369 | 0.80 | 0.051 | 0.046 |
| ICU death (L24x) | OASIS | 0.750 | 0.338 | 1454 | 779 | 431 | 0.84 | 0.049 | 0.044 |
| ICU death (L24x) | NEWS | 0.758 | 0.338 | 1454 | 763 | 431 | 0.84 | 0.048 | 0.052 |
| ICU death (L24x) | LightGBM | 0.882 | 0.338 | 1454 | 529 | 293 | 0.64 | 0.043 | 0.050 |
| ICU death (L24x, eICU) | Twin | 0.876 | 0.226 | 2485 | 1010 | 503 | 0.67 | 0.050 | 0.044 |
| ICU death (L24x, eICU) | APACHE IVa | 0.756 | 0.226 | 2276 | 1304 | 696 | 0.81 | 0.053 | 0.037 |
| ICU death (L24x, eICU) | LightGBM | 0.845 | 0.226 | 2485 | 1010 | 598 | 0.75 | 0.051 | 0.050 |
| In-hospital death (L24x) | Twin | 0.883 | 0.494 | 791 | 251 | 139 | 0.63 | 0.051 | 0.051 |
| In-hospital death (L24x) | SOFA | 0.690 | 0.494 | 791 | 481 | 320 | 0.92 | 0.066 | 0.060 |
| In-hospital death (L24x) | SAPS II | 0.750 | 0.494 | 791 | 396 | 239 | 0.86 | 0.049 | 0.045 |
| In-hospital death (L24x) | OASIS | 0.709 | 0.494 | 791 | 466 | 264 | 0.89 | 0.052 | 0.052 |
| In-hospital death (L24x) | NEWS | 0.715 | 0.494 | 791 | 446 | 298 | 0.84 | 0.051 | 0.051 |
| In-hospital death (L24x) | LightGBM | 0.858 | 0.494 | 791 | 295 | 154 | 0.69 | 0.043 | 0.040 |
| In-hospital death (L24x, eICU) | Twin | 0.874 | 0.464 | 883 | 289 | 234 | 0.62 | 0.053 | 0.054 |
| In-hospital death (L24x, eICU) | APACHE IVa | 0.711 | 0.464 | 866 | 527 | 363 | 0.89 | 0.049 | 0.049 |
| In-hospital death (L24x, eICU) | LightGBM | 0.825 | 0.464 | 883 | 319 | 234 | 0.76 | 0.041 | 0.043 |
| New IMV, 24 h (L24) | Twin | 0.838 | 0.031 | 22183 | 12833 | 6715 | 0.86 | 0.054 | 0.051 |
| New IMV, 24 h (L24) | SOFA | 0.530 | 0.031 | 22141 | 19538 | 19553 | 1.00 | 0.052 | 0.052 |
| New IMV, 24 h (L24) | SAPS II | 0.539 | 0.031 | 22183 | 19568 | 16096 | 1.00 | 0.049 | 0.048 |
| New IMV, 24 h (L24) | OASIS | 0.584 | 0.031 | 22183 | 17662 | 18027 | 1.00 | 0.054 | 0.053 |
| New IMV, 24 h (L24) | NEWS | 0.709 | 0.031 | 22183 | 14757 | 9687 | 0.99 | 0.048 | 0.049 |
| New IMV, 24 h (L24) | LightGBM | 0.829 | 0.031 | 22183 | 11796 | 7450 | 0.92 | 0.051 | 0.060 |
| New IMV, 24 h (L24, eICU) | Twin | 0.869 | 0.025 | 27759 | 15138 | 8280 | 0.72 | 0.044 | 0.046 |
| New IMV, 24 h (L24, eICU) | APACHE IVa | 0.788 | 0.025 | 24949 | 12156 | 7637 | 1.00 | 0.048 | 0.053 |
| New IMV, 24 h (L24, eICU) | LightGBM | 0.868 | 0.025 | 27759 | 13561 | 8280 | 0.89 | 0.056 | 0.058 |
| New vasopressor, 24 h (L24) | Twin | 0.880 | 0.045 | 15194 | 7658 | 4033 | 0.84 | 0.051 | 0.048 |
| New vasopressor, 24 h (L24) | SOFA | 0.675 | 0.045 | 15167 | 10379 | 7328 | 0.98 | 0.052 | 0.050 |
| New vasopressor, 24 h (L24) | SAPS II | 0.655 | 0.045 | 15194 | 10757 | 7463 | 0.99 | 0.059 | 0.060 |
| New vasopressor, 24 h (L24) | OASIS | 0.608 | 0.045 | 15194 | 11562 | 10164 | 0.97 | 0.049 | 0.045 |
| New vasopressor, 24 h (L24) | NEWS | 0.614 | 0.045 | 15194 | 11281 | 10164 | 1.01 | 0.057 | 0.057 |
| New vasopressor, 24 h (L24) | LightGBM | 0.806 | 0.045 | 15194 | 8225 | 5254 | 0.93 | 0.054 | 0.052 |
| New vasopressor, 24 h (L24, eICU) | Twin | 0.991 | 0.015 | 46388 | 22875 | 11169 | 0.57 | 0.042 | 0.067 |
| New vasopressor, 24 h (L24, eICU) | APACHE IVa | 0.791 | 0.015 | 43630 | 25922 | 12642 | 0.95 | 0.053 | 0.051 |
| New vasopressor, 24 h (L24, eICU) | LightGBM | 0.925 | 0.015 | 46388 | 22875 | 11169 | 0.80 | 0.050 | 0.055 |
| New RRT, 48 h (L24) | Twin | 0.939 | 0.028 | 25003 | 12605 | 6132 | 0.74 | 0.054 | 0.043 |
| New RRT, 48 h (L24) | SOFA | 0.816 | 0.028 | 24965 | 13864 | 8481 | 0.93 | 0.052 | 0.050 |
| New RRT, 48 h (L24) | SAPS II | 0.757 | 0.028 | 25003 | 15013 | 9405 | 0.96 | 0.038 | 0.044 |
| New RRT, 48 h (L24) | OASIS | 0.680 | 0.028 | 25003 | 16827 | 10731 | 0.99 | 0.052 | 0.053 |
| New RRT, 48 h (L24) | NEWS | 0.646 | 0.028 | 25003 | 18799 | 13375 | 1.00 | 0.050 | 0.050 |
| New RRT, 48 h (L24) | LightGBM | 0.922 | 0.028 | 25003 | 12461 | 6622 | 0.77 | 0.057 | 0.064 |
| New RRT, 48 h (L24, eICU) | Twin | 0.990 | 0.012 | 57404 | 28442 | 13961 | 0.69 | 0.047 | 0.071 |
| New RRT, 48 h (L24, eICU) | APACHE IVa | 0.638 | 0.012 | 55591 | 34505 | 70002 | 1.01 | 0.053 | 0.054 |
| New RRT, 48 h (L24, eICU) | LightGBM | 0.937 | 0.012 | 57404 | 28442 | 13961 | 0.91 | 0.054 | 0.066 |
| AKI stage >= 2, 48 h (L24) | Twin | 0.891 | 0.202 | 2858 | 1213 | 590 | 0.67 | 0.043 | 0.048 |
| AKI stage >= 2, 48 h (L24) | SOFA | 0.701 | 0.202 | 2852 | 1791 | 1237 | 0.92 | 0.039 | 0.040 |
| AKI stage >= 2, 48 h (L24) | SAPS II | 0.680 | 0.202 | 2858 | 1824 | 1425 | 0.94 | 0.044 | 0.043 |
| AKI stage >= 2, 48 h (L24) | OASIS | 0.674 | 0.202 | 2858 | 1818 | 1450 | 0.96 | 0.043 | 0.048 |
| AKI stage >= 2, 48 h (L24) | NEWS | 0.653 | 0.202 | 2858 | 1921 | 1612 | 0.96 | 0.050 | 0.047 |
| AKI stage >= 2, 48 h (L24) | LightGBM | 0.859 | 0.202 | 2858 | 1240 | 736 | 0.74 | 0.044 | 0.044 |
| AKI stage >= 2, 48 h (L24, eICU) | Twin | 0.845 | 0.210 | 2727 | 1274 | 737 | 0.69 | 0.045 | 0.049 |
| AKI stage >= 2, 48 h (L24, eICU) | APACHE IVa | 0.578 | 0.210 | 3047 | 2541 | 2126 | 0.99 | 0.054 | 0.056 |
| AKI stage >= 2, 48 h (L24, eICU) | LightGBM | 0.825 | 0.210 | 2727 | 1298 | 714 | 0.77 | 0.051 | 0.045 |

Relative risk 0.7 (sizes in patients, both arms; 80% power, two-sided 0.05):

| Endpoint | Score | AUROC | Event rate | N, all | N, top 50% | N, top 25% | N adjusted / unadjusted | Type I (unadjusted) | Type I (adjusted) |
|---|---|---|---|---|---|---|---|---|---|
| ICU death (L24x) | Twin | 0.878 | 0.338 | 624 | 229 | 142 | 0.68 | 0.051 | 0.051 |
| ICU death (L24x) | SOFA | 0.733 | 0.338 | 624 | 373 | 215 | 0.83 | 0.053 | 0.056 |
| ICU death (L24x) | SAPS II | 0.774 | 0.338 | 624 | 311 | 172 | 0.81 | 0.053 | 0.046 |
| ICU death (L24x) | OASIS | 0.750 | 0.338 | 624 | 337 | 190 | 0.87 | 0.051 | 0.046 |
| ICU death (L24x) | NEWS | 0.758 | 0.338 | 624 | 337 | 183 | 0.84 | 0.049 | 0.049 |
| ICU death (L24x) | LightGBM | 0.882 | 0.338 | 624 | 236 | 136 | 0.65 | 0.046 | 0.054 |
| ICU death (L24x, eICU) | Twin | 0.876 | 0.226 | 1057 | 437 | 225 | 0.70 | 0.042 | 0.043 |
| ICU death (L24x, eICU) | APACHE IVa | 0.756 | 0.226 | 969 | 561 | 306 | 0.83 | 0.046 | 0.038 |
| ICU death (L24x, eICU) | LightGBM | 0.845 | 0.226 | 1057 | 437 | 264 | 0.78 | 0.052 | 0.058 |
| In-hospital death (L24x) | Twin | 0.883 | 0.494 | 345 | 119 | 72 | 0.64 | 0.056 | 0.049 |
| In-hospital death (L24x) | SOFA | 0.690 | 0.494 | 345 | 217 | 148 | 0.91 | 0.048 | 0.053 |
| In-hospital death (L24x) | SAPS II | 0.750 | 0.494 | 345 | 182 | 114 | 0.84 | 0.043 | 0.049 |
| In-hospital death (L24x) | OASIS | 0.709 | 0.494 | 345 | 209 | 124 | 0.90 | 0.046 | 0.051 |
| In-hospital death (L24x) | NEWS | 0.715 | 0.494 | 345 | 203 | 139 | 0.89 | 0.044 | 0.056 |
| In-hospital death (L24x) | LightGBM | 0.858 | 0.494 | 345 | 137 | 78 | 0.71 | 0.049 | 0.051 |
| In-hospital death (L24x, eICU) | Twin | 0.874 | 0.464 | 384 | 135 | 112 | 0.69 | 0.051 | 0.059 |
| In-hospital death (L24x, eICU) | APACHE IVa | 0.711 | 0.464 | 377 | 235 | 166 | 0.91 | 0.052 | 0.051 |
| In-hospital death (L24x, eICU) | LightGBM | 0.825 | 0.464 | 384 | 147 | 112 | 0.77 | 0.044 | 0.046 |
| New IMV, 24 h (L24) | Twin | 0.838 | 0.031 | 9325 | 5174 | 2786 | 0.84 | 0.052 | 0.051 |
| New IMV, 24 h (L24) | SOFA | 0.530 | 0.031 | 9307 | 8395 | 8991 | 1.01 | 0.055 | 0.057 |
| New IMV, 24 h (L24) | SAPS II | 0.539 | 0.031 | 9325 | 8227 | 6770 | 1.00 | 0.052 | 0.050 |
| New IMV, 24 h (L24) | OASIS | 0.584 | 0.031 | 9325 | 7148 | 7290 | 1.01 | 0.045 | 0.046 |
| New IMV, 24 h (L24) | NEWS | 0.709 | 0.031 | 9325 | 6208 | 4174 | 0.98 | 0.053 | 0.052 |
| New IMV, 24 h (L24) | LightGBM | 0.829 | 0.031 | 9325 | 4965 | 3141 | 0.95 | 0.054 | 0.056 |
| New IMV, 24 h (L24, eICU) | Twin | 0.869 | 0.025 | 11665 | 7196 | 3489 | 0.73 | 0.047 | 0.044 |
| New IMV, 24 h (L24, eICU) | APACHE IVa | 0.788 | 0.025 | 10486 | 5116 | 3219 | 1.00 | 0.056 | 0.059 |
| New IMV, 24 h (L24, eICU) | LightGBM | 0.868 | 0.025 | 11665 | 5706 | 3489 | 0.93 | 0.048 | 0.055 |
| New vasopressor, 24 h (L24) | Twin | 0.880 | 0.045 | 6391 | 3256 | 1689 | 0.84 | 0.054 | 0.055 |
| New vasopressor, 24 h (L24) | SOFA | 0.675 | 0.045 | 6380 | 4470 | 3089 | 0.98 | 0.052 | 0.051 |
| New vasopressor, 24 h (L24) | SAPS II | 0.655 | 0.045 | 6391 | 4426 | 3200 | 1.01 | 0.051 | 0.052 |
| New vasopressor, 24 h (L24) | OASIS | 0.608 | 0.045 | 6391 | 4867 | 4013 | 0.99 | 0.058 | 0.058 |
| New vasopressor, 24 h (L24) | NEWS | 0.614 | 0.045 | 6391 | 4692 | 4280 | 1.00 | 0.064 | 0.061 |
| New vasopressor, 24 h (L24) | LightGBM | 0.806 | 0.045 | 6391 | 3466 | 2219 | 0.91 | 0.051 | 0.051 |
| New vasopressor, 24 h (L24, eICU) | Twin | 0.991 | 0.015 | 19485 | 9615 | 4702 | 0.59 | 0.049 | 0.079 |
| New vasopressor, 24 h (L24, eICU) | APACHE IVa | 0.791 | 0.015 | 18327 | 10895 | 5320 | 0.96 | 0.057 | 0.054 |
| New vasopressor, 24 h (L24, eICU) | LightGBM | 0.925 | 0.015 | 19485 | 9615 | 4702 | 0.84 | 0.060 | 0.064 |
| New RRT, 48 h (L24) | Twin | 0.939 | 0.028 | 10508 | 5366 | 2587 | 0.76 | 0.057 | 0.054 |
| New RRT, 48 h (L24) | SOFA | 0.816 | 0.028 | 10493 | 5690 | 3573 | 0.93 | 0.053 | 0.054 |
| New RRT, 48 h (L24) | SAPS II | 0.757 | 0.028 | 10508 | 6402 | 4032 | 0.99 | 0.043 | 0.048 |
| New RRT, 48 h (L24) | OASIS | 0.680 | 0.028 | 10508 | 6870 | 4610 | 0.99 | 0.053 | 0.056 |
| New RRT, 48 h (L24) | NEWS | 0.646 | 0.028 | 10508 | 8038 | 6069 | 1.00 | 0.047 | 0.047 |
| New RRT, 48 h (L24) | LightGBM | 0.922 | 0.028 | 10508 | 5244 | 2793 | 0.79 | 0.054 | 0.066 |
| New RRT, 48 h (L24, eICU) | Twin | 0.990 | 0.012 | 24109 | 11952 | 5874 | 0.71 | 0.050 | 0.072 |
| New RRT, 48 h (L24, eICU) | APACHE IVa | 0.638 | 0.012 | 23348 | 14497 | 29397 | 1.00 | 0.049 | 0.050 |
| New RRT, 48 h (L24, eICU) | LightGBM | 0.937 | 0.012 | 24109 | 11952 | 5874 | 0.91 | 0.041 | 0.052 |
| AKI stage >= 2, 48 h (L24) | Twin | 0.891 | 0.202 | 1213 | 525 | 265 | 0.66 | 0.051 | 0.050 |
| AKI stage >= 2, 48 h (L24) | SOFA | 0.701 | 0.202 | 1211 | 767 | 533 | 0.91 | 0.046 | 0.046 |
| AKI stage >= 2, 48 h (L24) | SAPS II | 0.680 | 0.202 | 1213 | 774 | 608 | 0.97 | 0.051 | 0.051 |
| AKI stage >= 2, 48 h (L24) | OASIS | 0.674 | 0.202 | 1213 | 787 | 615 | 0.96 | 0.047 | 0.052 |
| AKI stage >= 2, 48 h (L24) | NEWS | 0.653 | 0.202 | 1213 | 812 | 690 | 0.98 | 0.053 | 0.051 |
| AKI stage >= 2, 48 h (L24) | LightGBM | 0.859 | 0.202 | 1213 | 534 | 322 | 0.76 | 0.051 | 0.057 |
| AKI stage >= 2, 48 h (L24, eICU) | Twin | 0.845 | 0.210 | 1158 | 548 | 323 | 0.76 | 0.053 | 0.043 |
| AKI stage >= 2, 48 h (L24, eICU) | APACHE IVa | 0.578 | 0.210 | 1293 | 1047 | 906 | 1.00 | 0.050 | 0.048 |
| AKI stage >= 2, 48 h (L24, eICU) | LightGBM | 0.825 | 0.210 | 1158 | 558 | 313 | 0.84 | 0.045 | 0.051 |

Annotations: L24x is F1's case-cohort sample (deaths enriched), so its event rates and sizes are not population values; the scores' comparison holds within it. L24's anchors are a natural sample. The eICU rows restrict to eICU's anchors (APACHE IVa's own data). Type I errors outside 0.04-0.06 mostly come from Monte Carlo noise (2,000 trials: SE 0.005) and from the sparse eICU subsets, where a near-perfect score strains the Wald test.

### V5. Teaching cases (`runs/toolkit/v5/`)

| Objective | Cases | Candidates | Meeting the objective | Self-contained pages |
|---|---|---|---|---|
| shock | 3 | 40 | 3 | yes |
| respiratory | 3 | 40 | 3 | yes |
| renal | 3 | 40 | 3 | yes |

Novelty: not computed on this host: the twins passed the bank's production novelty filter.

## 16a. Transfer to two databases never seen in training (RICD, zigongICU)

Amendment 16a. The databases were harmonized with new adapters and encoded with MUC32_main's frozen vocabulary; prompts carry the unknown database and site tokens. The pre-run gate (`docs/paper/external_gate.md`, recorded before any future was drawn) fixed the anchors and the endpoints scored: rolling anchors R (998 kept of 1000) and the 24-hour landmark's case-cohort sample L24e (1400), 128 futures per anchor.

The transferred models' refit (A1's recipe and seed, westc) reproduces A1's stored predictions on A1's test anchors: LightGBM exactly (largest difference 0.000); logistic regression to the solver's numerics (AUROC within 0.0061 on every program, median 0.0003).

### Rolling anchors (R)

| Endpoint | Database | Anchors | Positives | Didymos estimate | Didymos AUROC (95% CI) | Didymos O/E | Didymos AUROC, internal (follow-up) | LightGBM, transferred | Logistic, transferred | TabPFN-3.5, transferred | Didymos probe | LightGBM, local (cross-fitted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | RICD | 419 | 43 | sampled | 0.690 (0.545 to 0.929) | 0.19 | 0.888 | 0.733 | 0.703 | 0.748 | 0.358 | 0.880 |
| Decompensation, 24 h | RICD | 500 | 103 | hazard (E2) | 0.934 (0.897 to 0.963) | 1.23 | 0.960 | 0.878 | 0.833 | 0.908 | 0.906 | 0.935 |
| Hyperkalaemia (top bin), 24 h | RICD | 435 | 26 | sampled | 0.536 (0.438 to 0.748) | 0.22 | 0.831 | 0.481 | 0.549 | 0.650 | 0.426 | 0.850 |
| New vasopressor, 24 h | RICD | 490 | 31 | hazard (E2) | 0.754 (0.665 to 0.828) | 0.04 | 0.903 | 0.707 | 0.602 | 0.670 | 0.671 | 0.771 |
| Decompensation, 24 h | zigongICU | 498 | 171 | hazard (E2) | 0.829 (0.784 to 0.871) | 0.84 | 0.960 | 0.798 | 0.728 | 0.836 | 0.782 | 0.833 |
| Hyperkalaemia (top bin), 24 h | zigongICU | 455 | 23 | sampled | 0.784 (0.637 to 0.880) | 0.32 | 0.831 | 0.791 | 0.779 | 0.823 | 0.645 | 0.800 |
| Thrombocytopenia, 48 h | zigongICU | 448 | 21 | sampled | 0.826 (0.724 to 0.911) | 0.31 | 0.932 | 0.671 | 0.513 | 0.727 | 0.730 | 0.894 |
| New vasopressor, 24 h | zigongICU | 445 | 56 | hazard (E2) | 0.645 (0.553 to 0.735) | 0.24 | 0.903 | 0.745 | 0.599 | 0.734 | 0.570 | 0.836 |

Paired differences (Didymos minus each; 95% CI) and the best clinical score:

| Endpoint | Database | minus LightGBM, transferred | minus LightGBM, local | minus TabPFN-3.5 | Best clinical score (AUROC; the sampled estimate minus it) |
|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | RICD | -0.043 (-0.254 to 0.218) | -0.190 (-0.385 to 0.052) | -0.058 (-0.185 to 0.100) | NEWS: 0.809; sampled minus it -0.119 (-0.253 to 0.131) |
| Decompensation, 24 h | RICD | 0.056 (0.013 to 0.109) | -0.001 (-0.034 to 0.028) | 0.027 (0.004 to 0.052) | RICD's recorded SOFA: 0.911; sampled minus it 0.036 (-0.009 to 0.091) |
| Hyperkalaemia (top bin), 24 h | RICD | 0.055 (-0.044 to 0.176) | -0.314 (-0.420 to -0.070) | -0.114 (-0.203 to 0.058) | MEWS: 0.723; sampled minus it -0.187 (-0.284 to -0.030) |
| New vasopressor, 24 h | RICD | 0.047 (-0.060 to 0.170) | -0.017 (-0.139 to 0.116) | 0.084 (-0.008 to 0.192) | SOFA: 0.880; sampled minus it -0.093 (-0.199 to 0.006) |
| Decompensation, 24 h | zigongICU | 0.031 (-0.008 to 0.072) | -0.004 (-0.041 to 0.035) | -0.007 (-0.039 to 0.028) | MEWS: 0.680; sampled minus it 0.147 (0.088 to 0.205) |
| Hyperkalaemia (top bin), 24 h | zigongICU | -0.008 (-0.134 to 0.103) | -0.016 (-0.127 to 0.094) | -0.040 (-0.159 to 0.053) | SOFA: 0.681; sampled minus it 0.102 (-0.109 to 0.357) |
| Thrombocytopenia, 48 h | zigongICU | 0.155 (-0.022 to 0.281) | -0.068 (-0.187 to 0.047) | 0.099 (-0.056 to 0.211) | SOFA: 0.677; sampled minus it 0.153 (-0.050 to 0.332) |
| New vasopressor, 24 h | zigongICU | -0.100 (-0.217 to 0.026) | -0.191 (-0.272 to -0.099) | -0.088 (-0.202 to 0.020) | MEWS: 0.687; sampled minus it 0.023 (-0.103 to 0.148) |

Calibration before and after a logistic recalibration within the database (cross-fitted by patient, 5 folds):

| Endpoint | Database | Didymos O/E | after recalibration | Didymos AUROC after | LightGBM (transferred) O/E | after recalibration |
|---|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | RICD | 0.19 | 0.96 | 0.583 | 0.49 | 0.96 |
| Decompensation, 24 h | RICD | 1.23 | 0.93 | 0.931 | 1.36 | 0.96 |
| Hyperkalaemia (top bin), 24 h | RICD | 0.22 | 0.99 | 0.357 | 0.38 | 1.00 |
| New vasopressor, 24 h | RICD | 0.04 | 0.98 | 0.729 | 0.23 | 1.00 |
| Decompensation, 24 h | zigongICU | 0.84 | 1.00 | 0.825 | 2.09 | 1.00 |
| Hyperkalaemia (top bin), 24 h | zigongICU | 0.32 | 0.98 | 0.767 | 0.60 | 0.99 |
| Thrombocytopenia, 48 h | zigongICU | 0.31 | 0.98 | 0.773 | 0.70 | 0.98 |
| New vasopressor, 24 h | zigongICU | 0.24 | 1.01 | 0.626 | 1.57 | 1.00 |

### The 24-hour landmark, case-cohort (L24e)

| Endpoint | Database | Anchors | Positives | Didymos estimate | Didymos AUROC (95% CI) | Didymos O/E | Didymos AUROC, internal (follow-up) | LightGBM, transferred | Logistic, transferred | TabPFN-3.5, transferred | Didymos probe | LightGBM, local (cross-fitted) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| In-hospital death | RICD | 700 | 350 | sampled | 0.650 (0.609 to 0.691) | 1.63 | 0.883 | 0.710 | 0.673 | 0.731 | 0.688 | 0.757 |
| ICU death | RICD | 700 | 289 | hazard (E2) | 0.686 (0.647 to 0.725) | 1.40 | 0.924 | 0.671 | 0.659 | 0.711 | 0.698 | 0.775 |
| ICU stay > 7 days | RICD | 700 | 507 | sampled | 0.569 (0.508 to 0.628) | 3.46 | 0.864 | 0.543 | 0.456 | 0.530 | 0.557 | 0.787 |
| Decompensation, 24 h | zigongICU | 700 | 77 | hazard (E2) | 0.796 (0.743 to 0.847) | 0.29 | 0.982 | 0.733 | 0.636 | 0.778 | 0.781 | 0.679 |
| In-hospital death | zigongICU | 700 | 350 | sampled | 0.717 (0.681 to 0.754) | 0.67 | 0.883 | 0.696 | 0.599 | 0.721 | 0.738 | 0.710 |
| ICU death | zigongICU | 700 | 274 | hazard (E2) | 0.765 (0.730 to 0.801) | 0.55 | 0.924 | 0.721 | 0.548 | 0.755 | 0.765 | 0.785 |
| ICU stay > 7 days | zigongICU | 700 | 184 | sampled | 0.603 (0.554 to 0.653) | 1.36 | 0.864 | 0.676 | 0.413 | 0.671 | 0.642 | 0.704 |

Paired differences (Didymos minus each; 95% CI) and the best clinical score:

| Endpoint | Database | minus LightGBM, transferred | minus LightGBM, local | minus TabPFN-3.5 | Best clinical score (AUROC; the sampled estimate minus it) |
|---|---|---|---|---|---|
| In-hospital death | RICD | -0.060 (-0.109 to -0.012) | -0.107 (-0.148 to -0.066) | -0.081 (-0.124 to -0.037) | RICD's recorded SOFA: 0.714; sampled minus it -0.054 (-0.102 to -0.007) |
| ICU death | RICD | 0.015 (-0.037 to 0.066) | -0.089 (-0.128 to -0.053) | -0.025 (-0.075 to 0.022) | RICD's recorded SOFA: 0.752; sampled minus it -0.065 (-0.114 to -0.020) |
| ICU stay > 7 days | RICD | 0.027 (-0.064 to 0.110) | -0.218 (-0.282 to -0.149) | 0.040 (-0.051 to 0.122) | OASIS: 0.665; sampled minus it -0.095 (-0.160 to -0.027) |
| Decompensation, 24 h | zigongICU | 0.064 (0.020 to 0.115) | 0.117 (0.049 to 0.188) | 0.019 (-0.019 to 0.056) | SAPS II: 0.727; sampled minus it 0.067 (0.003 to 0.137) |
| In-hospital death | zigongICU | 0.022 (-0.011 to 0.053) | 0.008 (-0.027 to 0.048) | -0.004 (-0.030 to 0.022) | SAPS II: 0.664; sampled minus it 0.054 (0.012 to 0.095) |
| ICU death | zigongICU | 0.044 (0.011 to 0.075) | -0.020 (-0.052 to 0.014) | 0.010 (-0.014 to 0.036) | SAPS II: 0.677; sampled minus it 0.085 (0.041 to 0.126) |
| ICU stay > 7 days | zigongICU | -0.072 (-0.130 to -0.012) | -0.101 (-0.160 to -0.044) | -0.068 (-0.129 to -0.008) | OASIS: 0.593; sampled minus it 0.011 (-0.061 to 0.086) |

Calibration before and after a logistic recalibration within the database (cross-fitted by patient, 5 folds):

| Endpoint | Database | Didymos O/E | after recalibration | Didymos AUROC after | LightGBM (transferred) O/E | after recalibration |
|---|---|---|---|---|---|---|
| In-hospital death | RICD | 1.63 | 0.99 | 0.638 | 2.28 | 0.99 |
| ICU death | RICD | 1.40 | 0.99 | 0.680 | 3.09 | 1.00 |
| ICU stay > 7 days | RICD | 3.46 | 1.00 | 0.549 | 4.14 | 1.00 |
| Decompensation, 24 h | zigongICU | 0.29 | 1.00 | 0.787 | 4.41 | 0.99 |
| In-hospital death | zigongICU | 0.67 | 1.00 | 0.716 | 2.02 | 1.00 |
| ICU death | zigongICU | 0.55 | 1.00 | 0.761 | 4.29 | 1.00 |
| ICU stay > 7 days | zigongICU | 1.36 | 1.00 | 0.592 | 1.35 | 1.00 |

Annotations: weighted metrics (R: the design's inclusion weights; L24e: the case-cohort weights); intervals from 1,000 resamples of stays, paired differences from the same resamples. Didymos's estimate is the hazard estimate (E2) for death within 24 h, new vasopressor and ICU death, the sampled estimate otherwise; the internal AUROC is the same estimator on the follow-up anchors of the seven databases (R128, L24x), shown for reference. Transferred models were trained on the seven databases with A1's recipe (a new database's one-hot columns are all zero); the local LightGBM was trained on the database's own stays, cross-fitted by patient. Endpoints not listed are not defined in that database (the gate's rule).

## 16b. The 48-hour landmark with the follow-up's settings (L48x)

Anchors: 809 of A1's test_L48 (455 in-hospital deaths and 354 controls weighted 8.60: a case-cohort sample, seed 16); left out as continued context windows: 0. 128 futures per anchor at four times A1's caps.

### Rollout count to 128 (as F1)

### L48x (`runs/paper/f1/L48x`)

809 anchors x 128 rollouts; 11.2% end at the token cap.

| Endpoint | Anchors | Positives | Twin, 128 rollouts | Twin, A1's 32 | LightGBM | TabPFN-3.5 | Twin probe | 128 - A1's 32 (95% CI) | 128 - LightGBM | 128 - TabPFN-3.5 | O/E (128) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | 505 | 127 | 0.912 | 0.884 | 0.915 | 0.925 | 0.919 | 0.029 (0.006 to 0.053) | -0.003 (-0.026 to 0.021) | -0.012 (-0.034 to 0.009) | 1.09 |
| In-hospital death | 809 | 455 | 0.861 | 0.840 | 0.861 | 0.864 | 0.857 | 0.021 (0.005 to 0.037) | -0.001 (-0.020 to 0.016) | -0.004 (-0.021 to 0.013) | 1.20 |
| ICU stay > 7 days | 796 | 219 | 0.860 | 0.780 | 0.837 | 0.841 | 0.816 | 0.080 (0.040 to 0.127) | 0.023 (-0.007 to 0.051) | 0.019 (-0.015 to 0.051) | 0.92 |

AUROC against the number of rollouts (E0; the mean over 20 random subsets of each size):

| Endpoint | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| ICU death | 0.641 | 0.685 | 0.738 | 0.797 | 0.838 | 0.868 | 0.881 | 0.889 |
| In-hospital death | 0.626 | 0.683 | 0.741 | 0.789 | 0.821 | 0.843 | 0.854 | 0.861 |
| ICU stay > 7 days | 0.662 | 0.733 | 0.777 | 0.814 | 0.839 | 0.848 | 0.853 | 0.856 |

Annotations: the same anchors throughout; A1's 32-rollout Twin is an independent replicate (other seeds) of the same estimator. Weighted metrics (R: the design's inclusion weights; L24x: the case-cohort weights).

### Estimators (as F2)

### L48x

| Endpoint | Anchors | Positives | AUROC E0 | E1 | E2 | O/E E0 | E1 | E2 | E2 - E0 (95% CI) | E2 - LightGBM | E2 - TabPFN-3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 808 | 57 | 0.946 | 0.946 | 0.948 | 1.30 | 1.30 | 1.28 | 0.002 (-0.013 to 0.016) | 0.033 (0.005 to 0.066) | - |
| New IMV, 24 h | 509 | 38 | 0.845 | 0.844 | 0.847 | 0.66 | 0.66 | 0.66 | 0.002 (-0.034 to 0.034) | 0.164 (0.038 to 0.329) | - |
| New vasopressor, 24 h | 598 | 59 | 0.938 | 0.938 | 0.938 | 0.80 | 0.80 | 0.80 | -0.001 (-0.007 to 0.006) | 0.101 (0.041 to 0.166) | - |
| New RRT, 48 h | 758 | 37 | 0.940 | 0.940 | 0.946 | 0.53 | 0.53 | 0.52 | 0.007 (-0.004 to 0.022) | -0.008 (-0.047 to 0.027) | - |
| ICU death | 505 | 127 | 0.912 | 0.911 | 0.910 | 1.11 | 1.14 | 1.13 | -0.003 (-0.010 to 0.005) | -0.005 (-0.027 to 0.018) | -0.015 (-0.037 to 0.006) |
| ICU death, at A1's cap | 505 | 127 | 0.902 | 0.897 | 0.902 | 1.24 | 1.58 | 1.55 | - | - | - |

AUROC against the number of rollouts, E0 and E2 on the same random subsets:

| Endpoint | Estimator | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | E0 (A1's share) | 0.652 | 0.701 | 0.759 | 0.810 | 0.847 | 0.883 | 0.922 | 0.946 |
| Decompensation, 24 h | E2 (hybrid) | 0.888 | 0.896 | 0.913 | 0.920 | 0.929 | 0.936 | 0.944 | 0.948 |
| New IMV, 24 h | E0 (A1's share) | 0.656 | 0.731 | 0.790 | 0.808 | 0.831 | 0.830 | 0.840 | 0.845 |
| New IMV, 24 h | E2 (hybrid) | 0.825 | 0.848 | 0.854 | 0.850 | 0.853 | 0.846 | 0.846 | 0.847 |
| New vasopressor, 24 h | E0 (A1's share) | 0.657 | 0.741 | 0.806 | 0.866 | 0.911 | 0.930 | 0.936 | 0.938 |
| New vasopressor, 24 h | E2 (hybrid) | 0.889 | 0.910 | 0.917 | 0.926 | 0.931 | 0.936 | 0.936 | 0.938 |
| New RRT, 48 h | E0 (A1's share) | 0.669 | 0.750 | 0.791 | 0.858 | 0.883 | 0.910 | 0.932 | 0.940 |
| New RRT, 48 h | E2 (hybrid) | 0.912 | 0.930 | 0.943 | 0.945 | 0.943 | 0.946 | 0.946 | 0.946 |
| ICU death | E0 (A1's share) | 0.651 | 0.694 | 0.749 | 0.806 | 0.842 | 0.881 | 0.900 | 0.912 |
| ICU death | E2 (hybrid) | 0.843 | 0.856 | 0.871 | 0.881 | 0.892 | 0.901 | 0.905 | 0.910 |

The self-check (sampled events against summed hazards within the rollouts; 1 expected):

| Endpoint | Sampled events | Summed hazards | Ratio |
|---|---|---|---|
| Decompensation, 24 h | 15626 | 15647.6 | 0.999 |
| New IMV, 24 h | 44224 | 44294.9 | 0.998 |
| New vasopressor, 24 h | 24363 | 24274.0 | 1.004 |
| New RRT, 48 h | 12880 | 12852.5 | 1.002 |
| ICU death | 15626 | 15647.6 | 0.999 |

### Mortality (as F3)

Minutes per token (L48x): rollouts 3.52, real futures 3.84 (medians).

### D2. Generation drift (the same anchors' rollouts against their real futures)

| Subset | Endpoint | Observed | Real futures' summed hazards | Rollouts' summed hazards (mean) | Drift ratio |
|---|---|---|---|---|---|
| L48x | Decompensation, 24 h | 57.0 | 58.4 | 44.3 | 0.76 |
| L48x | New IMV, 24 h | 114.0 | 145.9 | 171.6 | 1.18 |
| L48x | New vasopressor, 24 h | 195.8 | 213.0 | 244.0 | 1.15 |
| L48x | New RRT, 48 h | 89.2 | 144.8 | 171.4 | 1.18 |
| L48x | ICU death | 127.0 | 160.4 | 109.2 | 0.68 |

Hourly hazard of the rollouts over the real futures' (paths still at risk), by hours after the anchor:

| Subset | Cause | 0-6 h | 6-12 h | 12-24 h | 24-48 h |
|---|---|---|---|---|---|
| L48x | death | 0.62 | 0.44 | 0.53 | 0.42 |
| L48x | alive discharge | 1.13 | 0.99 | 0.90 | 0.92 |

### D3. Capped rollouts by the real outcome (mean share per anchor)

| Subset | Endpoint | Real outcome 0 | Real outcome 1 |
|---|---|---|---|
| L48x | In-hospital death | 0.050 | 0.135 |
| L48x | ICU death | 0.022 | 0.050 |
| L48x | Decompensation, 24 h | 0.098 | 0.088 |

### Every comparator on L48x (`runs/paper/a16/comparators_L48x.json`)

| Endpoint | Anchors | Positives | Didymos, 128 (sampled) | Didymos E2 | LightGBM | TabPFN-3.5 | Didymos probe | Best of the 8 learners | Best clinical score |
|---|---|---|---|---|---|---|---|---|---|
| ICU death | 505 | 127 | 0.912 | 0.910 | 0.915 (-0.003) | 0.925 (-0.012) | 0.919 (-0.007) | Elastic-net logistic: 0.917 (-0.005) | SAPS II: 0.812 (0.101) |
| In-hospital death | 809 | 455 | 0.861 | - | 0.861 (-0.001) | 0.864 (-0.004) | 0.857 (0.004) | Elastic-net logistic: 0.840 (0.020) | SAPS II: 0.754 (0.107) |
| ICU stay > 7 days | 796 | 219 | 0.860 | - | 0.837 (0.023) | 0.841 (0.019) | 0.816 (0.044) | Elastic-net logistic: 0.833 (0.027) | OASIS: 0.673 (0.187) |
| Decompensation, 24 h | 808 | 57 | 0.946 | 0.948 | 0.915 (0.031) | 0.943 (0.003) | 0.928 (0.018) | Elastic-net logistic: 0.940 (0.006) | NEWS: 0.869 (0.077) |
| New IMV, 24 h | 509 | 38 | 0.859 | 0.847 | 0.683 (0.177) | - | 0.704 (0.155) | - | NEWS: 0.736 (0.123) |
| New vasopressor, 24 h | 597 | 59 | 0.940 | 0.938 | 0.837 (0.103) | - | 0.844 (0.097) | - | SOFA: 0.699 (0.242) |
| New RRT, 48 h | 758 | 37 | 0.941 | 0.946 | 0.954 (-0.013) | - | 0.936 (0.005) | - | SOFA: 0.877 (0.064) |

AUROC (the sampled 128-future estimate minus it, paired). TabPFN-3.5 and the eight learners of amendment 10 for death within 24 h were fitted for this subset with their recipes; the other cells are A1's stored predictions, reindexed.

Annotations: as F1-F3 at L24x; the comparators are A1's stored predictions on the same anchors unless stated.

## 16c. Attention knockout against the generator (`runs/paper/f4r/c2b.json`)

F4's 160 anchors and 2,698 prompts drawn again (48 futures each, seed 16, F4's caps); every group's effect recomputed as F4 did. Knockout (amendment 9's mask and score) on all 160 anchors.

| Endpoint | Anchors | Ceiling (Spearman-Brown) | Editing: rho / top-3 | Knockout: rho / top-3 | Sign agreement, editing | Knockout | F4's editing rho (first draw) |
|---|---|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | 40 | 0.58 | 0.13 / 0.32 | 0.19 / 0.31 | 0.63 (n=101) | 0.68 (n=100) | 0.20 |
| Decompensation, 24 h | 40 | 0.50 | 0.13 / 0.28 | 0.23 / 0.33 | 0.62 (n=68) | 0.66 (n=68) | 0.19 |
| Hypotension, 4 h | 40 | 0.34 | 0.09 / 0.31 | 0.06 / 0.29 | 0.73 (n=41) | 0.63 (n=41) | 0.07 |
| New IMV, 24 h | 40 | 0.54 | 0.22 / 0.29 | 0.06 / 0.31 | 0.67 (n=70) | 0.54 (n=70) | 0.12 |

Per anchor on the groups both score (median rho with the redrawn effects):

| Endpoint | Anchors | Knockout | Editing | Knockout minus editing |
|---|---|---|---|---|
| AKI stage >= 2, 48 h | 40 | 0.19 | 0.13 | 0.05 |
| Decompensation, 24 h | 40 | 0.23 | 0.13 | 0.01 |
| Hypotension, 4 h | 38 | 0.06 | 0.09 | 0.02 |
| New IMV, 24 h | 40 | 0.06 | 0.22 | -0.10 |

Annotations: agreement is bounded by the reliability ceiling (split halves of the 48 futures, Spearman-Brown corrected). The statistics (family 20) are in `docs/paper/tables/manuscript/stats_family20.csv`.

## 17. Adapting Didymos to RICD and zigongICU

Amendment 17 (2026-10-12, the study lead's request; fixed before any adapted model was trained or any new future drawn). Each database's patients split once (seed 17): TEST 50% (locked; amendment 16a's anchors there), FT 35% (unlabelled continued training) and LABEL 15% (the few-label methods). Labels are amendment 16a's for every arm.

| Database | Patients | Stays |
|---|---|---|
| RICD, TEST | 2015 | 2646 |
| RICD, FT | 1388 | 1859 |
| RICD, LABEL | 645 | 877 |
| zigongICU, TEST | 1384 | 1460 |
| zigongICU, FT | 927 | 973 |
| zigongICU, LABEL | 431 | 466 |

### The improved encoding (Z1's back-off)

| Database | Domain | Events to OOV, 16a (%) | after the back-off (%) |
|---|---|---|---|
| RICD | ADT | 12.7 | 0.0 |
| RICD | MED_CON | 85.1 | 0.0 |
| RICD | MED_INT | 31.6 | 2.9 |
| RICD | OUTPUT | 14.5 | 0.0 |
| zigongICU | ADT | 0.1 | 0.0 |
| zigongICU | MED_CON | 55.5 | 0.6 |
| zigongICU | MED_INT | 6.5 | 1.2 |

### F1: the grid (next-token loss, nats per token)

MUC32_main before adaptation: held-out FT loss 2.0995.

| Run | Learning rate | Passes over FT | Steps | Held-out FT loss | Seven databases' tuning loss, change (mean) | change (largest) |  |
|---|---|---|---|---|---|---|---|
| E17_lr3e-05_p1 | 3e-05 | 1 | 76 | 1.5298 | 0.0117 | 0.0186 |  |
| E17_lr0.0001_p1 | 0.0001 | 1 | 76 | 1.2936 | 0.0208 | 0.0379 |  |
| E17_lr3e-05_p3 | 3e-05 | 3 | 228 | 1.2909 | 0.0131 | 0.0266 |  |
| E17_lr0.0001_p3 | 0.0001 | 3 | 228 | 1.1305 | 0.0189 | 0.0327 | chosen |

### F1: the M7 gate

The M7 gate of the chosen run, P1 (membership by loss on the seven databases) at the registered size and at amendment 17c's larger size, TPR at 1% FPR against the 0.03 limit:

| Database | Per side | TPR (95% CI) | Per side, 17c | TPR, 17c (95% CI) | AUC, 17c |
|---|---|---|---|---|---|
| MIMIC-IV | 3000 | 0.0120 (0.0043 to 0.0197) | 10000 | 0.0117 (0.0085 to 0.0166) | 0.548 |
| AmsterdamUMCdb | 2747 | 0.0182 (0.0124 to 0.0266) | 2747 | 0.0226 (0.0153 to 0.0313) | 0.555 |
| eICU | 3000 | 0.0227 (0.0127 to 0.0303) | 10000 | 0.0157 (0.0125 to 0.0194) | 0.559 |
| HiRID | 3000 | 0.0197 (0.0133 to 0.0263) | 5011 | 0.0156 (0.0110 to 0.0224) | 0.554 |
| SICdb | 3000 | 0.0143 (0.0097 to 0.0207) | 4098 | 0.0137 (0.0093 to 0.0207) | 0.555 |
| NWICU | 3000 | 0.0107 (0.0050 to 0.0157) | 4100 | 0.0112 (0.0071 to 0.0173) | 0.556 |
| INSPIRE | 2072 | 0.0111 (0.0058 to 0.0203) | 2072 | 0.0125 (0.0058 to 0.0285) | 0.559 |
| Pooled | 19819 | 0.0151 (0.0124 to 0.0178) | 38028 | 0.0151 (0.0132 to 0.0169) | 0.544 |

P2-P5 as run: P2 PASS, P3 PASS, P4 PASS, P5 PASS; the gate INCONCLUSIVE after 17c (one group's upper bound above the limit); F1's futures were drawn under the study lead's exemption (amendment 17d: these futures only, deleted after scoring; no twins or other synthetic data from this run).

### F1: membership of its own FT stays

Membership of the FT stays against the TEST stays (loss attack, P1's method):

| Model | Database | FT stays | TEST stays | AUC (95% CI) | Mean loss, FT | Mean loss, TEST |
|---|---|---|---|---|---|---|
| E17_lr0.0001_p3 | RICD | 1666 | 1666 | 0.581 (0.562 to 0.599) | 1.4565 | 1.5238 |
| E17_lr0.0001_p3 | zigongICU | 867 | 867 | 0.542 (0.515 to 0.570) | 1.3270 | 1.3807 |
| E17_lr0.0001_p3 | pooled | 2533 | 2533 | 0.564 (0.548 to 0.579) | 1.4122 | 1.4748 |
| MUC32_main | RICD | 1666 | 1666 | 0.506 (0.487 to 0.524) | 2.7045 | 2.7101 |
| MUC32_main | zigongICU | 867 | 867 | 0.495 (0.469 to 0.524) | 2.7278 | 2.7274 |
| MUC32_main | pooled | 2533 | 2533 | 0.502 (0.486 to 0.517) | 2.7125 | 2.7160 |

### Rolling anchors (R): the TEST anchors

| Endpoint | Database | Anchors | Positives | LABEL positives | Z0: 16a | Z1: back-off | F1: adapted | F1+S: stacked | LightGBM, LABEL | LightGBM, FT+LABEL | TabPFN-3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | RICD | 232 | 51 | 58 | 0.898 | 0.883 | 0.898 (0.836 to 0.953) | 0.903 | 0.874 | 0.891 | 0.866 |
| New vasopressor, 24 h (descriptive) | RICD | 228 | 8 | 11 | 0.628 | 0.810 | 0.851 (0.709 to 0.958) | 0.854 | 0.709 | 0.826 | 0.549 |
| AKI stage >= 2, 48 h | RICD | 226 | 25 | 33 | 0.881 | 0.857 | 0.847 (0.727 to 0.917) | 0.880 | 0.900 | 0.756 | 0.848 |
| Hyperkalaemia (top bin), 24 h | RICD | 216 | 12 | 13 | 0.779 | 0.751 | 0.873 (0.782 to 0.959) | 0.898 | 0.501 | 0.708 | 0.752 |
| Decompensation, 24 h | zigongICU | 258 | 93 | 77 | 0.813 | 0.832 | 0.834 (0.771 to 0.888) | 0.841 | 0.802 | 0.847 | 0.848 |
| New vasopressor, 24 h | zigongICU | 228 | 28 | 37 | 0.620 | 0.644 | 0.745 (0.664 to 0.821) | 0.792 | 0.748 | 0.816 | 0.723 |
| Hyperkalaemia (top bin), 24 h (descriptive) | zigongICU | 235 | 9 | 21 | 0.702 | 0.742 | 0.851 (0.648 to 0.971) | 0.850 | 0.775 | 0.768 | 0.891 |
| Thrombocytopenia, 48 h | zigongICU | 233 | 10 | 10 | 0.836 | 0.793 | 0.902 (0.827 to 0.975) | 0.902 | 0.810 | 0.770 | 0.707 |

Paired differences in AUROC (95% CI):

| Endpoint | Database | Z1 minus Z0 | F1 minus Z0 | F1+R minus Z0 | F1+R minus LightGBM, FT+LABEL | F1+R minus TabPFN-3.5 |
|---|---|---|---|---|---|---|
| Decompensation, 24 h | RICD | -0.015 (-0.047 to 0.012) | 0.000 (-0.042 to 0.040) | 0.000 (-0.042 to 0.040) | 0.007 (-0.034 to 0.043) | 0.032 (-0.018 to 0.083) |
| New vasopressor, 24 h (descriptive) | RICD | 0.182 (0.020 to 0.395) | 0.223 (-0.003 to 0.473) | 0.223 (-0.003 to 0.473) | 0.025 (-0.158 to 0.205) | 0.302 (0.154 to 0.460) |
| AKI stage >= 2, 48 h | RICD | -0.026 (-0.062 to 0.026) | -0.029 (-0.116 to 0.116) | -0.029 (-0.116 to 0.116) | 0.091 (-0.007 to 0.201) | -0.001 (-0.086 to 0.155) |
| Hyperkalaemia (top bin), 24 h | RICD | -0.028 (-0.127 to 0.057) | 0.095 (0.016 to 0.213) | 0.095 (0.016 to 0.213) | 0.164 (0.027 to 0.291) | 0.120 (-0.038 to 0.258) |
| Decompensation, 24 h | zigongICU | 0.019 (-0.002 to 0.044) | 0.021 (-0.024 to 0.071) | 0.021 (-0.024 to 0.071) | -0.013 (-0.062 to 0.032) | -0.014 (-0.059 to 0.032) |
| New vasopressor, 24 h | zigongICU | 0.025 (-0.028 to 0.083) | 0.125 (0.005 to 0.233) | 0.125 (0.005 to 0.233) | -0.071 (-0.174 to 0.043) | 0.022 (-0.085 to 0.127) |
| Hyperkalaemia (top bin), 24 h (descriptive) | zigongICU | 0.040 (-0.010 to 0.127) | 0.149 (0.031 to 0.370) | 0.149 (0.031 to 0.370) | 0.083 (-0.020 to 0.306) | -0.041 (-0.176 to 0.046) |
| Thrombocytopenia, 48 h | zigongICU | -0.043 (-0.089 to 0.019) | 0.015 (-0.075 to 0.138) | 0.015 (-0.075 to 0.138) | 0.071 (-0.101 to 0.260) | 0.142 (0.022 to 0.231) |

Calibration (O/E) before and after recalibration on the LABEL anchors:

| Endpoint | Database | Z0 | Z1 | F1 | Z1+R | F1+R | F1+S | F1's map |
|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | RICD | 1.18 | 1.07 | 1.95 | 0.54 | 0.66 | 0.87 | intercept, slope |
| New vasopressor, 24 h (descriptive) | RICD | 0.02 | 0.02 | 0.22 | 0.72 | 0.41 | 0.19 | intercept, slope |
| AKI stage >= 2, 48 h | RICD | 0.15 | 0.15 | 0.45 | 0.35 | 0.35 | 0.34 | intercept, slope |
| Hyperkalaemia (top bin), 24 h | RICD | 0.13 | 0.14 | 0.34 | 0.33 | 0.28 | 0.26 | intercept, slope |
| Decompensation, 24 h | zigongICU | 0.88 | 0.89 | 2.34 | 1.23 | 1.16 | 1.16 | intercept, slope |
| New vasopressor, 24 h | zigongICU | 0.27 | 0.25 | 0.54 | 0.95 | 0.92 | 0.65 | intercept, slope |
| Hyperkalaemia (top bin), 24 h (descriptive) | zigongICU | 0.23 | 0.27 | 0.49 | 0.32 | 0.32 | 0.27 | intercept, slope |
| Thrombocytopenia, 48 h | zigongICU | 0.30 | 0.33 | 0.47 | 1.00 | 1.18 | 1.47 | intercept, slope |

### The 24-hour landmark, case-cohort (L24e): the TEST anchors

| Endpoint | Database | Anchors | Positives | LABEL positives | Z0: 16a | Z1: back-off | F1: adapted | F1+S: stacked | LightGBM, LABEL | LightGBM, FT+LABEL | TabPFN-3.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | RICD | 338 | 133 | 94 | 0.695 | 0.685 | 0.738 (0.678 to 0.792) | 0.744 | 0.711 | 0.768 | 0.701 |
| In-hospital death | RICD | 338 | 163 | 125 | 0.658 | 0.660 | 0.715 (0.659 to 0.767) | 0.719 | 0.684 | 0.718 | 0.680 |
| ICU stay > 7 days | RICD | 338 | 241 | 179 | 0.601 | 0.593 | 0.720 (0.644 to 0.792) | 0.789 | 0.772 | 0.795 | 0.553 |
| ICU death | zigongICU | 345 | 137 | 73 | 0.733 | 0.742 | 0.710 (0.651 to 0.770) | 0.730 | 0.695 | 0.752 | 0.741 |
| In-hospital death | zigongICU | 345 | 165 | 99 | 0.690 | 0.699 | 0.676 (0.616 to 0.733) | 0.692 | 0.646 | 0.710 | 0.722 |
| ICU stay > 7 days | zigongICU | 345 | 94 | 68 | 0.617 | 0.605 | 0.705 (0.634 to 0.769) | 0.706 | 0.588 | 0.676 | 0.649 |
| Decompensation, 24 h | zigongICU | 345 | 42 | 19 | 0.755 | 0.746 | 0.736 (0.657 to 0.809) | 0.735 | 0.637 | 0.725 | 0.750 |

Paired differences in AUROC (95% CI):

| Endpoint | Database | Z1 minus Z0 | F1 minus Z0 | F1+R minus Z0 | F1+R minus LightGBM, FT+LABEL | F1+R minus TabPFN-3.5 |
|---|---|---|---|---|---|---|
| ICU death | RICD | -0.010 (-0.042 to 0.022) | 0.043 (-0.011 to 0.092) | 0.043 (-0.011 to 0.092) | -0.031 (-0.074 to 0.012) | 0.036 (-0.020 to 0.090) |
| In-hospital death | RICD | 0.001 (-0.029 to 0.030) | 0.057 (-0.004 to 0.116) | 0.057 (-0.004 to 0.116) | -0.002 (-0.043 to 0.038) | 0.035 (-0.005 to 0.077) |
| ICU stay > 7 days | RICD | -0.008 (-0.056 to 0.034) | 0.119 (0.014 to 0.214) | 0.119 (0.014 to 0.214) | -0.075 (-0.142 to -0.009) | 0.167 (0.064 to 0.264) |
| ICU death | zigongICU | 0.008 (-0.006 to 0.022) | -0.024 (-0.057 to 0.009) | -0.024 (-0.057 to 0.009) | -0.043 (-0.091 to 0.006) | -0.031 (-0.084 to 0.019) |
| In-hospital death | zigongICU | 0.009 (-0.005 to 0.024) | -0.014 (-0.055 to 0.024) | -0.014 (-0.055 to 0.024) | -0.034 (-0.093 to 0.022) | -0.046 (-0.097 to 0.005) |
| ICU stay > 7 days | zigongICU | -0.013 (-0.049 to 0.029) | 0.088 (0.015 to 0.161) | 0.088 (0.015 to 0.161) | 0.030 (-0.033 to 0.102) | 0.057 (-0.007 to 0.118) |
| Decompensation, 24 h | zigongICU | -0.009 (-0.036 to 0.019) | -0.019 (-0.060 to 0.026) | -0.019 (-0.060 to 0.026) | 0.012 (-0.050 to 0.078) | -0.013 (-0.067 to 0.047) |

Calibration (O/E) before and after recalibration on the LABEL anchors:

| Endpoint | Database | Z0 | Z1 | F1 | Z1+R | F1+R | F1+S | F1's map |
|---|---|---|---|---|---|---|---|---|
| ICU death | RICD | 1.48 | 1.43 | 1.09 | 0.86 | 0.71 | 0.66 | intercept, slope |
| In-hospital death | RICD | 1.71 | 1.60 | 0.87 | 0.77 | 0.65 | 0.69 | intercept, slope |
| ICU stay > 7 days | RICD | 3.60 | 3.36 | 1.11 | 1.01 | 0.97 | 0.97 | intercept, slope |
| ICU death | zigongICU | 0.56 | 0.56 | 1.04 | 1.02 | 1.06 | 1.08 | intercept, slope |
| In-hospital death | zigongICU | 0.64 | 0.63 | 0.98 | 0.91 | 0.97 | 1.03 | intercept, slope |
| ICU stay > 7 days | zigongICU | 1.41 | 1.25 | 0.71 | 0.89 | 0.85 | 0.85 | intercept, slope |
| Decompensation, 24 h | zigongICU | 0.32 | 0.32 | 1.39 | 1.15 | 1.24 | 1.18 | intercept, slope |

### Label efficiency (LightGBM on n LABEL stays, median AUROC of 20 draws, against F1+R)

| Design | Endpoint | Database | F1+R | n=25 | n=50 | n=100 | n=200 | all LABEL | LightGBM reaches F1+R at n |
|---|---|---|---|---|---|---|---|---|---|
| R | Decompensation, 24 h | RICD | 0.898 | 0.684 | 0.704 | 0.745 | 0.776 | 0.874 (n=877) | not reached |
| R | New vasopressor, 24 h | RICD | 0.851 | 0.600 | 0.675 | 0.680 | 0.661 | 0.709 (n=877) | not reached |
| R | AKI stage >= 2, 48 h | RICD | 0.847 | 0.590 | 0.708 | 0.779 | 0.803 | 0.900 (n=877) | interpolated, 393 |
| R | Hyperkalaemia (top bin), 24 h | RICD | 0.873 | 0.689 | 0.624 | 0.615 | 0.643 | 0.501 (n=877) | not reached |
| R | Decompensation, 24 h | zigongICU | 0.834 | 0.643 | 0.708 | 0.733 | 0.772 | 0.802 (n=466) | not reached |
| R | New vasopressor, 24 h | zigongICU | 0.745 | 0.681 | 0.664 | 0.702 | 0.744 | 0.748 (n=466) | interpolated, 249 |
| R | Hyperkalaemia (top bin), 24 h | zigongICU | 0.851 | 0.659 | 0.668 | 0.720 | 0.706 | 0.775 (n=466) | not reached |
| R | Thrombocytopenia, 48 h | zigongICU | 0.902 | 0.669 | 0.580 | 0.562 | 0.646 | 0.810 (n=466) | not reached |
| L24e | ICU death | RICD | 0.738 | 0.500 | 0.500 | 0.677 | 0.676 | 0.711 (n=877) | not reached |
| L24e | In-hospital death | RICD | 0.715 | 0.500 | 0.500 | 0.635 | 0.654 | 0.684 (n=877) | not reached |
| L24e | ICU stay > 7 days | RICD | 0.720 | 0.500 | 0.500 | 0.669 | 0.709 | 0.772 (n=877) | interpolated, 257 |
| L24e | ICU death | zigongICU | 0.710 | 0.500 | 0.500 | 0.604 | 0.625 | 0.695 (n=466) | not reached |
| L24e | In-hospital death | zigongICU | 0.676 | 0.500 | 0.500 | 0.579 | 0.593 | 0.646 (n=466) | not reached |
| L24e | ICU stay > 7 days | zigongICU | 0.705 | 0.500 | 0.500 | 0.533 | 0.576 | 0.588 (n=466) | not reached |
| L24e | Decompensation, 24 h | zigongICU | 0.736 | 0.500 | 0.500 | 0.612 | 0.624 | 0.637 (n=466) | not reached |

### Ranks over the TEST pairs (family 24)

Mean ranks by AUROC over the 13 TEST pairs with at least 10 positives (every predictor):

| Predictor | Mean rank (95% CI) | Firsts | Mean AUROC |
|---|---|---|---|
| F1S | 3.15 (2.15 to 4.23) | 4 | 0.795 |
| lgbm_local | 4.00 (2.46 to 5.85) | 3 | 0.788 |
| F1 | 4.54 (3.31 to 5.85) | 0 | 0.777 |
| lgbm_ftlabel | 4.62 (2.62 to 7.00) | 2 | 0.764 |
| tabpfn_v3_5 | 6.00 (4.31 to 7.77) | 2 | 0.734 |
| Z0 | 6.31 (4.77 to 7.77) | 1 | 0.737 |
| Z1 | 6.85 (5.46 to 8.23) | 0 | 0.730 |
| lgbm_label | 7.31 (5.46 to 9.31) | 1 | 0.721 |
| lgbm_pooled | 7.85 (6.08 to 9.69) | 0 | 0.711 |
| sofa | 10.08 (7.77 to 12.23) | 0 | 0.652 |
| news | 10.54 (8.85 to 12.00) | 0 | 0.655 |
| mews | 10.85 (9.77 to 11.85) | 0 | 0.655 |
| qsofa | 12.08 (10.85 to 13.23) | 0 | 0.624 |
| logistic_pooled | 12.69 (11.92 to 13.46) | 0 | 0.605 |
| sirs | 13.15 (11.62 to 14.31) | 0 | 0.607 |

Friedman P 0.0000. F1 against each (Wilcoxon over the pairs, Holm):

| Comparator | F1 higher / lower / tied | Median AUROC difference (F1 minus) | P (Holm) |
|---|---|---|---|
| Z0 | 9 / 4 / 0 | 0.043 | 0.279 |
| Z1 | 9 / 4 / 0 | 0.052 | 0.279 |
| F1S | 2 / 11 / 0 | -0.007 | 0.020 |
| lgbm_pooled | 8 / 5 / 0 | 0.047 | 0.279 |
| logistic_pooled | 13 / 0 / 0 | 0.116 | 0.003 |
| tabpfn_v3_5 | 8 / 5 / 0 | 0.032 | 0.279 |
| sofa | 12 / 1 / 0 | 0.086 | 0.007 |
| news | 13 / 0 / 0 | 0.100 | 0.003 |
| mews | 13 / 0 / 0 | 0.111 | 0.003 |
| qsofa | 13 / 0 / 0 | 0.116 | 0.003 |
| sirs | 13 / 0 / 0 | 0.168 | 0.003 |
| lgbm_label | 10 / 3 / 0 | 0.030 | 0.279 |
| lgbm_ftlabel | 6 / 7 / 0 | -0.002 | 1.000 |
| lgbm_local | 7 / 6 / 0 | 0.001 | 1.000 |

Mean ranks by AUROC over the 13 TEST pairs with at least 10 positives (without the three local LightGBMs; F1+S keeps its LABEL-trained part):

| Predictor | Mean rank (95% CI) | Firsts | Mean AUROC |
|---|---|---|---|
| F1S | 2.15 (1.38 to 3.08) | 7 | 0.795 |
| F1 | 3.31 (2.31 to 4.38) | 1 | 0.777 |
| tabpfn_v3_5 | 4.15 (2.85 to 5.62) | 2 | 0.734 |
| Z0 | 4.54 (3.46 to 5.62) | 1 | 0.737 |
| Z1 | 4.85 (3.69 to 6.15) | 1 | 0.730 |
| lgbm_pooled | 5.77 (4.38 to 7.31) | 0 | 0.711 |
| sofa | 7.38 (5.38 to 9.31) | 1 | 0.652 |
| news | 8.00 (6.62 to 9.31) | 0 | 0.655 |
| mews | 8.23 (7.38 to 9.00) | 0 | 0.655 |
| qsofa | 9.31 (8.23 to 10.38) | 0 | 0.624 |
| logistic_pooled | 9.85 (9.00 to 10.69) | 0 | 0.605 |
| sirs | 10.46 (9.31 to 11.38) | 0 | 0.607 |

Friedman P 0.0000. F1 against each (Wilcoxon over the pairs, Holm):

| Comparator | F1 higher / lower / tied | Median AUROC difference (F1 minus) | P (Holm) |
|---|---|---|---|
| Z0 | 9 / 4 / 0 | 0.043 | 0.159 |
| Z1 | 9 / 4 / 0 | 0.052 | 0.159 |
| F1S | 2 / 11 / 0 | -0.007 | 0.012 |
| lgbm_pooled | 8 / 5 / 0 | 0.047 | 0.159 |
| logistic_pooled | 13 / 0 / 0 | 0.116 | 0.003 |
| tabpfn_v3_5 | 8 / 5 / 0 | 0.032 | 0.159 |
| sofa | 12 / 1 / 0 | 0.086 | 0.004 |
| news | 13 / 0 / 0 | 0.100 | 0.003 |
| mews | 13 / 0 / 0 | 0.111 | 0.003 |
| qsofa | 13 / 0 / 0 | 0.116 | 0.003 |
| sirs | 13 / 0 / 0 | 0.168 | 0.003 |

Labels under the improved encoding against 16a's (paper_a1's alignment control, lowest label agreement over programs): Z1:test_R 0.982; Z1:test_L24e 0.993; Z1:label_R 0.988; Z1:label_L24e 0.992; F1:test_R 0.982; F1:test_L24e 0.993; F1:label_R 0.988; F1:label_L24e 0.992.

Annotations: weighted metrics on the TEST anchors (R: 16a's inclusion weights; L24e: the case-cohort weights recomputed within TEST); intervals from 1,000 resamples of stays, paired differences from the same resamples. Each arm's estimate is the hazard estimate (E2) on 16a's hazard endpoints, the 128-future share otherwise. Z0 is 16a's prediction; Z1 the same model on the back-off encoding; F1 the adapted model. R: logistic recalibration on the LABEL anchors (the intercept alone below 10 LABEL positives), which keeps the ranks; S: stacking with the LABEL-only LightGBM. Endpoints with fewer than 10 TEST positives are descriptive. The statistics (families 21-24) are in `docs/paper/tables/manuscript/stats_family21-23.csv` and `stats_family24.csv`.
