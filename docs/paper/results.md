# Materials: results of the supplementary experiments

Every table is computed from the aggregate JSON files under `runs/paper/` on the host (protocol: `docs/p1_paper_protocol.md`, amendments 1-6), rendered by `tools/paper_results.py`. Held-out split unless stated; the tuning-split dry runs are not results.

## A1. Prediction by sampling (held-out split)

### 24-hour landmark (`runs/paper/a1/test_L24`)

3500 anchors x 32 rollouts; 21.6% of rollouts end at the token cap; alignment control (bin-edge programs) 1.000.

| Endpoint | Anchors | Positives | Twin AUROC | 95% CI | Twin (common anchors) | TabPFN-3.5 | TabPFN v2 | LightGBM pooled | LightGBM per dataset | Logistic | Twin probe | Twin - LightGBM | 95% CI (paired) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | 2904 | 104 | 0.858 | 0.814 to 0.902 | 0.858 | 0.932 | 0.914 | 0.915 | 0.895 | 0.864 | 0.906 | -0.057 | -0.096 to -0.019 |
| In-hospital death | 3500 | 346 | 0.858 | 0.837 to 0.879 | 0.858 | 0.881 | 0.869 | 0.867 | 0.853 | 0.830 | 0.870 | -0.008 | -0.028 to 0.010 |
| ICU stay > 7 days | 3465 | 485 | 0.808 | 0.786 to 0.828 | 0.808 | 0.856 | 0.839 | 0.855 | 0.844 | 0.787 | 0.837 | -0.047 | -0.066 to -0.028 |

All 57 binary catalog programs of this design: Twin AUROC median 0.888, 10th-90th percentile 0.838-0.941.

Twin AUROC by dataset:

| Endpoint | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|
| ICU death | 0.843 | 0.939 | 0.877 | 0.875 | 0.786 | 0.761 | 0.998 |
| In-hospital death | 0.857 | 0.913 | 0.847 | 0.850 | 0.869 | 0.896 | 0.731 |
| ICU stay > 7 days | 0.825 | 0.673 | 0.727 | 0.846 | 0.799 | 0.797 | 0.845 |

AUROC against the number of rollouts:

| Endpoint | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ICU death | 0.695 | 0.696 | 0.719 | 0.770 | 0.812 | 0.865 |
| In-hospital death | 0.620 | 0.672 | 0.729 | 0.789 | 0.833 | 0.858 |
| ICU stay > 7 days | 0.574 | 0.600 | 0.615 | 0.631 | 0.644 | 0.673 |

Regression and time-to-event programs (Twin ensembles):

| Program | Anchors | MAE (median) | MAE (predict the mean) | CRPS | 80% coverage | C-index (48 h share) |
|---|---|---|---|---|---|---|
| remaining_los@0 | 2904 | 41.746 | 58.560 | 34.477 | 0.697 | - |
| creatinine_24h@0 | 320 | 0.431 | 0.915 | 0.335 | 0.794 | - |
| creatinine_change_48h@0 | 492 | 0.690 | 0.777 | 0.564 | 0.728 | - |
| lactate_max_12h@0 | 941 | 1.001 | 1.846 | 0.762 | 0.744 | - |
| map_min_6h@0 | 2242 | 8.537 | 13.033 | 6.394 | 0.739 | - |
| heart_rate_next_1h@0 | 1965 | 6.462 | 15.367 | 5.147 | 0.768 | - |
| imv_time@0 | 2758 | - | - | - | - | 0.795 |
| vasopressor_time@0 | 2948 | - | - | - | - | 0.827 |
| death_time@0 | 3500 | - | - | - | - | 0.854 |

Label efficiency: LightGBM per dataset trained on n train stays, against the Twin (zero-shot):

| Endpoint | Dataset | Twin | n=100 | 300 | 1,000 | 3,000 | all (up to 5,000) |
|---|---|---|---|---|---|---|---|
| ICU death | MIMIC-IV | 0.843 | 0.749 | 0.850 | 0.868 | 0.850 | 0.852 |
| ICU death | eICU | 0.939 | 0.697 | 0.784 | 0.795 | 0.828 | 0.828 |
| ICU death | HiRID | 0.877 | 0.868 | 0.872 | 0.896 | - | 0.872 |
| ICU death | AmsterdamUMCdb | 0.875 | 0.818 | 0.816 | 0.855 | 0.865 | 0.867 |
| ICU death | SICdb | 0.786 | 0.879 | 0.888 | 0.880 | - | 0.898 |
| ICU death | NWICU | 0.761 | 0.755 | 0.783 | 0.838 | 0.849 | 0.881 |
| ICU death | INSPIRE | 0.998 | 0.720 | 0.829 | 0.875 | - | 0.913 |
| In-hospital death | MIMIC-IV | 0.857 | 0.670 | 0.802 | 0.823 | 0.864 | 0.850 |
| In-hospital death | eICU | 0.913 | 0.680 | 0.724 | 0.781 | 0.837 | 0.818 |
| In-hospital death | HiRID | 0.847 | 0.820 | 0.879 | 0.878 | - | 0.870 |
| In-hospital death | AmsterdamUMCdb | 0.850 | 0.818 | 0.816 | 0.855 | 0.865 | 0.867 |
| In-hospital death | SICdb | 0.869 | 0.737 | 0.804 | 0.792 | - | 0.832 |
| In-hospital death | NWICU | 0.896 | 0.731 | 0.840 | 0.852 | 0.847 | 0.852 |
| In-hospital death | INSPIRE | 0.731 | 0.757 | 0.734 | 0.756 | - | 0.755 |
| ICU stay > 7 days | MIMIC-IV | 0.825 | 0.705 | 0.749 | 0.789 | 0.822 | 0.826 |
| ICU stay > 7 days | eICU | 0.673 | 0.627 | 0.721 | 0.744 | 0.803 | 0.787 |
| ICU stay > 7 days | HiRID | 0.727 | 0.640 | 0.688 | 0.709 | - | 0.737 |
| ICU stay > 7 days | AmsterdamUMCdb | 0.846 | 0.763 | 0.846 | 0.844 | 0.851 | 0.850 |
| ICU stay > 7 days | SICdb | 0.799 | 0.770 | 0.814 | 0.819 | - | 0.842 |
| ICU stay > 7 days | NWICU | 0.797 | 0.684 | 0.732 | 0.770 | 0.766 | 0.796 |
| ICU stay > 7 days | INSPIRE | 0.845 | 0.736 | 0.881 | 0.915 | - | 0.890 |

### 48-hour landmark (`runs/paper/a1/test_L48`)

3500 anchors x 32 rollouts; 29.1% of rollouts end at the token cap; alignment control (bin-edge programs) 1.000.

| Endpoint | Anchors | Positives | Twin AUROC | 95% CI | Twin (common anchors) | TabPFN-3.5 | TabPFN v2 | LightGBM pooled | LightGBM per dataset | Logistic | Twin probe | Twin - LightGBM | 95% CI (paired) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | 2643 | 127 | 0.882 | 0.845 to 0.914 | 0.882 | 0.913 | 0.907 | 0.903 | 0.873 | 0.864 | 0.919 | -0.021 | -0.053 to 0.010 |
| In-hospital death | 3500 | 455 | 0.830 | 0.808 to 0.848 | 0.830 | 0.858 | 0.847 | 0.859 | 0.831 | 0.798 | 0.848 | -0.029 | -0.049 to -0.010 |
| ICU stay > 7 days | 3449 | 686 | 0.797 | 0.778 to 0.816 | 0.797 | 0.835 | 0.829 | 0.834 | 0.810 | 0.791 | 0.825 | -0.036 | -0.055 to -0.019 |

All 56 binary catalog programs of this design: Twin AUROC median 0.885, 10th-90th percentile 0.841-0.923.

Twin AUROC by dataset:

| Endpoint | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|
| ICU death | 0.830 | 0.978 | 0.908 | 0.836 | 0.912 | 0.838 | - |
| In-hospital death | 0.849 | 0.890 | 0.790 | 0.816 | 0.882 | 0.835 | 0.785 |
| ICU stay > 7 days | 0.834 | 0.785 | 0.696 | 0.810 | 0.784 | 0.771 | 0.797 |

AUROC against the number of rollouts:

| Endpoint | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| ICU death | 0.694 | 0.704 | 0.730 | 0.781 | 0.818 | 0.850 |
| In-hospital death | 0.621 | 0.668 | 0.716 | 0.776 | 0.804 | 0.830 |
| ICU stay > 7 days | 0.591 | 0.608 | 0.625 | 0.629 | 0.640 | 0.642 |

Regression and time-to-event programs (Twin ensembles):

| Program | Anchors | MAE (median) | MAE (predict the mean) | CRPS | 80% coverage | C-index (48 h share) |
|---|---|---|---|---|---|---|
| remaining_los@0 | 2643 | 54.826 | 73.383 | 45.400 | 0.661 | - |
| creatinine_24h@0 | 447 | 0.513 | 1.040 | 0.401 | 0.734 | - |
| creatinine_change_48h@0 | 660 | 0.660 | 0.692 | 0.532 | 0.758 | - |
| lactate_max_12h@0 | 932 | 0.737 | 1.240 | 0.546 | 0.736 | - |
| map_min_6h@0 | 2451 | 8.218 | 12.823 | 6.076 | 0.755 | - |
| heart_rate_next_1h@0 | 2034 | 6.803 | 15.467 | 5.328 | 0.749 | - |
| imv_time@0 | 2598 | - | - | - | - | 0.782 |
| vasopressor_time@0 | 2946 | - | - | - | - | 0.809 |
| death_time@0 | 3500 | - | - | - | - | 0.790 |

Label efficiency: LightGBM per dataset trained on n train stays, against the Twin (zero-shot):

| Endpoint | Dataset | Twin | n=100 | 300 | 1,000 | 3,000 | all (up to 5,000) |
|---|---|---|---|---|---|---|---|
| ICU death | MIMIC-IV | 0.830 | 0.775 | 0.760 | 0.803 | - | 0.834 |
| ICU death | eICU | 0.978 | 0.759 | 0.836 | 0.901 | - | 0.897 |
| ICU death | HiRID | 0.908 | 0.806 | 0.862 | 0.864 | - | 0.866 |
| ICU death | AmsterdamUMCdb | 0.836 | 0.754 | 0.788 | 0.788 | - | 0.823 |
| ICU death | SICdb | 0.912 | 0.807 | 0.872 | 0.886 | - | 0.908 |
| ICU death | NWICU | 0.838 | 0.688 | 0.767 | 0.792 | - | 0.796 |
| ICU death | INSPIRE | - | 0.857 | 0.897 | 0.925 | - | 0.925 |
| In-hospital death | MIMIC-IV | 0.849 | 0.734 | 0.796 | 0.817 | - | 0.844 |
| In-hospital death | eICU | 0.890 | 0.663 | 0.808 | 0.804 | - | 0.875 |
| In-hospital death | HiRID | 0.790 | 0.649 | 0.865 | 0.849 | - | 0.855 |
| In-hospital death | AmsterdamUMCdb | 0.816 | 0.754 | 0.788 | 0.788 | - | 0.823 |
| In-hospital death | SICdb | 0.882 | 0.814 | 0.783 | 0.858 | - | 0.854 |
| In-hospital death | NWICU | 0.835 | 0.662 | 0.702 | 0.768 | - | 0.794 |
| In-hospital death | INSPIRE | 0.785 | 0.680 | 0.775 | 0.780 | - | 0.789 |
| ICU stay > 7 days | MIMIC-IV | 0.834 | 0.754 | 0.763 | 0.789 | - | 0.819 |
| ICU stay > 7 days | eICU | 0.785 | 0.567 | 0.680 | 0.750 | - | 0.759 |
| ICU stay > 7 days | HiRID | 0.696 | 0.590 | 0.651 | 0.715 | - | 0.713 |
| ICU stay > 7 days | AmsterdamUMCdb | 0.810 | 0.721 | 0.763 | 0.768 | - | 0.799 |
| ICU stay > 7 days | SICdb | 0.784 | 0.702 | 0.755 | 0.777 | - | 0.784 |
| ICU stay > 7 days | NWICU | 0.771 | 0.708 | 0.699 | 0.731 | - | 0.740 |
| ICU stay > 7 days | INSPIRE | 0.797 | 0.814 | 0.882 | 0.884 | - | 0.900 |

### rolling times (enriched, weighted) (`runs/paper/a1/test_R`)

3500 anchors x 32 rollouts; 36.1% of rollouts end at the token cap; alignment control (bin-edge programs) 1.000.

| Endpoint | Anchors | Positives | Twin AUROC | 95% CI | Twin (common anchors) | TabPFN-3.5 | TabPFN v2 | LightGBM pooled | LightGBM per dataset | Logistic | Twin probe | Twin - LightGBM | 95% CI (paired) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 3486 | 742 | 0.893 | 0.873 to 0.911 | 0.893 | 0.956 | 0.952 | 0.955 | 0.938 | 0.936 | 0.944 | -0.062 | -0.079 to -0.048 |
| New IMV, 24 h | 2441 | 236 | 0.826 | 0.766 to 0.884 | 0.826 | 0.814 | 0.782 | 0.817 | 0.822 | 0.750 | 0.769 | 0.009 | -0.043 to 0.065 |
| New vasopressor, 24 h | 2730 | 357 | 0.859 | 0.827 to 0.890 | 0.859 | 0.813 | 0.808 | 0.803 | 0.802 | 0.812 | 0.738 | 0.056 | 0.028 to 0.086 |
| New RRT, 48 h | 3285 | 296 | 0.939 | 0.894 to 0.968 | 0.939 | 0.944 | 0.922 | 0.932 | 0.919 | 0.880 | 0.933 | 0.007 | -0.010 to 0.025 |
| AKI stage >= 2, 48 h | 2922 | 648 | 0.895 | 0.871 to 0.915 | 0.895 | 0.853 | 0.836 | 0.858 | 0.857 | 0.824 | 0.813 | 0.037 | 0.020 to 0.055 |
| Sepsis (ASE), 24 h | 3486 | 28 | 0.941 | 0.873 to 0.983 | 0.883 | 0.811 | 0.811 | 0.766 | 0.774 | 0.749 | 0.760 | 0.117 | 0.008 to 0.219 |
| Hypotension, 4 h | 3011 | 274 | 0.906 | 0.877 to 0.935 | 0.894 | 0.907 | 0.895 | 0.910 | 0.895 | 0.873 | 0.822 | -0.016 | -0.039 to 0.003 |
| Hyperkalaemia (top bin), 24 h | 3065 | 283 | 0.804 | 0.748 to 0.859 | 0.804 | 0.822 | 0.810 | 0.826 | 0.808 | 0.783 | 0.716 | -0.021 | -0.061 to 0.017 |
| Thrombocytopenia, 48 h | 3061 | 150 | 0.939 | 0.894 to 0.976 | 0.939 | 0.954 | 0.932 | 0.953 | 0.947 | 0.947 | 0.884 | -0.014 | -0.047 to 0.010 |

All 53 binary catalog programs of this design: Twin AUROC median 0.882, 10th-90th percentile 0.830-0.946.

Twin AUROC by dataset:

| Endpoint | MIMIC-IV | eICU | HiRID | AmsterdamUMCdb | SICdb | NWICU | INSPIRE |
|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 0.927 | 0.828 | 0.885 | 0.892 | 0.873 | 0.896 | 0.912 |
| New IMV, 24 h | 0.935 | 0.795 | 0.698 | 0.894 | 0.945 | 0.860 | 0.630 |
| New vasopressor, 24 h | 0.942 | 0.910 | 0.693 | 0.884 | 0.769 | 0.874 | 0.701 |
| New RRT, 48 h | 0.971 | 0.975 | 0.907 | 0.912 | 0.944 | 0.986 | 0.721 |
| AKI stage >= 2, 48 h | 0.842 | 0.863 | 0.866 | 0.875 | 0.854 | 0.911 | 0.864 |
| Sepsis (ASE), 24 h | 0.842 | - | - | 0.945 | - | - | - |
| Hypotension, 4 h | 0.841 | 0.886 | 0.884 | 0.904 | 0.893 | - | 0.915 |
| Hyperkalaemia (top bin), 24 h | 0.799 | 0.814 | 0.853 | 0.901 | 0.650 | 0.746 | 0.729 |
| Thrombocytopenia, 48 h | 0.871 | 0.730 | 0.990 | 0.995 | 0.981 | 0.917 | 0.966 |

AUROC against the number of rollouts:

| Endpoint | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| Decompensation, 24 h | 0.652 | 0.707 | 0.762 | 0.812 | 0.852 | 0.893 |
| New IMV, 24 h | 0.601 | 0.625 | 0.675 | 0.697 | 0.765 | 0.826 |
| New vasopressor, 24 h | 0.594 | 0.635 | 0.719 | 0.780 | 0.829 | 0.860 |
| New RRT, 48 h | 0.682 | 0.743 | 0.786 | 0.883 | 0.910 | 0.939 |
| AKI stage >= 2, 48 h | 0.710 | 0.768 | 0.823 | 0.868 | 0.880 | 0.894 |
| Sepsis (ASE), 24 h | 0.586 | 0.619 | 0.711 | 0.853 | 0.885 | 0.941 |
| Hypotension, 4 h | 0.637 | 0.727 | 0.806 | 0.851 | 0.888 | 0.906 |
| Hyperkalaemia (top bin), 24 h | 0.585 | 0.611 | 0.645 | 0.710 | 0.771 | 0.804 |
| Thrombocytopenia, 48 h | 0.672 | 0.730 | 0.826 | 0.870 | 0.915 | 0.940 |

Regression and time-to-event programs (Twin ensembles):

| Program | Anchors | MAE (median) | MAE (predict the mean) | CRPS | 80% coverage | C-index (48 h share) |
|---|---|---|---|---|---|---|
| creatinine_24h@0 | 268 | 0.784 | 1.586 | 0.604 | 0.720 | - |
| creatinine_change_48h@0 | 251 | 1.029 | 1.054 | 0.825 | 0.745 | - |
| lactate_max_12h@0 | 876 | 1.693 | 3.365 | 1.243 | 0.743 | - |
| map_min_6h@0 | 2210 | 10.164 | 14.404 | 7.397 | 0.724 | - |
| heart_rate_next_1h@0 | 2018 | 7.341 | 17.182 | 5.934 | 0.764 | - |
| imv_time@0 | 2454 | - | - | - | - | 0.758 |
| vasopressor_time@0 | 2743 | - | - | - | - | 0.746 |
| death_time@0 | 3499 | - | - | - | - | 0.834 |

Label efficiency: LightGBM per dataset trained on n train stays, against the Twin (zero-shot):

| Endpoint | Dataset | Twin | n=100 | 300 | 1,000 | 3,000 | all (up to 5,000) |
|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | MIMIC-IV | 0.927 | 0.721 | 0.703 | 0.918 | 0.942 | 0.954 |
| Decompensation, 24 h | eICU | 0.828 | 0.745 | 0.784 | 0.845 | 0.861 | 0.887 |
| Decompensation, 24 h | HiRID | 0.885 | 0.857 | 0.770 | 0.912 | 0.914 | 0.941 |
| Decompensation, 24 h | AmsterdamUMCdb | 0.892 | 0.820 | 0.830 | 0.926 | 0.948 | 0.951 |
| Decompensation, 24 h | SICdb | 0.873 | 0.886 | 0.878 | 0.933 | 0.941 | 0.940 |
| Decompensation, 24 h | NWICU | 0.896 | 0.805 | 0.832 | 0.913 | 0.923 | 0.935 |
| Decompensation, 24 h | INSPIRE | 0.912 | 0.684 | 0.786 | 0.872 | 0.926 | 0.881 |
| New IMV, 24 h | MIMIC-IV | 0.935 | 0.576 | 0.763 | 0.740 | 0.805 | 0.869 |
| New IMV, 24 h | eICU | 0.795 | 0.500 | 0.650 | 0.826 | 0.845 | 0.873 |
| New IMV, 24 h | HiRID | 0.698 | 0.649 | 0.678 | 0.741 | 0.719 | 0.805 |
| New IMV, 24 h | AmsterdamUMCdb | 0.894 | 0.550 | 0.788 | 0.764 | 0.860 | 0.854 |
| New IMV, 24 h | SICdb | 0.945 | 0.672 | 0.866 | 0.748 | 0.842 | 0.872 |
| New IMV, 24 h | NWICU | 0.860 | 0.737 | 0.726 | 0.688 | 0.791 | 0.811 |
| New IMV, 24 h | INSPIRE | 0.630 | 0.450 | 0.655 | 0.446 | 0.619 | 0.817 |
| New vasopressor, 24 h | MIMIC-IV | 0.942 | 0.804 | 0.827 | 0.753 | 0.857 | 0.869 |
| New vasopressor, 24 h | eICU | 0.910 | 0.694 | 0.709 | 0.719 | 0.807 | 0.805 |
| New vasopressor, 24 h | HiRID | 0.693 | 0.535 | 0.574 | 0.611 | 0.695 | 0.655 |
| New vasopressor, 24 h | AmsterdamUMCdb | 0.884 | 0.594 | 0.870 | 0.887 | 0.861 | 0.888 |
| New vasopressor, 24 h | SICdb | 0.769 | 0.645 | 0.694 | 0.712 | 0.693 | 0.719 |
| New vasopressor, 24 h | NWICU | 0.874 | 0.662 | 0.704 | 0.810 | 0.807 | 0.824 |
| New vasopressor, 24 h | INSPIRE | 0.701 | 0.417 | 0.525 | 0.508 | 0.472 | 0.563 |
| New RRT, 48 h | MIMIC-IV | 0.971 | 0.590 | 0.870 | 0.900 | 0.948 | 0.944 |
| New RRT, 48 h | eICU | 0.975 | 0.466 | 0.545 | 0.836 | 0.848 | 0.888 |
| New RRT, 48 h | HiRID | 0.907 | 0.686 | 0.742 | 0.835 | 0.845 | 0.879 |
| New RRT, 48 h | AmsterdamUMCdb | 0.912 | 0.918 | 0.829 | 0.859 | 0.915 | 0.939 |
| New RRT, 48 h | SICdb | 0.944 | - | 0.736 | 0.781 | 0.925 | 0.928 |
| New RRT, 48 h | NWICU | 0.986 | 0.927 | 0.935 | 0.966 | 0.970 | 0.977 |
| New RRT, 48 h | INSPIRE | 0.721 | - | 0.639 | 0.652 | 0.733 | 0.639 |
| AKI stage >= 2, 48 h | MIMIC-IV | 0.842 | 0.707 | 0.726 | 0.721 | 0.752 | 0.765 |
| AKI stage >= 2, 48 h | eICU | 0.863 | 0.656 | 0.706 | 0.718 | 0.756 | 0.777 |
| AKI stage >= 2, 48 h | HiRID | 0.866 | 0.706 | 0.743 | 0.844 | 0.852 | 0.869 |
| AKI stage >= 2, 48 h | AmsterdamUMCdb | 0.875 | 0.547 | 0.629 | 0.767 | 0.787 | 0.785 |
| AKI stage >= 2, 48 h | SICdb | 0.854 | 0.693 | 0.741 | 0.782 | 0.766 | 0.801 |
| AKI stage >= 2, 48 h | NWICU | 0.911 | 0.353 | 0.714 | 0.836 | 0.828 | 0.856 |
| AKI stage >= 2, 48 h | INSPIRE | 0.864 | 0.665 | 0.649 | 0.796 | 0.780 | 0.844 |
| Sepsis (ASE), 24 h | MIMIC-IV | 0.842 | 0.618 | 0.672 | 0.662 | 0.688 | 0.787 |
| Sepsis (ASE), 24 h | AmsterdamUMCdb | 0.945 | 0.636 | 0.673 | 0.820 | 0.890 | 0.801 |
| Hypotension, 4 h | MIMIC-IV | 0.841 | 0.889 | 0.848 | 0.830 | 0.880 | 0.890 |
| Hypotension, 4 h | eICU | 0.886 | 0.873 | 0.852 | 0.855 | 0.924 | 0.891 |
| Hypotension, 4 h | HiRID | 0.884 | 0.853 | 0.859 | 0.857 | 0.864 | 0.845 |
| Hypotension, 4 h | AmsterdamUMCdb | 0.904 | 0.910 | 0.917 | 0.885 | 0.895 | 0.886 |
| Hypotension, 4 h | SICdb | 0.893 | 0.750 | 0.808 | 0.843 | 0.870 | 0.883 |
| Hypotension, 4 h | INSPIRE | 0.915 | 0.765 | 0.538 | 0.863 | 0.933 | 0.973 |
| Hyperkalaemia (top bin), 24 h | MIMIC-IV | 0.799 | 0.651 | 0.722 | 0.744 | 0.743 | 0.752 |
| Hyperkalaemia (top bin), 24 h | eICU | 0.814 | 0.583 | 0.511 | 0.612 | 0.820 | 0.814 |
| Hyperkalaemia (top bin), 24 h | HiRID | 0.853 | 0.753 | 0.801 | 0.859 | 0.882 | 0.873 |
| Hyperkalaemia (top bin), 24 h | AmsterdamUMCdb | 0.901 | 0.628 | 0.833 | 0.799 | 0.850 | 0.874 |
| Hyperkalaemia (top bin), 24 h | SICdb | 0.650 | 0.643 | 0.674 | 0.699 | 0.751 | 0.691 |
| Hyperkalaemia (top bin), 24 h | NWICU | 0.746 | 0.391 | 0.660 | 0.638 | 0.593 | 0.614 |
| Hyperkalaemia (top bin), 24 h | INSPIRE | 0.729 | 0.584 | 0.799 | 0.578 | 0.735 | 0.829 |
| Thrombocytopenia, 48 h | MIMIC-IV | 0.871 | 0.742 | 0.489 | 0.881 | 0.917 | 0.907 |
| Thrombocytopenia, 48 h | eICU | 0.730 | 0.826 | 0.784 | 0.647 | 0.772 | 0.869 |
| Thrombocytopenia, 48 h | HiRID | 0.990 | 0.866 | 0.883 | 0.880 | 0.970 | 0.983 |
| Thrombocytopenia, 48 h | AmsterdamUMCdb | 0.995 | 0.933 | 0.948 | 0.937 | 0.984 | 0.985 |
| Thrombocytopenia, 48 h | SICdb | 0.981 | 0.932 | 0.840 | 0.939 | 0.914 | 0.953 |
| Thrombocytopenia, 48 h | NWICU | 0.917 | 0.757 | 0.822 | 0.901 | 0.924 | 0.927 |
| Thrombocytopenia, 48 h | INSPIRE | 0.966 | 0.678 | 0.960 | 0.927 | 0.940 | 0.964 |

Annotations: the Twin is zero-shot (MUC32_main's 32 rollouts per anchor, no real labels); every other model is trained on real train stays (TabPFN on a stratified sample, the probe on the model's frozen states). The rolling design is enriched (half its anchors precede an event) and its metrics are inclusion-weighted. 'Common anchors' are those every model scored; the paired difference resamples stays within datasets. SOFA (first day) is in `baselines.json` for the death endpoints.

## A2. Calibration and uncertainty

### 24-hour landmark

| Endpoint | O/E | Slope | ECE | Brier | O/E recalibrated | ECE recalibrated | O/E by ICU day of the anchor |
|---|---|---|---|---|---|---|---|
| ICU death | 1.14 | 0.45 | 0.010 | 0.024 | 0.97 | 0.013 | d1: 1.14 |
| In-hospital death | 1.40 | 0.55 | 0.028 | 0.068 | 0.93 | 0.020 | d1: 1.40 |
| ICU stay > 7 days | 0.81 | 0.38 | 0.055 | 0.104 | 0.83 | 0.032 | d1: 0.81 |

### 48-hour landmark

| Endpoint | O/E | Slope | ECE | Brier | O/E recalibrated | ECE recalibrated | O/E by ICU day of the anchor |
|---|---|---|---|---|---|---|---|
| ICU death | 1.21 | 0.50 | 0.011 | 0.031 | 1.11 | 0.012 | d2: 1.21 |
| In-hospital death | 1.49 | 0.49 | 0.043 | 0.089 | 0.95 | 0.019 | d2: 1.49 |
| ICU stay > 7 days | 0.83 | 0.38 | 0.069 | 0.132 | 0.88 | 0.037 | d2: 0.83 |

### rolling times (enriched, weighted)

| Endpoint | O/E | Slope | ECE | Brier | O/E recalibrated | ECE recalibrated | O/E by ICU day of the anchor |
|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 1.83 | 0.62 | 0.012 | 0.018 | 0.86 | 0.006 | d1: 1.84, d2: 1.78, d3: 2.03, d4+: 1.76 |
| New IMV, 24 h | 0.69 | 0.41 | 0.018 | 0.023 | 1.13 | 0.008 | d1: 0.66, d2: 0.73, d3: 0.77, d4+: 0.70 |
| New vasopressor, 24 h | 0.49 | 0.71 | 0.026 | 0.022 | 0.96 | 0.005 | d1: 0.59, d2: 0.43, d3: 0.72, d4+: 0.32 |
| New RRT, 48 h | 0.66 | 1.03 | 0.015 | 0.017 | 1.06 | 0.004 | d1: 0.47, d2: 0.75, d3: 0.54, d4+: 0.85 |
| AKI stage >= 2, 48 h | 0.73 | 0.86 | 0.059 | 0.088 | 0.92 | 0.031 | d1: 0.76, d2: 0.80, d3: 0.57, d4+: 0.69 |
| Sepsis (ASE), 24 h | 1.01 | 0.88 | 0.001 | 0.007 | 1.23 | 0.006 | d4+: 0.96 |
| Hypotension, 4 h | 1.04 | 0.63 | 0.008 | 0.038 | 0.82 | 0.027 | d1: 1.06, d2: 0.85, d3: 0.98, d4+: 1.17 |
| Hyperkalaemia (top bin), 24 h | 0.78 | 0.44 | 0.025 | 0.040 | 0.60 | 0.040 | d1: 0.77, d2: 1.00, d3: 0.73, d4+: 0.69 |
| Thrombocytopenia, 48 h | 0.81 | 0.86 | 0.008 | 0.015 | 0.91 | 0.004 | d1: 0.91, d2: 0.98, d4+: 0.52 |

### Teacher-forced against rollout calibration (exposure bias)

| Event | Teacher-forced O/E after 24 h (pooled, masked) | Rollout endpoint | Rollout O/E (R) |
|---|---|---|---|
| imv | 0.76 | New IMV, 24 h | 0.69 |
| vasopressor | 0.59 | New vasopressor, 24 h | 0.49 |
| rrt | 0.44 | New RRT, 48 h | 0.66 |
| icu_death | 0.97 | Decompensation, 24 h | 1.83 |

The teacher-forced O/E sums the model's next-event probability of each event over every eligible position after the 24-h landmark (whole remaining stay); the rollout O/E is over the endpoint's horizon from rolling anchors, so the death row compares related, not identical, quantities.

Remaining ICU stay (L24) dispersion: stays 2942, pit_ks 0.087, pit_deciles [228, 274, 268, 296, 265, 301, 252, 282, 238, 538], crps 34.747, spread_skill 0.272

Remaining ICU stay (L48) dispersion: stays 2688, pit_ks 0.112, pit_deciles [216, 246, 241, 228, 263, 244, 228, 230, 235, 557], crps 45.362, spread_skill 0.256

Annotations: O/E above 1 means the Twin under-predicts. Recalibration is isotonic per endpoint, fitted on the tuning-split dry run of the same design (A2 of the protocol) and applied to the held-out anchors; it is reported beside the raw Twin, never instead of it.

## A3. Privacy: the gate and the fit-privacy frontier (`runs/paper/a3/a3.json`)

P1 membership inference of MUC32_main (gate PASS):

| Group | AUC | 95% CI | TPR at 1% FPR | Status |
|---|---|---|---|---|
| mimic | 0.551 | 0.536 to 0.565 | 0.011 | PASS |
| aumc | 0.552 | 0.537 to 0.568 | 0.019 | PASS |
| eicu | 0.552 | 0.538 to 0.566 | 0.024 | PASS |
| hirid | 0.552 | 0.537 to 0.566 | 0.020 | PASS |
| sicdb | 0.555 | 0.541 to 0.570 | 0.015 | PASS |
| nwicu | 0.549 | 0.535 to 0.563 | 0.010 | PASS |
| inspire | 0.551 | 0.533 to 0.568 | 0.011 | PASS |
| pooled | 0.538 | 0.532 to 0.544 | 0.015 | PASS |

Checkpoints (the frontier):

| Checkpoint | Optimizer | Tuning NLL | P1 AUC pooled | P1 AUC, worst dataset | P1 status |
|---|---|---|---|---|---|
| MUH32_main | muon | 1.5092 | 0.564 | 0.623 | FAIL |
| MU32_main | muon | 1.5274 | 0.554 | 0.604 | INCONCLUSIVE |
| MUF32_main | muon | 1.5388 | 0.549 | 0.573 | INCONCLUSIVE |
| MUC32_main | muon | 1.5525 | 0.538 | 0.555 | PASS |
| M32_main | adamw | 1.5611 | 0.530 | 0.561 | PASS |
| ML32_main | adamw | 1.5631 | 0.539 | 0.577 | PASS |
| MC32_main | adamw | 1.5636 | 0.530 | 0.560 | PASS |
| M_main | adamw | 1.5820 | 0.542 | 0.650 | FAIL |
| MCN32_main | adamw | 1.6067 | 0.521 | 0.547 | PASS |

Rank correlation across checkpoints, per dataset:

| Dataset | AUC vs member/non-member NLL gap | AUC vs held-out NLL |
|---|---|---|
| aumc | 1.00 | -0.88 |
| eicu | 1.00 | -0.97 |
| hirid | 1.00 | -0.97 |
| inspire | 1.00 | -0.95 |
| mimic | 0.98 | -0.93 |
| nwicu | 1.00 | -0.85 |
| sicdb | 1.00 | -0.88 |

Annotations: AdamW and Muon checkpoints lie on one curve: the membership signal follows the fit reached, not the optimizer; MUC32_main's per-dataset passes were set on this frontier (AUC 0.549-0.555 in every dataset). The audit's samples are unconstrained by protocol (worst case).

## A4. Utility of synthetic data (`baselines.json` per design)

### 24-hour landmark

| Endpoint | Rows | TSTR | TRTR | Ratio of gains | MIMIC-IV TSTR/TRTR | eICU TSTR/TRTR | HiRID TSTR/TRTR | AmsterdamUMCdb TSTR/TRTR | SICdb TSTR/TRTR | NWICU TSTR/TRTR | INSPIRE TSTR/TRTR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | 7963 | 0.870 | 0.900 | 0.93 | 0.798 / 0.830 | 0.781 / 0.800 | 0.887 / 0.858 | 0.780 / 0.840 | 0.868 / 0.899 | 0.828 / 0.827 | 0.599 / 0.847 |
| In-hospital death | 8047 | 0.841 | 0.853 | 0.96 | 0.799 / 0.863 | 0.782 / 0.803 | 0.893 / 0.863 | 0.788 / 0.846 | 0.794 / 0.794 | 0.781 / 0.834 | 0.747 / 0.796 |
| ICU stay > 7 days | 7979 | 0.819 | 0.821 | 1.00 | 0.777 / 0.808 | 0.795 / 0.707 | 0.680 / 0.725 | 0.801 / 0.832 | 0.825 / 0.786 | 0.765 / 0.793 | 0.857 / 0.920 |

Small-site augmentation (AUROC on that dataset's held-out anchors):

| Endpoint | Dataset | 300 real | 300 real + 300 twins | 300 real + 900 twins | all real |
|---|---|---|---|---|---|
| ICU death | INSPIRE | 0.841 | 0.652 | - | 0.894 |
| ICU death | NWICU | 0.804 | 0.797 | 0.851 | 0.885 |
| In-hospital death | INSPIRE | 0.759 | 0.765 | - | 0.794 |
| In-hospital death | NWICU | 0.822 | 0.820 | 0.809 | 0.854 |
| ICU stay > 7 days | INSPIRE | 0.818 | 0.887 | - | 0.902 |
| ICU stay > 7 days | NWICU | 0.728 | 0.762 | 0.757 | 0.794 |

### 48-hour landmark

| Endpoint | Rows | TSTR | TRTR | Ratio of gains | MIMIC-IV TSTR/TRTR | eICU TSTR/TRTR | HiRID TSTR/TRTR | AmsterdamUMCdb TSTR/TRTR | SICdb TSTR/TRTR | NWICU TSTR/TRTR | INSPIRE TSTR/TRTR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | 4665 | 0.856 | 0.873 | 0.95 | 0.792 / 0.743 | 0.820 / 0.870 | 0.839 / 0.869 | 0.709 / 0.783 | 0.845 / 0.864 | 0.784 / 0.779 | 0.677 / 0.874 |
| In-hospital death | 4717 | 0.815 | 0.840 | 0.93 | 0.810 / 0.802 | 0.807 / 0.832 | 0.697 / 0.832 | 0.703 / 0.771 | 0.837 / 0.799 | 0.732 / 0.723 | 0.714 / 0.784 |
| ICU stay > 7 days | 4680 | 0.796 | 0.812 | 0.95 | 0.782 / 0.789 | 0.818 / 0.805 | 0.650 / 0.737 | 0.730 / 0.791 | 0.782 / 0.759 | 0.708 / 0.695 | 0.781 / 0.822 |

Small-site augmentation (AUROC on that dataset's held-out anchors):

| Endpoint | Dataset | 300 real | 300 real + 300 twins | 300 real + 900 twins | all real |
|---|---|---|---|---|---|
| ICU death | NWICU | 0.773 | 0.789 | - | 0.806 |
| In-hospital death | NWICU | 0.739 | 0.726 | - | 0.777 |
| ICU stay > 7 days | NWICU | 0.677 | 0.680 | - | 0.730 |

### rolling times (enriched, weighted)

| Endpoint | Rows | TSTR | TRTR | Ratio of gains | MIMIC-IV TSTR/TRTR | eICU TSTR/TRTR | HiRID TSTR/TRTR | AmsterdamUMCdb TSTR/TRTR | SICdb TSTR/TRTR | NWICU TSTR/TRTR | INSPIRE TSTR/TRTR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | 47485 | 0.929 | 0.950 | 0.95 | 0.937 / 0.953 | 0.867 / 0.873 | 0.883 / 0.935 | 0.860 / 0.934 | 0.807 / 0.936 | 0.905 / 0.926 | 0.593 / 0.661 |
| New IMV, 24 h | 40890 | 0.778 | 0.776 | 1.00 | 0.677 / 0.799 | 0.758 / 0.822 | 0.743 / 0.521 | 0.667 / 0.759 | 0.749 / 0.504 | 0.807 / 0.763 | 0.380 / 0.442 |
| New vasopressor, 24 h | 43743 | 0.792 | 0.805 | 0.96 | 0.834 / 0.879 | 0.752 / 0.777 | 0.716 / 0.635 | 0.803 / 0.907 | 0.714 / 0.707 | 0.761 / 0.834 | 0.579 / 0.518 |
| New RRT, 48 h | 47188 | 0.898 | 0.925 | 0.93 | 0.854 / 0.951 | 0.804 / 0.838 | 0.793 / 0.854 | 0.819 / 0.820 | 0.470 / 0.840 | 0.942 / 0.953 | 0.838 / 0.777 |
| AKI stage >= 2, 48 h | 43910 | 0.830 | 0.852 | 0.94 | 0.689 / 0.746 | 0.725 / 0.802 | 0.755 / 0.844 | 0.721 / 0.768 | 0.732 / 0.781 | 0.689 / 0.584 | 0.826 / 0.616 |
| Sepsis (ASE), 24 h | 47484 | 0.927 | 0.914 | 1.03 | 0.661 / 0.709 | - / - | - / - | 0.575 / 0.567 | - / - | - / - | - / - |
| Hypotension, 4 h | 44637 | 0.919 | 0.919 | 1.00 | 0.906 / 0.854 | 0.906 / 0.902 | 0.856 / 0.870 | 0.868 / 0.904 | 0.837 / 0.851 | - / - | 0.847 / 0.791 |
| Hyperkalaemia (top bin), 24 h | 43782 | 0.809 | 0.825 | 0.95 | 0.730 / 0.723 | 0.798 / 0.842 | 0.858 / 0.875 | 0.865 / 0.827 | 0.797 / 0.764 | 0.571 / 0.621 | 0.810 / 0.809 |
| Thrombocytopenia, 48 h | 45806 | 0.940 | 0.939 | 1.00 | 0.930 / 0.901 | 0.842 / 0.886 | 0.957 / 0.964 | 0.980 / 0.963 | 0.899 / 0.965 | 0.918 / 0.904 | 0.928 / 0.940 |

Small-site augmentation (AUROC on that dataset's held-out anchors):

| Endpoint | Dataset | 300 real | 300 real + 300 twins | 300 real + 900 twins | all real |
|---|---|---|---|---|---|
| Decompensation, 24 h | INSPIRE | 0.698 | 0.726 | 0.655 | 0.835 |
| Decompensation, 24 h | NWICU | 0.854 | 0.897 | 0.923 | 0.938 |
| New IMV, 24 h | INSPIRE | 0.495 | 0.447 | 0.396 | 0.807 |
| New IMV, 24 h | NWICU | 0.715 | 0.783 | 0.731 | 0.788 |
| New vasopressor, 24 h | INSPIRE | 0.497 | 0.429 | 0.486 | 0.462 |
| New vasopressor, 24 h | NWICU | 0.520 | 0.774 | 0.790 | 0.831 |
| New RRT, 48 h | INSPIRE | 0.573 | 0.740 | 0.652 | 0.560 |
| New RRT, 48 h | NWICU | 0.958 | 0.917 | 0.947 | 0.977 |
| AKI stage >= 2, 48 h | INSPIRE | 0.533 | 0.375 | 0.478 | 0.839 |
| AKI stage >= 2, 48 h | NWICU | 0.745 | 0.837 | 0.804 | 0.824 |
| Sepsis (ASE), 24 h | INSPIRE | - | - | - | - |
| Sepsis (ASE), 24 h | NWICU | - | - | - | - |
| Hypotension, 4 h | INSPIRE | 0.883 | 0.885 | 0.901 | 0.965 |
| Hypotension, 4 h | NWICU | - | - | - | - |
| Hyperkalaemia (top bin), 24 h | INSPIRE | 0.836 | 0.827 | 0.822 | 0.824 |
| Hyperkalaemia (top bin), 24 h | NWICU | 0.572 | 0.579 | 0.604 | 0.614 |
| Thrombocytopenia, 48 h | INSPIRE | 0.951 | 0.947 | 0.954 | 0.963 |
| Thrombocytopenia, 48 h | NWICU | 0.444 | 0.317 | 0.903 | 0.926 |

Annotations: LightGBM on the A1 features; TSTR trained on the bank's twins, TRTR on as many real train rows; ratio of gains = (TSTR - 0.5) / (TRTR - 0.5). '-' in an augmentation arm: the bank has fewer eligible twins of that dataset than the arm needs. The cross-dataset matrices are in `baselines.json`.

## A5. Subgroups (`compare.json` per design)

### 24-hour landmark

| Endpoint | Attribute | Value | Anchors | Positives | Twin AUROC | Twin O/E | LightGBM AUROC |
|---|---|---|---|---|---|---|---|
| ICU death | sex | female | 1182 | 37 | 0.878 | 0.91 | 0.911 |
| ICU death | sex | male | 1711 | 65 | 0.844 | 1.33 | 0.917 |
| ICU death | age_band | 1 | 940 | 30 | 0.791 | 1.26 | 0.906 |
| ICU death | age_band | 2 | 1083 | 34 | 0.855 | 1.11 | 0.904 |
| ICU death | age_band | 3 | 507 | 31 | 0.921 | 1.05 | 0.918 |
| ICU death | admission_type | ed | 966 | 64 | 0.844 | 1.14 | 0.865 |
| ICU death | race | not recorded | 1573 | 48 | 0.880 | 1.20 | 0.953 |
| ICU death | race | white | 955 | 42 | 0.877 | 1.14 | 0.861 |
| In-hospital death | sex | female | 1390 | 127 | 0.870 | 1.18 | 0.867 |
| In-hospital death | sex | male | 2094 | 217 | 0.853 | 1.58 | 0.869 |
| In-hospital death | age_band | 1 | 1138 | 101 | 0.849 | 1.69 | 0.873 |
| In-hospital death | age_band | 2 | 1333 | 129 | 0.844 | 1.31 | 0.861 |
| In-hospital death | age_band | 3 | 594 | 93 | 0.864 | 1.23 | 0.832 |
| In-hospital death | dataset | aumc | 500 | 82 | 0.850 | 2.25 | 0.856 |
| In-hospital death | dataset | eicu | 500 | 39 | 0.913 | 1.13 | 0.838 |
| In-hospital death | dataset | mimic | 500 | 67 | 0.857 | 1.48 | 0.863 |
| In-hospital death | dataset | nwicu | 500 | 46 | 0.896 | 0.99 | 0.867 |
| In-hospital death | dataset | sicdb | 500 | 56 | 0.869 | 1.08 | 0.856 |
| In-hospital death | admission_type | UNK | 801 | 64 | 0.877 | 1.63 | 0.901 |
| In-hospital death | admission_type | ed | 1144 | 159 | 0.867 | 1.38 | 0.854 |
| In-hospital death | admission_type | elective | 905 | 47 | 0.818 | 1.52 | 0.895 |
| In-hospital death | admission_type | other | 485 | 60 | 0.801 | 1.11 | 0.763 |
| In-hospital death | race | not recorded | 2000 | 194 | 0.843 | 1.61 | 0.876 |
| In-hospital death | race | white | 1055 | 97 | 0.885 | 1.13 | 0.860 |
| ICU stay > 7 days | sex | female | 1381 | 170 | 0.793 | 0.79 | 0.857 |
| ICU stay > 7 days | sex | male | 2068 | 308 | 0.813 | 0.82 | 0.853 |
| ICU stay > 7 days | age_band | 0 | 433 | 60 | 0.799 | 0.83 | 0.853 |
| ICU stay > 7 days | age_band | 1 | 1128 | 178 | 0.836 | 0.89 | 0.876 |
| ICU stay > 7 days | age_band | 2 | 1317 | 183 | 0.798 | 0.80 | 0.835 |
| ICU stay > 7 days | age_band | 3 | 587 | 64 | 0.769 | 0.67 | 0.854 |
| ICU stay > 7 days | dataset | aumc | 500 | 142 | 0.846 | 0.94 | 0.861 |
| ICU stay > 7 days | dataset | eicu | 500 | 44 | 0.673 | 0.63 | 0.802 |
| ICU stay > 7 days | dataset | hirid | 499 | 58 | 0.727 | 0.80 | 0.775 |
| ICU stay > 7 days | dataset | mimic | 499 | 60 | 0.825 | 0.80 | 0.837 |
| ICU stay > 7 days | dataset | nwicu | 500 | 67 | 0.797 | 0.95 | 0.806 |
| ICU stay > 7 days | dataset | sicdb | 468 | 86 | 0.799 | 1.13 | 0.821 |
| ICU stay > 7 days | admission_type | UNK | 788 | 99 | 0.735 | 0.84 | 0.797 |
| ICU stay > 7 days | admission_type | direct | 144 | 30 | 0.825 | 1.15 | 0.878 |
| ICU stay > 7 days | admission_type | ed | 1143 | 175 | 0.812 | 0.82 | 0.826 |
| ICU stay > 7 days | admission_type | elective | 890 | 90 | 0.885 | 0.71 | 0.926 |
| ICU stay > 7 days | admission_type | other | 479 | 85 | 0.756 | 0.81 | 0.833 |
| ICU stay > 7 days | race | not recorded | 1966 | 314 | 0.824 | 0.82 | 0.871 |
| ICU stay > 7 days | race | white | 1055 | 104 | 0.765 | 0.75 | 0.810 |

### 48-hour landmark

| Endpoint | Attribute | Value | Anchors | Positives | Twin AUROC | Twin O/E | LightGBM AUROC |
|---|---|---|---|---|---|---|---|
| ICU death | sex | female | 1053 | 51 | 0.882 | 1.13 | 0.898 |
| ICU death | sex | male | 1584 | 76 | 0.882 | 1.27 | 0.907 |
| ICU death | age_band | 1 | 842 | 34 | 0.814 | 1.29 | 0.900 |
| ICU death | age_band | 2 | 1003 | 50 | 0.915 | 1.25 | 0.901 |
| ICU death | age_band | 3 | 455 | 36 | 0.875 | 1.13 | 0.856 |
| ICU death | dataset | aumc | 320 | 42 | 0.836 | 1.51 | 0.858 |
| ICU death | admission_type | ed | 954 | 69 | 0.868 | 1.24 | 0.868 |
| ICU death | race | not recorded | 1381 | 67 | 0.894 | 1.25 | 0.931 |
| ICU death | race | white | 926 | 39 | 0.862 | 1.09 | 0.864 |
| In-hospital death | sex | female | 1367 | 168 | 0.831 | 1.35 | 0.851 |
| In-hospital death | sex | male | 2118 | 287 | 0.829 | 1.60 | 0.865 |
| In-hospital death | age_band | 0 | 447 | 33 | 0.774 | 1.74 | 0.866 |
| In-hospital death | age_band | 1 | 1156 | 125 | 0.834 | 1.70 | 0.854 |
| In-hospital death | age_band | 2 | 1326 | 196 | 0.837 | 1.54 | 0.877 |
| In-hospital death | age_band | 3 | 571 | 101 | 0.806 | 1.19 | 0.800 |
| In-hospital death | dataset | aumc | 500 | 103 | 0.816 | 2.47 | 0.833 |
| In-hospital death | dataset | eicu | 500 | 43 | 0.890 | 1.14 | 0.893 |
| In-hospital death | dataset | hirid | 500 | 47 | 0.790 | 2.74 | 0.894 |
| In-hospital death | dataset | inspire | 500 | 42 | 0.785 | 1.26 | 0.833 |
| In-hospital death | dataset | mimic | 500 | 78 | 0.849 | 1.48 | 0.842 |
| In-hospital death | dataset | nwicu | 500 | 73 | 0.835 | 1.28 | 0.816 |
| In-hospital death | dataset | sicdb | 500 | 69 | 0.882 | 1.06 | 0.876 |
| In-hospital death | admission_type | UNK | 809 | 96 | 0.845 | 1.85 | 0.904 |
| In-hospital death | admission_type | ed | 1182 | 191 | 0.842 | 1.48 | 0.839 |
| In-hospital death | admission_type | elective | 775 | 63 | 0.768 | 1.56 | 0.846 |
| In-hospital death | admission_type | other | 579 | 81 | 0.787 | 1.12 | 0.822 |
| In-hospital death | race | black_or_african_american | 189 | 31 | 0.867 | 1.40 | 0.830 |
| In-hospital death | race | not recorded | 2000 | 261 | 0.815 | 1.66 | 0.867 |
| In-hospital death | race | white | 1073 | 123 | 0.860 | 1.26 | 0.870 |
| ICU stay > 7 days | sex | female | 1351 | 245 | 0.807 | 0.78 | 0.837 |
| ICU stay > 7 days | sex | male | 2083 | 432 | 0.790 | 0.86 | 0.830 |
| ICU stay > 7 days | age_band | 0 | 444 | 97 | 0.760 | 0.90 | 0.809 |
| ICU stay > 7 days | age_band | 1 | 1144 | 243 | 0.823 | 0.85 | 0.854 |
| ICU stay > 7 days | age_band | 2 | 1301 | 260 | 0.780 | 0.84 | 0.822 |
| ICU stay > 7 days | age_band | 3 | 560 | 86 | 0.813 | 0.72 | 0.844 |
| ICU stay > 7 days | dataset | aumc | 500 | 179 | 0.810 | 0.90 | 0.808 |
| ICU stay > 7 days | dataset | eicu | 500 | 57 | 0.785 | 0.69 | 0.830 |
| ICU stay > 7 days | dataset | hirid | 495 | 97 | 0.696 | 0.85 | 0.773 |
| ICU stay > 7 days | dataset | inspire | 499 | 74 | 0.797 | 0.57 | 0.905 |
| ICU stay > 7 days | dataset | mimic | 500 | 86 | 0.834 | 0.84 | 0.833 |
| ICU stay > 7 days | dataset | nwicu | 500 | 85 | 0.771 | 0.87 | 0.786 |
| ICU stay > 7 days | dataset | sicdb | 455 | 108 | 0.784 | 1.10 | 0.803 |
| ICU stay > 7 days | admission_type | UNK | 786 | 150 | 0.721 | 0.88 | 0.798 |
| ICU stay > 7 days | admission_type | direct | 134 | 32 | 0.808 | 0.91 | 0.823 |
| ICU stay > 7 days | admission_type | ed | 1182 | 220 | 0.836 | 0.81 | 0.833 |
| ICU stay > 7 days | admission_type | elective | 758 | 141 | 0.815 | 0.78 | 0.866 |
| ICU stay > 7 days | admission_type | other | 568 | 135 | 0.786 | 0.85 | 0.829 |
| ICU stay > 7 days | race | not recorded | 1949 | 458 | 0.787 | 0.85 | 0.835 |
| ICU stay > 7 days | race | white | 1073 | 147 | 0.784 | 0.80 | 0.821 |

### rolling times (enriched, weighted)

| Endpoint | Attribute | Value | Anchors | Positives | Twin AUROC | Twin O/E | LightGBM AUROC |
|---|---|---|---|---|---|---|---|
| Decompensation, 24 h | sex | female | 1326 | 308 | 0.908 | 1.74 | 0.954 |
| Decompensation, 24 h | sex | male | 2152 | 428 | 0.880 | 1.89 | 0.955 |
| Decompensation, 24 h | age_band | 0 | 414 | 68 | 0.916 | 1.74 | 0.981 |
| Decompensation, 24 h | age_band | 1 | 1097 | 212 | 0.861 | 1.99 | 0.961 |
| Decompensation, 24 h | age_band | 2 | 1260 | 246 | 0.910 | 1.97 | 0.960 |
| Decompensation, 24 h | age_band | 3 | 715 | 216 | 0.890 | 1.59 | 0.921 |
| Decompensation, 24 h | dataset | aumc | 499 | 154 | 0.892 | 1.90 | 0.954 |
| Decompensation, 24 h | dataset | eicu | 500 | 125 | 0.828 | 1.87 | 0.908 |
| Decompensation, 24 h | dataset | hirid | 500 | 91 | 0.885 | 1.80 | 0.952 |
| Decompensation, 24 h | dataset | inspire | 500 | 51 | 0.912 | 2.21 | 0.980 |
| Decompensation, 24 h | dataset | mimic | 500 | 125 | 0.927 | 1.91 | 0.962 |
| Decompensation, 24 h | dataset | nwicu | 500 | 117 | 0.896 | 2.09 | 0.942 |
| Decompensation, 24 h | dataset | sicdb | 487 | 79 | 0.873 | 1.23 | 0.957 |
| Decompensation, 24 h | admission_type | UNK | 803 | 142 | 0.860 | 1.69 | 0.940 |
| Decompensation, 24 h | admission_type | ed | 1203 | 359 | 0.909 | 2.02 | 0.954 |
| Decompensation, 24 h | admission_type | elective | 707 | 63 | 0.890 | 1.69 | 0.981 |
| Decompensation, 24 h | admission_type | other | 614 | 141 | 0.875 | 1.60 | 0.940 |
| Decompensation, 24 h | race | black_or_african_american | 198 | 38 | 0.874 | 1.21 | 0.937 |
| Decompensation, 24 h | race | not recorded | 1986 | 375 | 0.893 | 1.73 | 0.964 |
| Decompensation, 24 h | race | unknown | 109 | 33 | 0.867 | 1.73 | 0.927 |
| Decompensation, 24 h | race | white | 1051 | 268 | 0.892 | 2.11 | 0.939 |
| New IMV, 24 h | sex | female | 957 | 78 | 0.827 | 0.79 | 0.859 |
| New IMV, 24 h | sex | male | 1480 | 157 | 0.825 | 0.64 | 0.785 |
| New IMV, 24 h | age_band | 0 | 283 | 34 | 0.751 | 0.66 | 0.787 |
| New IMV, 24 h | age_band | 1 | 744 | 81 | 0.816 | 0.77 | 0.779 |
| New IMV, 24 h | age_band | 2 | 887 | 90 | 0.863 | 0.78 | 0.872 |
| New IMV, 24 h | age_band | 3 | 527 | 31 | 0.815 | 0.42 | 0.783 |
| New IMV, 24 h | dataset | eicu | 350 | 45 | 0.795 | 1.09 | 0.885 |
| New IMV, 24 h | dataset | hirid | 275 | 33 | 0.698 | 0.63 | 0.767 |
| New IMV, 24 h | dataset | mimic | 359 | 36 | 0.935 | 0.53 | 0.817 |
| New IMV, 24 h | dataset | nwicu | 455 | 36 | 0.860 | 0.36 | 0.835 |
| New IMV, 24 h | dataset | sicdb | 395 | 36 | 0.945 | 0.99 | 0.833 |
| New IMV, 24 h | admission_type | UNK | 515 | 70 | 0.828 | 0.83 | 0.816 |
| New IMV, 24 h | admission_type | ed | 863 | 80 | 0.857 | 0.41 | 0.766 |
| New IMV, 24 h | admission_type | other | 448 | 55 | 0.714 | 1.11 | 0.790 |
| New IMV, 24 h | race | not recorded | 1277 | 119 | 0.801 | 0.74 | 0.811 |
| New IMV, 24 h | race | white | 821 | 69 | 0.820 | 0.58 | 0.830 |
| New vasopressor, 24 h | sex | female | 1011 | 120 | 0.841 | 0.42 | 0.776 |
| New vasopressor, 24 h | sex | male | 1713 | 236 | 0.867 | 0.54 | 0.814 |
| New vasopressor, 24 h | age_band | 0 | 353 | 33 | 0.922 | 0.37 | 0.921 |
| New vasopressor, 24 h | age_band | 1 | 878 | 116 | 0.849 | 0.46 | 0.821 |
| New vasopressor, 24 h | age_band | 2 | 966 | 139 | 0.868 | 0.63 | 0.782 |
| New vasopressor, 24 h | age_band | 3 | 533 | 69 | 0.803 | 0.38 | 0.712 |
| New vasopressor, 24 h | dataset | aumc | 297 | 48 | 0.884 | 0.55 | 0.824 |
| New vasopressor, 24 h | dataset | eicu | 448 | 48 | 0.910 | 0.63 | 0.800 |
| New vasopressor, 24 h | dataset | hirid | 416 | 57 | 0.693 | 0.46 | 0.690 |
| New vasopressor, 24 h | dataset | inspire | 418 | 47 | 0.701 | 0.60 | 0.654 |
| New vasopressor, 24 h | dataset | mimic | 393 | 38 | 0.942 | 0.54 | 0.875 |
| New vasopressor, 24 h | dataset | nwicu | 426 | 39 | 0.874 | 0.28 | 0.861 |
| New vasopressor, 24 h | dataset | sicdb | 332 | 80 | 0.769 | 0.56 | 0.706 |
| New vasopressor, 24 h | admission_type | UNK | 660 | 88 | 0.761 | 0.45 | 0.692 |
| New vasopressor, 24 h | admission_type | ed | 931 | 90 | 0.903 | 0.44 | 0.847 |
| New vasopressor, 24 h | admission_type | elective | 532 | 90 | 0.892 | 0.54 | 0.865 |
| New vasopressor, 24 h | admission_type | other | 495 | 72 | 0.824 | 0.49 | 0.793 |
| New vasopressor, 24 h | race | not recorded | 1463 | 232 | 0.816 | 0.52 | 0.778 |
| New vasopressor, 24 h | race | white | 910 | 88 | 0.920 | 0.42 | 0.831 |
| New RRT, 48 h | sex | female | 1252 | 107 | 0.892 | 0.68 | 0.891 |
| New RRT, 48 h | sex | male | 2025 | 189 | 0.965 | 0.66 | 0.959 |
| New RRT, 48 h | age_band | 0 | 388 | 36 | 0.960 | 0.60 | 0.954 |
| New RRT, 48 h | age_band | 1 | 1039 | 107 | 0.961 | 0.74 | 0.967 |
| New RRT, 48 h | age_band | 2 | 1176 | 114 | 0.904 | 0.73 | 0.880 |
| New RRT, 48 h | age_band | 3 | 682 | 39 | 0.960 | 0.43 | 0.943 |
| New RRT, 48 h | dataset | aumc | 462 | 37 | 0.912 | 0.44 | 0.959 |
| New RRT, 48 h | dataset | hirid | 467 | 100 | 0.907 | 0.98 | 0.857 |
| New RRT, 48 h | dataset | mimic | 483 | 40 | 0.971 | 0.52 | 0.961 |
| New RRT, 48 h | dataset | nwicu | 488 | 40 | 0.986 | 0.80 | 0.975 |
| New RRT, 48 h | admission_type | UNK | 753 | 119 | 0.922 | 1.00 | 0.890 |
| New RRT, 48 h | admission_type | ed | 1152 | 87 | 0.973 | 0.58 | 0.956 |
| New RRT, 48 h | admission_type | elective | 661 | 38 | 0.948 | 0.30 | 0.964 |
| New RRT, 48 h | admission_type | other | 567 | 42 | 0.902 | 0.40 | 0.941 |
| New RRT, 48 h | race | black_or_african_american | 186 | 31 | 0.966 | 0.88 | 0.962 |
| New RRT, 48 h | race | not recorded | 1838 | 188 | 0.918 | 0.65 | 0.920 |
| New RRT, 48 h | race | white | 1020 | 56 | 0.977 | 0.68 | 0.959 |
| AKI stage >= 2, 48 h | sex | female | 1135 | 249 | 0.906 | 0.70 | 0.873 |
| AKI stage >= 2, 48 h | sex | male | 1780 | 398 | 0.888 | 0.75 | 0.848 |
| AKI stage >= 2, 48 h | age_band | 0 | 375 | 50 | 0.927 | 0.65 | 0.893 |
| AKI stage >= 2, 48 h | age_band | 1 | 938 | 188 | 0.865 | 0.67 | 0.809 |
| AKI stage >= 2, 48 h | age_band | 2 | 1034 | 276 | 0.901 | 0.82 | 0.874 |
| AKI stage >= 2, 48 h | age_band | 3 | 575 | 134 | 0.900 | 0.69 | 0.855 |
| AKI stage >= 2, 48 h | dataset | aumc | 381 | 109 | 0.875 | 0.78 | 0.786 |
| AKI stage >= 2, 48 h | dataset | eicu | 438 | 86 | 0.863 | 0.86 | 0.775 |
| AKI stage >= 2, 48 h | dataset | hirid | 448 | 142 | 0.866 | 0.68 | 0.871 |
| AKI stage >= 2, 48 h | dataset | inspire | 438 | 33 | 0.864 | 0.41 | 0.896 |
| AKI stage >= 2, 48 h | dataset | mimic | 383 | 115 | 0.842 | 0.64 | 0.783 |
| AKI stage >= 2, 48 h | dataset | sicdb | 380 | 150 | 0.854 | 0.85 | 0.770 |
| AKI stage >= 2, 48 h | admission_type | UNK | 703 | 207 | 0.877 | 0.73 | 0.849 |
| AKI stage >= 2, 48 h | admission_type | direct | 105 | 30 | 0.897 | 0.73 | 0.795 |
| AKI stage >= 2, 48 h | admission_type | ed | 1000 | 185 | 0.905 | 0.69 | 0.853 |
| AKI stage >= 2, 48 h | admission_type | elective | 596 | 112 | 0.909 | 0.75 | 0.899 |
| AKI stage >= 2, 48 h | admission_type | other | 498 | 109 | 0.864 | 0.81 | 0.842 |
| AKI stage >= 2, 48 h | race | not recorded | 1647 | 434 | 0.890 | 0.75 | 0.862 |
| AKI stage >= 2, 48 h | race | white | 892 | 148 | 0.898 | 0.70 | 0.852 |
| Hypotension, 4 h | sex | female | 1152 | 102 | 0.932 | 0.85 | 0.931 |
| Hypotension, 4 h | sex | male | 1851 | 171 | 0.894 | 1.19 | 0.922 |
| Hypotension, 4 h | age_band | 1 | 976 | 81 | 0.919 | 1.15 | 0.927 |
| Hypotension, 4 h | age_band | 2 | 1080 | 107 | 0.900 | 1.16 | 0.911 |
| Hypotension, 4 h | age_band | 3 | 570 | 78 | 0.900 | 0.93 | 0.922 |
| Hypotension, 4 h | dataset | aumc | 439 | 40 | 0.904 | 0.83 | 0.900 |
| Hypotension, 4 h | dataset | eicu | 427 | 49 | 0.886 | 1.09 | 0.914 |
| Hypotension, 4 h | dataset | hirid | 369 | 46 | 0.884 | 1.13 | 0.850 |
| Hypotension, 4 h | dataset | mimic | 444 | 41 | 0.841 | 1.12 | 0.908 |
| Hypotension, 4 h | dataset | sicdb | 364 | 76 | 0.893 | 1.07 | 0.908 |
| Hypotension, 4 h | admission_type | UNK | 601 | 74 | 0.870 | 1.11 | 0.865 |
| Hypotension, 4 h | admission_type | ed | 1098 | 69 | 0.898 | 0.85 | 0.953 |
| Hypotension, 4 h | admission_type | elective | 660 | 47 | 0.924 | 1.00 | 0.930 |
| Hypotension, 4 h | admission_type | other | 518 | 68 | 0.943 | 1.23 | 0.953 |
| Hypotension, 4 h | race | not recorded | 1640 | 184 | 0.908 | 1.01 | 0.909 |
| Hypotension, 4 h | race | white | 960 | 67 | 0.913 | 1.05 | 0.952 |
| Hyperkalaemia (top bin), 24 h | sex | female | 1168 | 92 | 0.771 | 0.86 | 0.794 |
| Hyperkalaemia (top bin), 24 h | sex | male | 1891 | 191 | 0.825 | 0.74 | 0.849 |
| Hyperkalaemia (top bin), 24 h | age_band | 1 | 960 | 75 | 0.817 | 0.65 | 0.853 |
| Hyperkalaemia (top bin), 24 h | age_band | 2 | 1106 | 125 | 0.773 | 0.90 | 0.810 |
| Hyperkalaemia (top bin), 24 h | age_band | 3 | 628 | 61 | 0.824 | 0.78 | 0.857 |
| Hyperkalaemia (top bin), 24 h | dataset | aumc | 438 | 52 | 0.901 | 0.75 | 0.906 |
| Hyperkalaemia (top bin), 24 h | dataset | eicu | 440 | 30 | 0.814 | 1.14 | 0.812 |
| Hyperkalaemia (top bin), 24 h | dataset | hirid | 436 | 38 | 0.853 | 1.05 | 0.889 |
| Hyperkalaemia (top bin), 24 h | dataset | inspire | 453 | 41 | 0.729 | 0.38 | 0.867 |
| Hyperkalaemia (top bin), 24 h | dataset | mimic | 414 | 49 | 0.799 | 0.84 | 0.810 |
| Hyperkalaemia (top bin), 24 h | dataset | sicdb | 439 | 54 | 0.650 | 0.89 | 0.666 |
| Hyperkalaemia (top bin), 24 h | admission_type | UNK | 712 | 59 | 0.816 | 0.93 | 0.844 |
| Hyperkalaemia (top bin), 24 h | admission_type | ed | 1031 | 94 | 0.798 | 0.74 | 0.812 |
| Hyperkalaemia (top bin), 24 h | admission_type | elective | 643 | 65 | 0.857 | 0.63 | 0.867 |
| Hyperkalaemia (top bin), 24 h | admission_type | other | 537 | 52 | 0.725 | 0.94 | 0.763 |
| Hyperkalaemia (top bin), 24 h | race | not recorded | 1766 | 185 | 0.803 | 0.76 | 0.845 |
| Hyperkalaemia (top bin), 24 h | race | white | 926 | 65 | 0.791 | 0.77 | 0.783 |
| Thrombocytopenia, 48 h | sex | female | 1149 | 42 | 0.969 | 0.77 | 0.958 |
| Thrombocytopenia, 48 h | sex | male | 1905 | 107 | 0.919 | 0.84 | 0.951 |
| Thrombocytopenia, 48 h | age_band | 1 | 932 | 52 | 0.975 | 0.81 | 0.961 |
| Thrombocytopenia, 48 h | age_band | 2 | 1109 | 55 | 0.883 | 0.95 | 0.938 |
| Thrombocytopenia, 48 h | dataset | inspire | 411 | 49 | 0.966 | 1.25 | 0.958 |
| Thrombocytopenia, 48 h | admission_type | ed | 1060 | 37 | 0.895 | 0.79 | 0.931 |
| Thrombocytopenia, 48 h | admission_type | elective | 629 | 45 | 0.977 | 0.83 | 0.966 |
| Thrombocytopenia, 48 h | admission_type | other | 526 | 35 | 0.974 | 1.21 | 0.974 |
| Thrombocytopenia, 48 h | race | not recorded | 1736 | 107 | 0.978 | 0.83 | 0.972 |

Annotations: subgroups with fewer than 30 positives or negatives are not scored; race is recorded in MIMIC, eICU and NWICU only. Age bands: 0 = 18-44, 1 = 45-64, 2 = 65-79, 3 = 80+.

## B2. Unknown site (`runs/paper/a1/test_L24/compare.json`)

| Endpoint | Anchors | AUROC, site known | AUROC, site unknown | O/E known | O/E unknown |
|---|---|---|---|---|---|
| In-hospital death | 2045 | 0.851 | 0.862 | 1.40 | 1.36 |
| ICU death | 1615 | 0.851 | 0.882 | 1.08 | 1.03 |
| ICU stay > 7 days | 2080 | 0.814 | 0.846 | 0.77 | 0.74 |

Annotations: the same anchors (B2's subset of the L24 anchors) with the prompt's dataset and site tokens set to unknown; the rollouts are decoded with their stay's real dataset and site. A proxy only: no dataset was held out of training.

## B3. Ablations across checkpoints (inference only)

| Checkpoint | Optimizer | Tuning NLL | P1 AUC pooled | Probe AUROC ICU death (L24) | Probe AUROC new IMV 24 h | Probe AUROC new RRT 48 h | Teacher-forced O/E IMV after 24 h | Teacher-forced O/E RRT after 24 h |
|---|---|---|---|---|---|---|---|---|
| M32_main | adamw | 1.5611 | 0.530 | 0.907 | 0.823 | 0.937 | 0.68 | 0.39 |
| ML32_main | adamw | 1.5631 | 0.539 | 0.897 | 0.807 | 0.926 | - | - |
| MC32_main | adamw | 1.5636 | 0.530 | 0.908 | 0.826 | 0.922 | 0.72 | 0.41 |
| MCN32_main | adamw | 1.6067 | 0.521 | 0.913 | 0.821 | 0.921 | 0.70 | 0.36 |
| MU32_main | muon | 1.5274 | 0.554 | 0.898 | 0.837 | 0.933 | 0.79 | 0.51 |
| MUH32_main | muon | 1.5092 | 0.564 | 0.903 | 0.821 | 0.938 | 0.80 | 0.55 |
| MUF32_main | muon | 1.5388 | 0.549 | 0.906 | 0.812 | 0.934 | 0.78 | 0.52 |
| MUC32_main | muon | 1.5525 | 0.538 | 0.901 | 0.828 | 0.934 | 0.76 | 0.44 |

Annotations: M32 (32k context, AdamW), ML32 (2.2x larger), MC32 (ICU-day clock blocks), MCN32 (+ input noise), MU32/MUH32 (Muon, two rates), MUF32/MUC32 (Muon, passes on the privacy frontier; MUC32 adopted). M32 and ML32 read r3 tokens; ML32 has no teacher-forced audit. S and S_main (older tokens) are not included.

## C1. What-if rollouts (`runs/paper/c1/c1.json`)

| Treatment | Response | Expected | Prefixes | Share expected sign | Mean effect (give - withhold) | 95% CI | Withhold mean | Dose arms (mean response) |
|---|---|---|---|---|---|---|---|---|
| norepinephrine | map | + | 173 | 0.45 | -0.806 | -1.567 to -0.143 | 76.80 | high 76.2; low 76.1; median 76.0 |
| insulin | glucose | - | 200 | 0.63 | -14.157 | -22.592 to -6.378 | 196.79 | high 177.2; low 176.3; median 182.6 |
| crrt | potassium | - | 200 | 0.60 | -0.095 | -0.173 to -0.024 | 4.75 | - |
| crrt | creatinine_24h | - | 191 | 0.58 | -0.173 | -0.453 to 0.125 | 3.24 | - |
| furosemide | urine_output | + | 52 | 1.00 | 723.347 | 626.754 to 833.615 | 204.42 | - |
| imv | fio2_charted | + | 200 | 0.93 | 2.458 | 2.167 to 2.768 | 1.22 | - |
| imv | spo2 | + | 194 | 0.62 | 0.384 | 0.234 to 0.543 | 97.32 | - |
| fluid_bolus | map | + | 193 | 0.45 | -0.211 | -0.809 to 0.429 | 79.12 | - |
| fluid_bolus | urine_output | + | 186 | 0.52 | 30.535 | -23.232 to 86.514 | 307.33 | - |

Annotations: 16 rollouts per arm and prefix; response windows 6 h (creatinine 24 h); urine output without records above 2,000 mL (amendment 6). Effects are twin-world: they inherit the data's confounding by indication (norepinephrine starts accompany falling pressure), so they are not treatment-effect estimates.

## C2. Attribution (`runs/paper/c2/test_<design>/c2.json`)

### 24-hour landmark

| Endpoint | Anchors | Deletion drop: editing | IG | Attention rollout | Attention x grad | Random | IG vs editing (rho / top-3) | Rollout vs editing | Attn x grad vs editing | IG completeness error |
|---|---|---|---|---|---|---|---|---|---|---|
| ICU death | 200 | 1.883 | 0.815 | 0.673 | 0.165 | 0.079 | 0.16 / 0.37 | 0.13 / 0.35 | 0.05 / 0.31 | 0.03 |

Groups most often among the top 3 (share of anchors, mean edit effect on the probe's logit):

| Endpoint | Groups |
|---|---|
| ICU death | statics (0.62, 0.65); vitals\|6-24h (0.38, 0.26); vitals\|0-6h (0.23, -0.09); respiratory\|6-24h (0.21, 0.17); other_meds\|6-24h (0.18, 0.12); other\|6-24h (0.17, 0.04) |

| Endpoint | Anchors | Sign agreement probe vs rollouts | Rank correlation (median) |
|---|---|---|---|
| ICU death | 21 | 0.59 | 0.40 |

### rolling times (enriched, weighted)

| Endpoint | Anchors | Deletion drop: editing | IG | Attention rollout | Attention x grad | Random | IG vs editing (rho / top-3) | Rollout vs editing | Attn x grad vs editing | IG completeness error |
|---|---|---|---|---|---|---|---|---|---|---|
| AKI stage >= 2, 48 h | 200 | 2.008 | 0.542 | 0.319 | 0.853 | 0.418 | 0.12 / 0.30 | -0.07 / 0.24 | 0.31 / 0.43 | 0.02 |
| Hypotension, 4 h | 200 | 2.663 | 0.504 | 0.177 | 1.711 | 0.460 | 0.05 / 0.27 | -0.05 / 0.22 | 0.24 / 0.47 | 0.02 |
| New IMV, 24 h | 200 | 2.210 | 0.816 | 0.624 | 1.004 | 0.532 | 0.13 / 0.34 | -0.06 / 0.27 | 0.30 / 0.48 | 0.01 |

Groups most often among the top 3 (share of anchors, mean edit effect on the probe's logit):

| Endpoint | Groups |
|---|---|
| AKI stage >= 2, 48 h | vitals\|0-6h (0.51, 0.53); fluids_output\|0-6h (0.28, 0.37); statics (0.24, -0.01); labs\|0-6h (0.23, 0.28); labs\|6-24h (0.20, 0.17); fluids_output\|>24h (0.19, 0.37) |
| Hypotension, 4 h | vitals\|0-6h (0.80, 1.55); fluids_output\|0-6h (0.23, 0.22); other\|0-6h (0.21, 0.20); vitals\|6-24h (0.21, 0.18); other_meds\|0-6h (0.20, 0.17); labs\|0-6h (0.17, -0.00) |
| New IMV, 24 h | vitals\|0-6h (0.39, 0.24); labs\|0-6h (0.32, 0.38); other\|0-6h (0.28, 0.29); fluids_output\|0-6h (0.26, 0.41); statics (0.26, 0.04); other_meds\|0-6h (0.23, 0.32) |

| Endpoint | Anchors | Sign agreement probe vs rollouts | Rank correlation (median) |
|---|---|---|---|
| AKI stage >= 2, 48 h | 23 | 0.55 | 0.17 |
| Hypotension, 4 h | 6 | 0.63 | 0.41 |
| New IMV, 24 h | 22 | 0.57 | 0.13 |

Annotations: the target is the Twin probe's logit; deletion drop = mean fall of the logit when a method's top 1-5 groups are removed (higher = more faithful), against a random order. Attention methods cover prompts up to 4,096 tokens. Anchors: the 100 with the highest probe risk plus 100 at random per endpoint. Rollout check: the first 50 anchors per endpoint, original and its 3 highest and 3 lowest edits rolled out 16 times each; an anchor is scored when its original and at least 4 edits each have at least 8 rollouts with a valid label (eligible, horizon reached), so the scored anchors are few. Attributions are causal within the model only.

## C3. Virtual cohorts (`runs/paper/c3/c3.json`)

| Dataset | Set | Stays | Age | Female | ICU death | Hospital death | IMV | Vasoactive | RRT | Stay < 1 h | Events / h (median) | Labs / day (median) | LOS h (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AmsterdamUMCdb | real | 2000 | 60.388 | 0.317 | 0.129 | 0.129 | 0.868 | 0.706 | 0.068 | 0.002 | 23.856 | 142.791 | 31.4 |
| AmsterdamUMCdb | twins | 1203 | 62.047 | 0.313 | 0.058 | 0.060 | 0.842 | 0.652 | 0.044 | 0.006 | 23.202 | 146.563 | 29.6 |
| AmsterdamUMCdb | control | 2000 | 61.105 | 0.323 | 0.130 | 0.130 | 0.861 | 0.688 | 0.054 | 0.001 | 23.554 | 134.497 | 32.0 |
| eICU | real | 2000 | 63.688 | 0.467 | 0.048 | 0.084 | 0.246 | 0.115 | 0.042 | 0.021 | 11.824 | 44.709 | 38.6 |
| eICU | twins | 3975 | 63.914 | 0.462 | 0.039 | 0.072 | 0.233 | 0.114 | 0.040 | 0.016 | 11.875 | 44.223 | 36.9 |
| eICU | control | 2000 | 63.842 | 0.454 | 0.054 | 0.088 | 0.254 | 0.133 | 0.042 | 0.016 | 12.041 | 43.776 | 36.4 |
| HiRID | real | 2000 | 65.555 | 0.354 | 0.058 | 0.058 | 0.565 | 0.312 | 0.060 | 0.001 | 19.363 | 68.254 | 24.0 |
| HiRID | twins | 1601 | 67.057 | 0.344 | 0.031 | 0.031 | 0.528 | 0.294 | 0.049 | 0.000 | 18.038 | 66.747 | 24.1 |
| HiRID | control | 2000 | 66.100 | 0.328 | 0.059 | 0.059 | 0.580 | 0.292 | 0.051 | 0.000 | 20.061 | 68.205 | 23.9 |
| INSPIRE | real | 2000 | 62.114 | 0.414 | 0.012 | 0.056 | 0.960 | 0.236 | 0.025 | 0.001 | 17.754 | 124.194 | 25.6 |
| INSPIRE | twins | 1066 | 62.284 | 0.436 | 0.007 | 0.043 | 0.962 | 0.235 | 0.041 | 0.002 | 18.476 | 127.135 | 25.8 |
| INSPIRE | control | 2000 | 62.094 | 0.419 | 0.006 | 0.043 | 0.960 | 0.227 | 0.026 | 0.000 | 17.635 | 121.790 | 25.8 |
| MIMIC-IV | real | 2000 | 66.021 | 0.433 | 0.088 | 0.131 | 0.401 | 0.325 | 0.062 | 0.002 | 14.434 | 56.590 | 47.1 |
| MIMIC-IV | twins | 2608 | 65.484 | 0.475 | 0.057 | 0.102 | 0.368 | 0.285 | 0.060 | 0.002 | 13.926 | 55.547 | 45.6 |
| MIMIC-IV | control | 2000 | 64.787 | 0.434 | 0.080 | 0.130 | 0.403 | 0.319 | 0.060 | 0.001 | 14.451 | 56.858 | 46.8 |
| NWICU | real | 2000 | 62.775 | 0.432 | 0.067 | 0.103 | 0.158 | 0.213 | 0.044 | 0.004 | 6.544 | 43.904 | 46.5 |
| NWICU | twins | 1492 | 63.992 | 0.435 | 0.061 | 0.101 | 0.133 | 0.212 | 0.049 | 0.001 | 6.424 | 42.945 | 48.0 |
| NWICU | control | 2000 | 63.571 | 0.440 | 0.076 | 0.111 | 0.152 | 0.217 | 0.052 | 0.002 | 6.444 | 43.828 | 46.4 |
| SICdb | real | 2000 | 68.698 | 0.376 | 0.040 | 0.087 | 0.202 | 0.565 | 0.041 | 0.002 | 18.356 | 142.606 | 36.6 |
| SICdb | twins | 1471 | 68.875 | 0.368 | 0.037 | 0.073 | 0.193 | 0.550 | 0.026 | 0.002 | 17.927 | 146.053 | 32.1 |
| SICdb | control | 2000 | 68.300 | 0.382 | 0.039 | 0.076 | 0.195 | 0.542 | 0.031 | 0.005 | 18.261 | 141.749 | 35.3 |

Inter-dataset structure (correlation of profile distances with real): twins 0.73, real control 0.95. eICU hospitals, total variation from real: twins 0.146, control 0.145. Rare strata: 25 strata, share twins 0.022 against real 0.015, all reached: 25; their ICU death twins 0.047 against real 0.095.

Annotations: twins from the internal bank (round-3 sampler), real = held-out stays, control = real train stays against the same held-out stays. The death gap is the generator's known one.

## C4. Patient similarity (`runs/paper/c4/c4.json`)

| Endpoint | Prevalence | Positive neighbours of positives: model state | TF-IDF | Lift: model state | TF-IDF |
|---|---|---|---|---|---|
| ICU death | 0.069 | 0.222 | 0.200 | 3.24 | 2.92 |
| New IMV, 24 h | 0.033 | 0.080 | 0.066 | 2.47 | 2.03 |
| New RRT, 48 h | 0.028 | 0.122 | 0.090 | 4.42 | 3.29 |

Annotations: 3500 L24 anchors, 10 nearest other stays by cosine.
