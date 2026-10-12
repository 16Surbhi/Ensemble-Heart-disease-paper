# Reproducibility and reporting notes

- The output files under `reference_outputs_v2/` are the **saved** outputs of the October 2026 Revision 2 Colab run, not newly generated results from this packaging operation.
- The main experiment and embedded Colab script were compared byte-for-byte before release packaging.
- Underlying data are the four historical UCI files, which are excluded from this repository. Their raw SHA-256 hashes are hard-coded in the script and recorded in `dataset_manifest.csv`.
- The `raw_input_integrity_report.json` and `statlog_cleveland_overlap_report.csv` were produced by a separate input verification run. Their inclusion documents *reported* input checks, not a separate download/reanalysis during release packaging.
- `reference_outputs_v2/out_of_fold_predictions.csv` contains patient-row indices, labels, and predictions derived from publicly released benchmark records; it has no names or direct identifiers. Check applicable dataset conditions before public release.
- Computational times depend on hardware, library versions, and concurrent workload. Do not treat saved times as hardware-independent speed benchmarks.
- Only the original dataset level IDs and source records were used; no hospital deployment or prospective clinical validation occurred.
- Logistic-regression meta-learner variants were added post hoc; confidence intervals are descriptive/exploratory.
- Saved notebook output cells were not re-executed during packaging; open the notebook and run it for fresh results in a supported environment.
- The original script's `--no-hash-check` option is for debugging only. Never disable hash checks when claiming to reproduce the paper.

## Output dictionary (selected)

- `fold_level_metrics.csv`: each outer fold × dataset × model, including training/prediction times and warning counts.
- `out_of_fold_predictions.csv`: each held-out prediction keyed by dataset, shuffle seed, fold, model, and within-dataset row index.
- `summary_mean_sd.csv`: model-by-dataset averages and SDs across overlapping outer test folds.
- `paired_vs_logistic_regression.csv`: exploratory corrected-t and patient bootstrap intervals.
- `shap_values_all_patients.csv`: original-feature out-of-fold SHAP values for Cleveland, seed 42.
- `shap_mean_abs_by_fold.csv`: fold-wise average absolute attribution values.
- `environment.json`: versions recorded from the revision run (exact SciPy, SHAP, and Matplotlib versions were not logged).
