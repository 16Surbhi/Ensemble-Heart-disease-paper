# Heart disease heterogeneous stacking: reproducible Revision 2 experiments

Research code and saved experimental outputs supporting the paper **“Does Heterogeneous Stacking Help in Heart Disease Classification? A Leakage-Controlled, Four-Dataset Evaluation with Explainable AI.”**

**Scope.** Four historical public UCI benchmark files (Cleveland, Statlog Heart, Hungarian, Swiss); 10 fixed classifiers/stacking configurations; repeated, leakage-controlled stratified five-fold evaluation; supplementary statistical and SHAP analyses. This repository is **research software**, not a clinical diagnostic product.

## Repository contents

| Location | Contents |
|---|---|
| `experiments/heart_stacking_v2.py` | **Canonical Revision 2 script**: download, SHA-256 checks, preprocessing, all 10 models, repeated CV, statistics, figures, SHAP |
| `notebooks/Heart_Stacking_Revision2.ipynb` | Equivalent Colab notebook; its embedded experiment script matches the canonical `.py` source |
| `verification/verify_raw_inputs_and_overlap.py` | Independent hash and exact Cleveland–Statlog row-matching verification |
| `notebooks/Verify_Heart_Input_Files_and_Overlap.ipynb` | The same raw-input verification workflow as a Colab notebook |
| `verification/check_result_integrity.py` | Independent validation of saved per-fold metrics against out-of-fold predictions |
| `reference_outputs_v2/` | Saved Revision 2 experiment results, model metrics, out-of-fold probabilities, plots, SHAP values, environment and verification reports |
| `docs/` | Reproducibility details and release/ethics reminders |

The **original UCI raw data files are NOT redistributed**. The analysis script downloads the four files from their cited public source URLs and refuses to proceed if their SHA-256 digests differ from the recorded dataset manifest.

## Quick start: Google Colab

1. Upload `notebooks/Heart_Stacking_Revision2.ipynb` to Google Colab.
2. Use **Runtime → Run all**. The notebook installs dependencies, writes the experimental script, runs the analysis, and produces a downloadable `outputs_v2.zip`.
3. For raw input provenance and exact record overlap, also run `notebooks/Verify_Heart_Input_Files_and_Overlap.ipynb`.

## Local installation

The recorded execution environment used **Python 3.13.16**, **scikit-learn 1.6.1**, **XGBoost 3.4.1**, **NumPy 2.1.3**, **pandas 2.2.3**. Some other installed library versions were not recorded; see `reference_outputs_v2/environment.json` and `docs/REPRODUCIBILITY_NOTES.md`.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/heart_stacking_v2.py --full --shap
```

The full experiment uses 3 shuffle seeds (42, 52, 62), 5 outer folds per seed, inner 3-fold out-of-fold stacking, fixed hyperparameters, and a feed-forward MLP meta-learner with `max_iter=2000` (**not LSTM**). Results are written to the `outputs_v2/` directory. The quick dry-run option is `--quick` and uses only one 3-fold split; **quick mode must not be used for the publication tables**.

If you already have the raw UCI files locally, set `--data-dir data --no-fetch`. To examine a data-handling sensitivity, run `--no-zeros-as-missing` with a **different** `--output-dir`. The default treats zero `chol`/`trestbps` as missing. Swiss has 123 zero cholesterol values; these are recorded in the manifest. The Hungarian duplicated feature row was retained in the primary experiments.

Independent verification:

```bash
python verification/verify_raw_inputs_and_overlap.py --download
python verification/check_result_integrity.py reference_outputs_v2
# After a fresh run:
python verification/check_result_integrity.py outputs_v2
```

The `verify_raw_inputs_and_overlap.py` report is written in the **current working directory**, alongside the overlap CSV. Matching requires equality across the 13 numerical feature values and mapped binary labels. All 270 Statlog examples matched Cleveland in the recorded input verification, and **must not be described as independent external validation**.

## Model and statistical interpretation

The revision includes baseline dummy, Logistic Regression, KNN, Random Forest, XGBoost, KNN/RF stacks with XGBoost/MLP/LR meta-learners, a fixed equal-weight XGBoost+MLP fusion, and a KNN/RF/LR→LR stack. The LR-meta variants were added **after the initial eight-model benchmark** and are exploratory. The observed ~0.005 mean ROC-AUC gains on Cleveland and Statlog are small; corrected resampled-t intervals include zero, and Statlog duplicates records in Cleveland. Swiss has only eight negative cases. No model should be presented as clinically validated.

The repeated CV folds are correlated. The corrected resampled-t intervals and patient-level bootstrap results are **exploratory uncertainty summaries**, not proof of superiority; the bootstrap does not capture variation in fitted training samples. SHAP values are model explanations, not causal clinical inference.

`reference_outputs_v2/convergence_summary.json` documents 120 recorded MLP final-estimator fits, none of which reached the 2,000-iteration cap. The 420 recorded warnings are `DeprecationWarning` notices; consult `reference_outputs_v2/warning_messages.csv` for their actual wording.

## Before making this repository public

- Confirm authorship, permission to release the jointly developed code, and an appropriate software license. **No license is supplied or implied by this ZIP.** Add a LICENSE file after the authors agree on its terms.
- Verify all results and supporting figures against the final submitted manuscript; do not mix original eight-model tables with Revision 2 outputs.
- Check the final paper's ethics, funding, competing-interest, AI-use and data/code-availability statements.
- Create a versioned GitHub release, then optionally archive that specific release on Zenodo for a DOI. Do not cite a DOI until one exists.

## Data provenance

UCI Heart Disease: https://archive.ics.uci.edu/dataset/45/heart+disease  
UCI Statlog Heart: https://archive.ics.uci.edu/dataset/145/statlog+heart

The manifest and independent SHA-256 comparison reports are included under `reference_outputs_v2/`; SHA-256 values identify **specific input bytes**, not a guarantee of clinical data quality or permission to redistribute datasets.
