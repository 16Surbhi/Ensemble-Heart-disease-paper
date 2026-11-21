#!/usr/bin/env python3
"""
Instance-level SHAP waterfall plots for the KNN/RF -> XGBoost stack on Cleveland (seed 42).

Reviewer 3 (Section 5) asked for waterfall plots of one true positive, true negative, false positive
and false negative case to complement the global beeswarm plot.

NO MODEL IS RETRAINED AND NO SHAP VALUE IS RECOMPUTED. The script re-uses the saved out-of-fold results
from the paper's run, so the waterfalls are exactly consistent with the beeswarm figure (Figure 3):

    shap_values_all_patients.csv   out-of-fold SHAP values, one row per Cleveland patient (index = row_index)
    out_of_fold_predictions.csv    out-of-fold predicted probabilities for every model
    processed.cleveland.data       raw UCI Cleveland file (needed only to display the feature values)

Waterfall baseline for a patient = (predicted probability) - (sum of that patient's SHAP values). This is the
mean prediction over the 50-record background of that patient's fold, so it differs slightly between folds
(about 0.42 to 0.53). The script checks that it is constant within each fold.

Case-selection rule (fixed in advance, no cherry-picking): at threshold 0.5, patients are grouped into
TP, TN, FP, FN. From each group, among patients with no missing predictor value, the patient whose predicted
probability is closest to the median predicted probability of that group is shown (ties: lowest row_index).

Usage (from the folder that contains the repo CSVs):
    python make_waterfalls.py --cleveland processed.cleveland.data
    python make_waterfalls.py --cleveland path/to/processed.cleveland.data --repo-dir path/to/repo --out waterfall_out
Requires: pandas numpy matplotlib shap
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

COLS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal', 'target']
CLEVELAND_SHA256 = 'a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8'  # from dataset_manifest.csv
MODEL = 'Stack_KNN_RF_XGBoost'
SEED = 42
THRESH = 0.5
GROUP_NAMES = {'TP': 'True positive', 'TN': 'True negative', 'FP': 'False positive', 'FN': 'False negative'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for b in iter(lambda: fp.read(65536), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cleveland', default='processed.cleveland.data')
    ap.add_argument('--repo-dir', default='.')
    ap.add_argument('--out', default='waterfall_out')
    ap.add_argument('--no-hash-check', action='store_true', help='testing only; never use for the paper figures')
    a = ap.parse_args()
    repo, out = Path(a.repo_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- raw data (display values only)
    raw_path = Path(a.cleveland)
    if not a.no_hash_check and sha256(raw_path) != CLEVELAND_SHA256:
        raise SystemExit('SHA-256 of the Cleveland file differs from the manifest. Use the original UCI file.')
    df = pd.read_csv(raw_path, header=None, names=COLS, na_values=['?', '-9', 'NA'])
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    X = df.drop(columns='target')
    y = (df.target > 0).astype(int)
    assert len(X) == 303, f'expected 303 Cleveland rows, found {len(X)}'

    # ---- saved out-of-fold results
    sv = pd.read_csv(repo / 'shap_values_all_patients.csv', index_col=0)
    oof = pd.read_csv(repo / 'out_of_fold_predictions.csv')
    d = oof[(oof.dataset == 'Cleveland') & (oof.seed == SEED) & (oof.model == MODEL)].set_index('row_index')
    assert set(d.index) == set(sv.index) == set(X.index), 'row indices do not line up'
    d = d.loc[sv.index]
    assert (d.y_true.values == y.loc[sv.index].values).all(), 'labels in the raw file do not match the saved results'
    assert list(sv.columns) == list(X.columns)

    d['pred'] = (d.p_positive >= THRESH).astype(int)
    d['shap_sum'] = sv.sum(axis=1).values
    d['base'] = d.p_positive - d.shap_sum
    spread = d.groupby('fold').base.std().max()
    assert spread < 1e-6, f'baseline is not constant within folds (max SD {spread})'
    d['group'] = np.select([(d.y_true == 1) & (d.pred == 1), (d.y_true == 0) & (d.pred == 0),
                            (d.y_true == 0) & (d.pred == 1)], ['TP', 'TN', 'FP'], 'FN')
    d['complete'] = ~X.loc[d.index].isna().any(axis=1).values

    print('Group sizes at threshold 0.5:', d.group.value_counts().to_dict())
    print('Patients with a missing predictor (excluded from selection):', sorted(d.index[~d.complete]))

    # ---- selection
    chosen = {}
    for g in ['TP', 'TN', 'FP', 'FN']:
        sub = d[d.group == g]
        med = sub.p_positive.median()
        cand = sub[sub.complete].copy()
        cand['dist'] = (cand.p_positive - med).abs()
        pick = cand.assign(ri=cand.index).sort_values(['dist', 'ri']).index[0]
        chosen[g] = pick
        print(f'{g}: n={len(sub)}, median p={med:.3f} -> row_index {pick} (fold {int(d.loc[pick, "fold"])}, '
              f'p={d.loc[pick, "p_positive"]:.3f}, true label={int(d.loc[pick, "y_true"])})')

    # ---- plots
    rows, pngs = [], {}
    for g, ri in chosen.items():
        e = shap.Explanation(values=sv.loc[ri].values.astype(float), base_values=float(d.loc[ri, 'base']),
                             data=X.loc[ri].values.astype(float), feature_names=list(X.columns))
        plt.figure(figsize=(8, 5.6))
        shap.plots.waterfall(e, max_display=13, show=False)
        fig = plt.gcf()
        fig.suptitle(f'{GROUP_NAMES[g]}: Cleveland row {ri} (fold {int(d.loc[ri, "fold"])}), '
                     f'true label {"disease" if d.loc[ri, "y_true"] == 1 else "no disease"}, '
                     f'predicted probability {d.loc[ri, "p_positive"]:.3f}', fontsize=9, y=1.0)
        fn = out / f'waterfall_{g}.png'
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close('all')
        pngs[g] = fn
        rec = {'group': g, 'row_index': ri, 'fold': int(d.loc[ri, 'fold']), 'y_true': int(d.loc[ri, 'y_true']),
               'p_positive': round(float(d.loc[ri, 'p_positive']), 4), 'baseline': round(float(d.loc[ri, 'base']), 4)}
        rec.update({f'x_{c}': X.loc[ri, c] for c in X.columns})
        rec.update({f'shap_{c}': round(float(sv.loc[ri, c]), 4) for c in X.columns})
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out / 'waterfall_cases.csv', index=False)

    # ---- 2x2 montage for the manuscript
    fig, axes = plt.subplots(2, 2, figsize=(16, 11.6))
    for ax, g in zip(axes.ravel(), ['TP', 'TN', 'FP', 'FN']):
        ax.imshow(plt.imread(pngs[g])); ax.axis('off')
        ax.text(0.0, 1.0, f'({"abcd"[["TP", "TN", "FP", "FN"].index(g)]})', transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='bottom')
    fig.tight_layout()
    fig.savefig(out / 'Figure_waterfall_cases.png', dpi=200, bbox_inches='tight')
    plt.close('all')
    print('\nWritten to', out.resolve())
    print(' waterfall_TP/TN/FP/FN.png, Figure_waterfall_cases.png (2x2 montage), waterfall_cases.csv')


if __name__ == '__main__':
    main()
