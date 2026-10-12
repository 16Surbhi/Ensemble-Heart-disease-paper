#!/usr/bin/env python3
"""Independently validate a saved Revision 2 results directory.

Usage: python verification/check_result_integrity.py reference_outputs_v2
This checks internal consistency; it does not re-train models or validate their clinical utility.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score,
)

EXPECTED = ['Cleveland', 'Statlog', 'Hungarian', 'Swiss']
MEASURES = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity',
            'f1', 'roc_auc', 'pr_auc', 'brier', 'tn', 'fp', 'fn', 'tp']


def independently_recompute(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'specificity': recall_score(y, pred, pos_label=0, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, prob),
        'pr_auc': average_precision_score(y, prob),
        'brier': brier_score_loss(y, prob),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def check(directory: Path, tolerance: float = 1e-8):
    fold = pd.read_csv(directory / 'fold_level_metrics.csv')
    oof = pd.read_csv(directory / 'out_of_fold_predictions.csv')
    manifest = pd.read_csv(directory / 'dataset_manifest.csv')
    warnings_df = pd.read_csv(directory / 'warning_messages.csv')
    assert set(fold.dataset) == set(EXPECTED), 'Unexpected or missing dataset'
    assert len(fold) == 600, f'Expected 600 model–fold rows, found {len(fold)}'
    models = sorted(fold.model.unique())
    assert len(models) == 10, f'Expected 10 models, found {len(models)}'
    assert set(fold.seed.unique()) == {42, 52, 62}
    assert set(fold.fold.unique()) == {1, 2, 3, 4, 5}
    assert len(fold.groupby(['dataset', 'seed', 'fold', 'model'])) == len(fold), 'Duplicate fold metric key'
    assert len(oof.groupby(['dataset', 'seed', 'fold', 'model', 'row_index'])) == len(oof), 'Duplicate prediction key'
    assert np.isfinite(oof.p_positive).all() and oof.p_positive.between(0, 1).all()
    assert set(oof.y_true.unique()) == {0, 1}
    assert set(manifest.dataset) == set(EXPECTED)

    checked = 0
    deviations = []
    out_groups = dict(tuple(oof.groupby(['dataset', 'seed', 'fold', 'model'])))
    for row in fold.itertuples(index=False):
        key = (row.dataset, row.seed, row.fold, row.model)
        if key not in out_groups:
            raise AssertionError(f'Missing prediction group: {key}')
        p = out_groups[key]
        if len(p) != row.n_test:
            raise AssertionError(f'Wrong test set size in {key}: {len(p)} vs {row.n_test}')
        if p.y_true.nunique() != 2:
            raise AssertionError(f'ROC-AUC undefined (single class) in {key}')
        actual = independently_recompute(p.y_true.to_numpy(), p.p_positive.to_numpy())
        for metric, value in actual.items():
            expected = float(getattr(row, metric))
            diff = abs(float(value) - expected)
            if diff > tolerance:
                deviations.append((key, metric, expected, float(value), diff))
            checked += 1
    if deviations:
        print('MISMATCHES (first 12):', deviations[:12], file=sys.stderr)
        raise AssertionError(f'{len(deviations)} metric discrepancies')

    for ds in EXPECTED:
        n = int(manifest.loc[manifest.dataset == ds, 'n'].iloc[0])
        for seed in (42, 52, 62):
            for model in models:
                g = oof[(oof.dataset == ds) & (oof.seed == seed) & (oof.model == model)]
                if len(g) != n or not np.array_equal(np.sort(g.row_index.values), np.arange(n)):
                    raise AssertionError(f'Incomplete or duplicated held-out record coverage: {ds} {seed} {model}')

    summary = json.loads((directory / 'convergence_summary.json').read_text())
    assert summary.get('not_converged_at_cap') == 0
    actual_conv = int(fold.convergence_warnings.fillna(0).sum())
    assert actual_conv == 0
    if len(warnings_df):
        assert set(warnings_df.category) == {'DeprecationWarning'}, 'Unexpected saved warning category'
    print(f'PASS: {len(fold)} model-fold entries, {len(oof):,} held-out prediction records, {checked:,} independently verified metric values')
    print('Datasets:', ', '.join(EXPECTED))
    print('Models:', len(models), 'Seeds:', 3, 'Outer folds:', 5)
    print('Recorded fitting convergence warnings:', actual_conv,
          '| Other saved warnings:', len(warnings_df))
    print('NOTE: This checks saved internal consistency, not a fresh experiment or external validation.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', nargs='?', default='reference_outputs_v2', type=Path)
    args = p.parse_args()
    check(args.directory)


if __name__ == '__main__':
    main()
