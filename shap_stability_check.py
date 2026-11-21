#!/usr/bin/env python3
"""
Stability check for the permutation-SHAP explanations of the KNN/RF -> XGBoost stack (Cleveland, seed 42).

Reviewer 3 (5.5, 5.6) asked whether the explanations were checked for stability with respect to the number of
permutations and the background sample, and whether LOCAL (patient-level) explanations are stable, not only the
global ranking. Reviewer 1 (Comment 6) asked for the 50-record background / 10-permutation limitation to be
acknowledged.

What this script does
  * Rebuilds exactly the same five outer folds and the same fitted KNN/RF -> XGBoost model per fold as the
    paper's SHAP run (same seed, same code from heart_stacking_v2.py).
  * Re-explains the held-out patients with other settings (number of permutations, background size, random seed
    of the permutation sampler) and compares them with the SHAP values saved from the paper's run
    (shap_values_all_patients.csv; 10 permutations, 50-record background).
  * Reports global stability (rank correlation of mean |SHAP|, top-5 overlap) and local stability
    (per-patient rank correlation of |SHAP| across the 13 features, agreement of the top feature and top-3 set,
    agreement of the direction of the top feature, mean absolute difference).

Note: the paper's SHAP run did not fix the seed of the permutation sampler (shap re-seeds from system entropy when
seed=None), so it cannot be reproduced bit for bit. The 'repeat' configuration below (same settings, new seed)
measures that sampling noise directly. All runs here use fixed seeds.

Usage (put this file next to heart_stacking_v2.py and the repo CSVs):
    python shap_stability_check.py --data-dir data --n-patients 60
    python shap_stability_check.py --data-dir data --n-patients 0            # all 303 patients (slow)
    python shap_stability_check.py --data-dir data --configs 10x50 30x50 100x50 10x100 --cases waterfall_out/waterfall_cases.csv
Config format: <n_permutations>x<background_size>. Needs the Cleveland file in --data-dir
(it is downloaded and SHA-256 checked by heart_stacking_v2.load_dataset unless --no-fetch is given).
Requires: pandas numpy scipy scikit-learn xgboost shap
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
import shap

MODEL = 'Stack_KNN_RF_XGBoost'
SEED = 42


def parse_cfg(c):
    p, b = c.lower().split('x')
    return int(p), int(b)


def local_metrics(A, B, names):
    """A, B: (patients x features) SHAP arrays for the same patients."""
    absA, absB = np.abs(A), np.abs(B)
    rho = np.array([spearmanr(absA[i], absB[i])[0] for i in range(len(A))])
    top1 = (absA.argmax(1) == absB.argmax(1))
    top3 = np.array([set(np.argsort(-absA[i])[:3]) == set(np.argsort(-absB[i])[:3]) for i in range(len(A))])
    j = np.arange(len(A))
    sign_same = np.sign(A[j, absA.argmax(1)]) == np.sign(B[j, absA.argmax(1)])
    mad = np.abs(A - B).mean(1)
    return rho, top1, top3, sign_same, mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--repo-dir', default='.')
    ap.add_argument('--out', default='stability_out')
    ap.add_argument('--configs', nargs='*', default=['10x50', '30x50', '100x50', '10x100'])
    ap.add_argument('--n-patients', type=int, default=60, help='0 = all 303 patients')
    ap.add_argument('--cases', default=None, help='waterfall_cases.csv; its four patients are always included')
    ap.add_argument('--no-fetch', action='store_true')
    ap.add_argument('--no-hash-check', action='store_true', help='testing only')
    ap.add_argument('--sampler-seed', type=int, default=1000)
    a = ap.parse_args()

    repo = Path(a.repo_dir).resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import heart_stacking_v2 as H

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, y, _, _ = H.load_dataset('Cleveland', Path(a.data_dir), not a.no_fetch, not a.no_hash_check, True, False)
    M = X.shape[1]
    paper = pd.read_csv(repo / 'shap_values_all_patients.csv', index_col=0)
    oof = pd.read_csv(repo / 'out_of_fold_predictions.csv')
    d = oof[(oof.dataset == 'Cleveland') & (oof.seed == SEED) & (oof.model == MODEL)].set_index('row_index')
    saved_p = d.p_positive
    saved_base = (d.p_positive - paper.sum(axis=1).reindex(d.index))

    # ---- patient subset (fixed rule: seeded random sample, plus the four waterfall cases if given)
    rng = np.random.default_rng(0)
    all_idx = np.array(sorted(paper.index))
    if a.n_patients and a.n_patients < len(all_idx):
        keep = set(rng.choice(all_idx, size=a.n_patients, replace=False).tolist())
    else:
        keep = set(all_idx.tolist())
    if a.cases:
        keep |= set(pd.read_csv(a.cases).row_index.tolist())
    keep = sorted(keep)
    keep_set = set(keep)
    print(f'Explaining {len(keep)} of {len(all_idx)} patients per configuration.', flush=True)

    cfgs = [parse_cfg(c) for c in a.configs]
    # 'repeat' of the paper's own settings with a new seed is always included as the noise floor
    names = []
    for (p, b) in cfgs:
        names.append(f'{p}x{b}')
    results = {n: {} for n in names}
    timing = {n: 0.0 for n in names}
    checks = []

    for k, (tr, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y), 1):
        te_keep = [i for i in te if i in keep_set]
        if not te_keep:
            continue
        model = H.build_models(X.iloc[tr], SEED)[MODEL]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(X.iloc[tr], y.iloc[tr])
        f = lambda dd: model.predict_proba(pd.DataFrame(dd, columns=X.columns))[:, 1]
        pdiff = np.abs(model.predict_proba(X.iloc[te])[:, 1] - saved_p.loc[te].values).max()
        checks.append((k, 'model reproduces saved predictions, max |diff|', float(pdiff)))
        for (p, b), name in zip(cfgs, names):
            bg = X.iloc[tr].sample(min(b, len(tr)), random_state=SEED)
            t0 = time.perf_counter()
            expl = shap.Explainer(f, shap.maskers.Independent(bg), algorithm='permutation',
                                  feature_names=list(X.columns), seed=a.sampler_seed + p * 7 + b)
            sv = expl(X.iloc[te_keep], max_evals=2 * M * p + 1)
            timing[name] += time.perf_counter() - t0
            results[name].update({ri: v for ri, v in zip(te_keep, sv.values)})
            if b == 50:   # identical background to the paper -> identical baseline expected
                bdiff = np.abs(np.asarray(sv.base_values) - saved_base.loc[te_keep].values).max()
                checks.append((k, f'{name}: baseline equals paper baseline, max |diff|', float(bdiff)))
        print(f'fold {k}/5 done ({len(te_keep)} patients)', flush=True)

    # ---- assemble
    R = {'paper(10x50, unseeded)': paper.loc[keep]}
    for n in names:
        df = pd.DataFrame.from_dict(results[n], orient='index', columns=X.columns).loc[keep]
        df.to_csv(out / f'shap_values_{n}.csv')
        R[n] = df
    ref_name = max(names, key=lambda n: parse_cfg(n)[0] * parse_cfg(n)[1])
    pairs = [(n, 'paper(10x50, unseeded)') for n in names] + [(n, ref_name) for n in names if n != ref_name] \
        + [('paper(10x50, unseeded)', ref_name)]

    rows, plevel = [], []
    for (n1, n2) in pairs:
        A, B = R[n1].values, R[n2].values
        g1, g2 = np.abs(A).mean(0), np.abs(B).mean(0)
        grho = spearmanr(g1, g2)[0]
        top5 = len(set(np.argsort(-g1)[:5]) & set(np.argsort(-g2)[:5])) / 5
        rho, top1, top3, sgn, mad = local_metrics(A, B, list(X.columns))
        rows.append({'run_A': n1, 'run_B': n2, 'patients': len(A),
                     'global_spearman_mean_abs_shap': round(float(grho), 3), 'global_top5_overlap': round(top5, 2),
                     'local_spearman_median': round(float(np.median(rho)), 3),
                     'local_spearman_p10': round(float(np.percentile(rho, 10)), 3),
                     'local_spearman_min': round(float(rho.min()), 3),
                     'same_top_feature_pct': round(100 * top1.mean(), 1), 'same_top3_set_pct': round(100 * top3.mean(), 1),
                     'same_direction_of_top_feature_pct': round(100 * sgn.mean(), 1),
                     'mean_abs_diff_per_feature': round(float(np.mean(mad)), 4)})
        for i, ri in enumerate(keep):
            plevel.append({'run_A': n1, 'run_B': n2, 'row_index': ri, 'spearman_abs_shap': rho[i],
                           'same_top_feature': bool(top1[i]), 'same_top3_set': bool(top3[i]), 'mean_abs_diff': mad[i]})
    S = pd.DataFrame(rows)
    S.to_csv(out / 'shap_stability_summary.csv', index=False)
    pd.DataFrame(plevel).to_csv(out / 'shap_stability_patient_level.csv', index=False)
    pd.DataFrame(checks, columns=['fold', 'check', 'value']).to_csv(out / 'sanity_checks.csv', index=False)

    ranking = pd.DataFrame({k: np.abs(v.values).mean(0) for k, v in R.items()}, index=X.columns)
    ranking.round(4).to_csv(out / 'global_mean_abs_shap_by_run.csv')

    with open(out / 'shap_stability_report.txt', 'w') as fh:
        fh.write(f'Patients explained: {len(keep)} (fixed seeded sample plus waterfall cases)\n')
        fh.write('Runtime (s) per configuration: ' + ', '.join(f'{n}: {t:.0f}' for n, t in timing.items()) + '\n\n')
        fh.write(S.to_string(index=False) + '\n\nSanity checks\n')
        sc = pd.DataFrame(checks, columns=['fold', 'check', 'value'])
        fh.write(sc.groupby('check').value.max().to_string() + '\n')
        fh.write('\nMean |SHAP| by run:\n' + ranking.round(3).sort_values(ranking.columns[0], ascending=False).to_string() + '\n')
    print('\n' + (out / 'shap_stability_report.txt').read_text())


if __name__ == '__main__':
    main()
