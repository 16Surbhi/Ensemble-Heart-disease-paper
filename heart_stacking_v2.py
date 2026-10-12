#!/usr/bin/env python3
"""Heart-disease stacking benchmark, revision 2 (for the Discover AI resubmission).

Based on the original experiment script (same outer seeds 42/52/62, 5 folds,
inner 3-fold stacking, same fixed hyperparameters, same preprocessing).

What is NEW compared with the original run
  1. Two extra stacks:  Stack_KNN_RF_LR     (KNN+RF -> logistic-regression meta-learner)
                        Stack_KNN_RF_LRb_LR (KNN+RF+LR -> logistic-regression meta-learner)
  2. MLP meta-learner max_iter 350 -> 2000, with n_iter and convergence recorded for every fit
     (a full prespecified rerun instead of a post hoc check on 30 flagged fits).
  3. Every warning is logged by category per fit.
  4. Physiologically impossible zeros in chol / trestbps can be treated as missing
     (--zeros-as-missing, default ON; counts are written to the manifest so you can see whether it matters).
  5. Uncertainty: Nadeau-Bengio corrected resampled-t 95% CI and a paired patient-level
     bootstrap 95% CI for  model - logistic regression  (ROC-AUC).
  6. SHAP: ALL held-out patients in all 5 folds of seed 42 (each patient explained once, out of fold),
     many permutations, plus rank stability of mean|SHAP| across folds.
  7. Tables 3-7 style markdown output (mean +/- SD) written automatically.

Usage (Colab / local, with internet):
    pip install scikit-learn xgboost shap pandas numpy matplotlib scipy
    python heart_stacking_v2.py --full --shap
Quick smoke test (1 seed, 3 folds):
    python heart_stacking_v2.py --quick
Outputs go to ./outputs_v2
"""
from __future__ import annotations
import argparse, hashlib, json, platform, time, warnings
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import sklearn, xgboost
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MLP_MAX_ITER = 2000          # was 350 in the original run
COLS = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
        'oldpeak','slope','ca','thal','target']
CATEGORICAL = {'sex','cp','fbs','restecg','exang','slope','ca','thal'}
BASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
UCI = {'Cleveland': BASE+'heart-disease/processed.cleveland.data',
       'Hungarian': BASE+'heart-disease/processed.hungarian.data',
       'Swiss':     BASE+'heart-disease/processed.switzerland.data',
       'Statlog':   BASE+'statlog/heart/heart.dat'}
FILES = {'Cleveland':'processed.cleveland.data','Hungarian':'processed.hungarian.data',
         'Swiss':'processed.switzerland.data','Statlog':'heart.dat'}
SHA = {'Cleveland':'a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8',
       'Statlog':'f5f3b4204c285bafadd85cb735f38b47689f2be7047feb172dcbeab648110bf9',
       'Hungarian':'d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796',
       'Swiss':'834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314'}
MODEL_ORDER = ['DummyPrior','LogisticRegression','KNN','RandomForest','XGBoost',
               'Stack_KNN_RF_XGBoost','Stack_KNN_RF_MLP','Stack_Dual_XGBoost_MLP_Equal',
               'Stack_KNN_RF_LR','Stack_KNN_RF_LRb_LR']
LABEL = {'DummyPrior':'Dummy (class prior)','LogisticRegression':'Logistic regression','KNN':'KNN',
         'RandomForest':'Random forest','XGBoost':'XGBoost','Stack_KNN_RF_XGBoost':'Stack KNN/RF->XGBoost',
         'Stack_KNN_RF_MLP':'Stack KNN/RF->MLP','Stack_Dual_XGBoost_MLP_Equal':'Dual meta-fusion',
         'Stack_KNN_RF_LR':'Stack KNN/RF->LR','Stack_KNN_RF_LRb_LR':'Stack KNN/RF/LR->LR'}


# ----------------------------------------------------------------------------- data
def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for b in iter(lambda: fp.read(65536), b''):
            h.update(b)
    return h.hexdigest()


def get_file(name, data_dir, fetch=True):
    path = data_dir / FILES[name]
    if path.exists():
        return path
    if not fetch:
        raise FileNotFoundError(path)
    data_dir.mkdir(parents=True, exist_ok=True)
    req = Request(UCI[name], headers={'User-Agent': 'Mozilla/5.0 research-reproducibility'})
    with urlopen(req, timeout=60) as r:
        path.write_bytes(r.read())
    return path


def load_dataset(name, data_dir, fetch=True, check_hash=True, zeros_as_missing=True, drop_dups=False):
    path = get_file(name, data_dir, fetch)
    if check_hash and file_hash(path) != SHA[name]:
        raise ValueError(f'{name}: SHA-256 differs from the manuscript manifest - do not continue.')
    df = pd.read_csv(path, sep=r'\s+' if name == 'Statlog' else ',', header=None, names=COLS,
                     na_values=['?', '-9', 'NA'])
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    info = {'missing_cells_raw': int(df.drop(columns='target').isna().sum().sum()),
            'zero_chol': int((df.chol == 0).sum()), 'zero_trestbps': int((df.trestbps == 0).sum())}
    if zeros_as_missing:
        for c in ('chol', 'trestbps'):
            df.loc[df[c] == 0, c] = np.nan
    if name == 'Statlog':
        y = (df.target == 2).astype(int)
    else:
        y = (df.target > 0).astype(int)
    X = df.drop(columns='target')
    if drop_dups:
        keep = ~X.duplicated()
        X, y = X[keep], y[keep]
    info['missing_cells_used'] = int(X.isna().sum().sum())
    info['duplicate_feature_rows'] = int(X.duplicated().sum())
    return X.reset_index(drop=True), y.reset_index(drop=True), path, info


# ----------------------------------------------------------------------------- models
def preprocessor(X):
    cats = [c for c in X.columns if c in CATEGORICAL]
    nums = [c for c in X.columns if c not in cats]
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                          ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                          ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats)],
        remainder='drop', sparse_threshold=0)


class EqualWeightDualMeta(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_stack, mlp_stack):
        self.xgb_stack = xgb_stack
        self.mlp_stack = mlp_stack

    def fit(self, X, y):
        self.model_xgb_ = clone(self.xgb_stack).fit(X, y)
        self.model_mlp_ = clone(self.mlp_stack).fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return 0.5 * self.model_xgb_.predict_proba(X) + 0.5 * self.model_mlp_.predict_proba(X)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def build_models(X, seed, inner_splits=3):
    def pipe(est): return Pipeline([('preprocessing', preprocessor(X)), ('classifier', est)])
    def knn(): return pipe(KNeighborsClassifier(n_neighbors=7))
    def rf(): return pipe(RandomForestClassifier(n_estimators=120, min_samples_leaf=2, random_state=seed, n_jobs=1))
    def lr(): return pipe(LogisticRegression(max_iter=1500, random_state=seed))
    def xgb(): return XGBClassifier(n_estimators=70, max_depth=2, learning_rate=0.07, subsample=0.9,
                                    colsample_bytree=1.0, eval_metric='logloss', random_state=seed, n_jobs=1)
    def mlp(): return MLPClassifier(hidden_layer_sizes=(12, 6), alpha=0.05, max_iter=MLP_MAX_ITER, random_state=seed)
    def meta_lr(): return LogisticRegression(max_iter=1500, random_state=seed)
    def stack(estimators, meta):
        return StackingClassifier(estimators=estimators, final_estimator=meta, stack_method='predict_proba',
                                  passthrough=False, n_jobs=1,
                                  cv=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed))
    mx = stack([('KNN', knn()), ('RF', rf())], xgb())
    mn = stack([('KNN', knn()), ('RF', rf())], mlp())
    return {
        'DummyPrior': DummyClassifier(strategy='prior'),
        'LogisticRegression': lr(),
        'KNN': knn(),
        'RandomForest': rf(),
        'XGBoost': pipe(xgb()),
        'Stack_KNN_RF_XGBoost': mx,
        'Stack_KNN_RF_MLP': mn,
        'Stack_Dual_XGBoost_MLP_Equal': EqualWeightDualMeta(mx, mn),
        'Stack_KNN_RF_LR': stack([('KNN', knn()), ('RF', rf())], meta_lr()),
        'Stack_KNN_RF_LRb_LR': stack([('KNN', knn()), ('RF', rf()), ('LR', lr())], meta_lr()),
    }


def mlp_iterations(fitted, name):
    try:
        if name == 'Stack_KNN_RF_MLP':
            return int(fitted.final_estimator_.n_iter_)
        if name == 'Stack_Dual_XGBoost_MLP_Equal':
            return int(fitted.model_mlp_.final_estimator_.n_iter_)
    except AttributeError:
        pass
    return None


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0), 'recall': recall_score(y, pred, zero_division=0),
            'specificity': recall_score(y, pred, pos_label=0, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
            'roc_auc': roc_auc_score(y, prob), 'pr_auc': average_precision_score(y, prob),
            'brier': brier_score_loss(y, prob), 'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1])}


def run_one(name, X, y, seeds, folds, quick=False):
    rows, preds, warn_rows = [], [], []
    if y.value_counts().min() < folds:
        raise ValueError(f'{name}: minority class smaller than number of folds')
    for seed in seeds:
        for fold, (tr, te) in enumerate(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y), 1):
            xtr, xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
            if ytr.value_counts().min() < 3:
                raise ValueError(f'{name}: too few minority cases for inner 3-fold stacking')
            for mname, model in build_models(xtr, seed).items():
                t0 = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(xtr, ytr)
                t_fit = time.perf_counter() - t0
                t0 = time.perf_counter()
                p = model.predict_proba(xte)[:, 1]
                t_pred = time.perf_counter() - t0
                if not np.isfinite(p).all():
                    raise ValueError('non-finite predictions')
                cats = Counter(w.category.__name__ for w in caught)
                it = mlp_iterations(model, mname)
                rows.append({'dataset': name, 'seed': seed, 'fold': fold, 'model': mname, 'n_train': len(tr),
                             'n_test': len(te), 'fit_seconds': t_fit, 'predict_seconds': t_pred,
                             'warning_count': len(caught), 'convergence_warnings': cats.get('ConvergenceWarning', 0),
                             'mlp_n_iter': it, 'mlp_converged': (None if it is None else it < MLP_MAX_ITER),
                             **metrics(yte, p)})
                for w in caught:
                    warn_rows.append({'dataset': name, 'seed': seed, 'fold': fold, 'model': mname,
                                      'category': w.category.__name__, 'message': str(w.message)[:200]})
                preds.extend({'dataset': name, 'seed': seed, 'fold': fold, 'model': mname,
                              'row_index': int(te[k]), 'y_true': int(yte.iloc[k]), 'p_positive': float(p[k])}
                             for k in range(len(te)))
            print(f'{name:10} seed={seed} fold={fold}/{folds} done', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(preds), pd.DataFrame(warn_rows)


# ----------------------------------------------------------------------------- statistics
def nb_corrected_ci(diff, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t interval for the mean paired difference.
    Heuristic for repeated k-fold; treat as an approximate uncertainty guide, not an exact test."""
    d = np.asarray(diff, float)
    J = len(d)
    se = np.sqrt((1.0 / J + n_test / n_train) * d.var(ddof=1))
    t = stats.t.ppf(0.975, J - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se


def paired_stats(fold_df, oof, ref='LogisticRegression', n_boot=2000, seed=0):
    out = []
    rng = np.random.default_rng(seed)
    for ds, d in fold_df.groupby('dataset'):
        piv = d.pivot_table(index=['seed', 'fold'], columns='model', values='roc_auc')
        ntr, nte = d.n_train.mean(), d.n_test.mean()
        o = oof[oof.dataset == ds]
        seeds = sorted(o.seed.unique())
        y = (o[(o.model == ref) & (o.seed == seeds[0])].sort_values('row_index').y_true.values)
        n = len(y)
        P = {}
        for m in MODEL_ORDER:
            if m in ('DummyPrior', ref):
                continue
            P[m] = np.stack([o[(o.model == m) & (o.seed == s)].sort_values('row_index').p_positive.values for s in seeds])
        PR = np.stack([o[(o.model == ref) & (o.seed == s)].sort_values('row_index').p_positive.values for s in seeds])
        idxs = [rng.integers(0, n, n) for _ in range(n_boot)]
        for m in P:
            diff = (piv[m] - piv[ref]).values
            mean, lo, hi = nb_corrected_ci(diff, ntr, nte)
            bs = []
            for idx in idxs:
                yb = y[idx]
                if yb.min() == yb.max():
                    continue
                a = np.mean([roc_auc_score(yb, P[m][r][idx]) for r in range(len(seeds))])
                b = np.mean([roc_auc_score(yb, PR[r][idx]) for r in range(len(seeds))])
                bs.append(a - b)
            out.append({'dataset': ds, 'model': m, 'reference': ref, 'mean_delta_auc': mean,
                        'NB_ci_low': lo, 'NB_ci_high': hi,
                        'boot_ci_low': np.percentile(bs, 2.5), 'boot_ci_high': np.percentile(bs, 97.5),
                        'n_boot_valid': len(bs), 'frac_folds_model_better': float((diff > 0).mean())})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- tables & plots
def write_tables(fold, outdir):
    lines = []
    cols = [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Bal. acc.'), ('f1', 'F1'),
            ('roc_auc', 'ROC-AUC'), ('pr_auc', 'PR-AUC'), ('brier', 'Brier')]
    for ds, d in fold.groupby('dataset', sort=False):
        lines += [f'\n### {ds}\n', '| Model | ' + ' | '.join(c[1] for c in cols) + ' |',
                  '|---|' + '---|' * len(cols)]
        for m in MODEL_ORDER:
            g = d[d.model == m]
            if g.empty:
                continue
            lines.append(f'| {LABEL[m]} | ' + ' | '.join(f'{g[c].mean():.3f} ± {g[c].std():.3f}' for c, _ in cols) + ' |')
    lines += ['\n### Table 7 style: mean ROC-AUC of the stacking variants\n',
              '| Cohort | XGB stack | MLP stack | Dual | LR-meta stack | LR-base+LR-meta | Logistic regression |',
              '|---|---|---|---|---|---|---|']
    for ds, d in fold.groupby('dataset', sort=False):
        mm = d.groupby('model').roc_auc.mean()
        lines.append(f'| {ds} | ' + ' | '.join(f'{mm[m]:.3f}' for m in
                     ['Stack_KNN_RF_XGBoost', 'Stack_KNN_RF_MLP', 'Stack_Dual_XGBoost_MLP_Equal',
                      'Stack_KNN_RF_LR', 'Stack_KNN_RF_LRb_LR', 'LogisticRegression']) + ' |')
    (outdir / 'tables_mean_sd.md').write_text('\n'.join(lines))


def make_plots(fold, oof, outdir):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (ds, d) in zip(axes.ravel(), fold.groupby('dataset', sort=False)):
        g = d.groupby('model').roc_auc.agg(['mean', 'std']).reindex([m for m in MODEL_ORDER if m in set(d.model)])
        stack = g.index.str.startswith('Stack')
        ax.barh([LABEL[m] for m in g.index], g['mean'], xerr=g['std'], capsize=2,
                color=['#2a9d8f' if s else '#4a5f7a' for s in stack])
        ax.set_xlim(0.35, 1.02); ax.invert_yaxis(); ax.set_title(ds)
        ax.set_xlabel('Mean held-out ROC-AUC (bars: fold SD, descriptive)')
    fig.text(0.5, 0.005, 'Teal = stacked models; grey = single models and class-prior baseline', ha='center')
    fig.tight_layout(); fig.savefig(outdir / 'Figure1_roc_auc_by_cohort.png', dpi=300); plt.close(fig)
    seed0 = sorted(oof.seed.unique())[0]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (ds, d) in zip(axes.ravel(), oof[oof.seed == seed0].groupby('dataset', sort=False)):
        for m, g in d.groupby('model'):
            f, t, _ = roc_curve(g.y_true, g.p_positive)
            ax.plot(f, t, lw=1.4, label=f'{LABEL[m]} ({roc_auc_score(g.y_true, g.p_positive):.3f})')
        ax.plot([0, 1], [0, 1], '--', c='gray'); ax.set_title(f'{ds}: pooled out-of-fold ROC (seed {seed0})')
        ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate'); ax.legend(fontsize=6, loc='lower right')
    fig.tight_layout(); fig.savefig(outdir / 'Figure2_pooled_oof_roc.png', dpi=300); plt.close(fig)


# ----------------------------------------------------------------------------- SHAP
def run_shap(X, y, outdir, seed=42, n_perm=10, n_bg=50):
    """Full-pipeline permutation SHAP for the KNN/RF->XGBoost stack on ALL patients, out of fold."""
    import shap, matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    M = X.shape[1]
    vals, data, idxs, fold_imp = [], [], [], []
    for k, (tr, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y), 1):
        model = build_models(X.iloc[tr], seed)['Stack_KNN_RF_XGBoost']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); model.fit(X.iloc[tr], y.iloc[tr])
        bg = X.iloc[tr].sample(min(n_bg, len(tr)), random_state=seed)
        f = lambda d: model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1]
        expl = shap.Explainer(f, shap.maskers.Independent(bg), algorithm='permutation', feature_names=list(X.columns))
        sv = expl(X.iloc[te], max_evals=2 * M * n_perm + 1)
        vals.append(sv.values); data.append(X.iloc[te].values); idxs.extend(te.tolist())
        fold_imp.append(np.abs(sv.values).mean(0))
        print(f'SHAP fold {k}/5 done ({len(te)} patients)', flush=True)
    V, D = np.vstack(vals), np.vstack(data)
    pd.DataFrame(V, columns=X.columns, index=idxs).to_csv(outdir / 'shap_values_all_patients.csv')
    imp = pd.DataFrame(fold_imp, columns=X.columns, index=[f'fold{i}' for i in range(1, 6)]).T
    imp['mean_abs_shap_all'] = np.abs(V).mean(0)
    imp.sort_values('mean_abs_shap_all', ascending=False).to_csv(outdir / 'shap_mean_abs_by_fold.csv')
    rho = pd.DataFrame(fold_imp, index=range(5)).T.corr(method='spearman').values
    iu = np.triu_indices(5, 1)
    (outdir / 'shap_rank_stability.txt').write_text(
        f'Mean pairwise Spearman correlation of mean|SHAP| feature ranks across the 5 folds: {rho[iu].mean():.3f} '
        f'(min {rho[iu].min():.3f}); n_permutations={n_perm}, background={n_bg}, patients={len(idxs)}\n')
    plt.figure(figsize=(8, 6))
    shap.plots.beeswarm(shap.Explanation(V, data=D, feature_names=list(X.columns)), show=False, max_display=13)
    plt.title('Cleveland: out-of-fold full-pipeline SHAP (permutation), all patients')
    plt.xlabel('SHAP value (change in predicted probability of disease)')
    plt.tight_layout(); plt.savefig(outdir / 'Figure3_Cleveland_SHAP_all_patients.png', dpi=300, bbox_inches='tight')
    plt.close('all')


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--quick', action='store_true'); g.add_argument('--full', action='store_true')
    ap.add_argument('--data-dir', default='data'); ap.add_argument('--output-dir', default='outputs_v2')
    ap.add_argument('--datasets', nargs='*', default=['Cleveland', 'Statlog', 'Hungarian', 'Swiss'])
    ap.add_argument('--no-zeros-as-missing', action='store_true', help='keep chol/trestbps zeros as values (original behaviour)')
    ap.add_argument('--drop-duplicates', action='store_true', help='drop duplicated feature rows (Hungarian has 1)')
    ap.add_argument('--no-hash-check', action='store_true', help='testing only')
    ap.add_argument('--no-fetch', action='store_true')
    ap.add_argument('--shap', action='store_true'); ap.add_argument('--shap-perms', type=int, default=10)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    seeds, folds = ([42], 3) if a.quick else ([42, 52, 62], 5)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    R, P, W, MAN = [], [], [], []
    for name in a.datasets:
        X, y, path, info = load_dataset(name, Path(a.data_dir), not a.no_fetch, not a.no_hash_check,
                                        not a.no_zeros_as_missing, a.drop_duplicates)
        MAN.append({'dataset': name, 'file': str(path), 'sha256': file_hash(path), 'n': len(X),
                    'positive': int(y.sum()), 'negative': int((1 - y).sum()), **info})
        print(f'LOADED {name} {X.shape} {y.value_counts().to_dict()} {info}', flush=True)
        r, p, w = run_one(name, X, y, seeds, folds, a.quick)
        R.append(r); P.append(p); W.append(w)
    fold = pd.concat(R, ignore_index=True); oof = pd.concat(P, ignore_index=True)
    warn = pd.concat(W, ignore_index=True)
    fold.to_csv(out / 'fold_level_metrics.csv', index=False)
    oof.to_csv(out / 'out_of_fold_predictions.csv', index=False)
    warn.to_csv(out / 'warning_messages.csv', index=False)
    pd.DataFrame(MAN).to_csv(out / 'dataset_manifest.csv', index=False)
    keys = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc',
            'brier', 'fit_seconds', 'predict_seconds']
    s = fold.groupby(['dataset', 'model'], sort=False)[keys].agg(['mean', 'std']).round(4)
    s.columns = ['_'.join(c) for c in s.columns]; s.to_csv(out / 'summary_mean_sd.csv')
    mlp = fold[fold.mlp_n_iter.notna()]
    conv = {'mlp_fits': int(len(mlp)), 'not_converged_at_cap': int((~mlp.mlp_converged.astype(bool)).sum()),
            'max_iter_used': int(mlp.mlp_n_iter.max()) if len(mlp) else None,
            'warning_categories': dict(Counter(warn.category)) if len(warn) else {}}
    (out / 'convergence_summary.json').write_text(json.dumps(conv, indent=2))
    write_tables(fold, out)
    paired_stats(fold, oof, n_boot=(200 if a.quick else a.boot)).to_csv(out / 'paired_vs_logistic_regression.csv', index=False)
    make_plots(fold, oof, out)
    if a.shap:
        X, y, _, _ = load_dataset('Cleveland', Path(a.data_dir), False, not a.no_hash_check,
                                  not a.no_zeros_as_missing, a.drop_duplicates)
        run_shap(X, y, out, n_perm=a.shap_perms)
    (out / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__,
        'pandas': pd.__version__, 'numpy': np.__version__, 'seeds': seeds, 'outer_folds': folds, 'inner_folds': 3,
        'mlp_max_iter': MLP_MAX_ITER, 'zeros_as_missing': not a.no_zeros_as_missing,
        'drop_duplicates': a.drop_duplicates, 'original_run_env': 'Python 3.13.16, sklearn 1.6.1, xgboost 3.4.1, numpy 2.1.3, pandas 2.2.3'},
        indent=2))
    print('\nDONE ->', out)


if __name__ == '__main__':
    main()
