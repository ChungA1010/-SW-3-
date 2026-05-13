import os
import re
import pandas as pd
import numpy as np
from scipy import stats


def extract_intensity_num(row):
    s1 = str(row.get('intensity', ''))
    s2 = str(row.get('play_type_and_number', ''))
    m = re.search(r"(\d+)", s1)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", s2)
    if m:
        return int(m.group(1))
    return np.nan


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return np.nan
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_std


def main(csv_path=None):
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    csv_dir = os.path.join(project_root, 'csv')
    analysis_dir = os.path.join(csv_dir, 'feature_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    if csv_path is None:
        csv_path = os.path.join(csv_dir, 'features_pedalboarded_handmade.csv')

    df = pd.read_csv(csv_path)

    meta_cols = ['top_category', 'effect_kind', 'intensity', 'play_type_and_number']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    df['intensity_num'] = df.apply(extract_intensity_num, axis=1)
    df['is_clean'] = df['top_category'].str.lower() == 'clean'

    clean_df = df[df['is_clean'] == True]
    eff_df = df[df['is_clean'] == False]

    results_clean_vs_eff = []
    for feat in feature_cols:
        a = clean_df[feat].dropna()
        b = eff_df[feat].dropna()
        mean_clean = a.mean() if len(a) else np.nan
        mean_eff = b.mean() if len(b) else np.nan
        d = cohen_d(a.values, b.values) if len(a) and len(b) else np.nan
        tstat, pval = (np.nan, np.nan)
        if len(a) and len(b):
            try:
                tstat, pval = stats.ttest_ind(a, b, equal_var=False)
            except Exception:
                tstat, pval = (np.nan, np.nan)
        results_clean_vs_eff.append((feat, mean_clean, mean_eff, d, tstat, pval))

    df_clean_eff = pd.DataFrame(results_clean_vs_eff, columns=['feature', 'mean_clean', 'mean_effected', 'cohen_d', 'tstat', 'pval'])
    df_clean_eff['abs_cohen'] = df_clean_eff['cohen_d'].abs()
    df_clean_eff = df_clean_eff.sort_values('abs_cohen', ascending=False)
    df_clean_eff.to_csv(os.path.join(analysis_dir, 'analysis_clean_vs_effected.csv'), index=False)

    df_num = df.dropna(subset=['intensity_num']).copy()
    corr_results = []
    if not df_num.empty:
        for feat in feature_cols:
            x = df_num['intensity_num'].values
            y = df_num[feat].values
            try:
                rho, p = stats.spearmanr(x, y)
            except Exception:
                rho, p = (np.nan, np.nan)
            corr_results.append((feat, rho, p))
    df_corr = pd.DataFrame(corr_results, columns=['feature', 'spearman_rho', 'pval']).sort_values('spearman_rho', key=lambda s: s.abs(), ascending=False)
    df_corr.to_csv(os.path.join(analysis_dir, 'analysis_intensity_correlation.csv'), index=False)

    weak = df_num[df_num['intensity_num'] <= 25]
    strong = df_num[df_num['intensity_num'] >= 75]
    results_ws = []
    for feat in feature_cols:
        a = weak[feat].dropna()
        b = strong[feat].dropna()
        if len(a) and len(b):
            tstat, pval = stats.ttest_ind(a, b, equal_var=False)
            d = cohen_d(a.values, b.values)
        else:
            tstat, pval, d = (np.nan, np.nan, np.nan)
        results_ws.append((feat, len(a), len(b), d, tstat, pval))
    df_ws = pd.DataFrame(results_ws, columns=['feature', 'n_weak', 'n_strong', 'cohen_d', 'tstat', 'pval'])
    df_ws['abs_cohen'] = df_ws['cohen_d'].abs()
    df_ws = df_ws.sort_values('abs_cohen', ascending=False)
    df_ws.to_csv(os.path.join(analysis_dir, 'analysis_weak_vs_strong.csv'), index=False)

    print('\nTop features by effect size (clean vs effected):')
    print(df_clean_eff[['feature', 'mean_clean', 'mean_effected', 'cohen_d']].head(15).to_string(index=False))

    print('\nTop features by intensity correlation (abs rho):')
    print(df_corr.head(15).to_string(index=False))

    print('\nTop features by effect size (weak vs strong):')
    print(df_ws[['feature', 'n_weak', 'n_strong', 'cohen_d']].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
